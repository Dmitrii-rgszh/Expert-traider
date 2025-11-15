from __future__ import annotations

import argparse
import math
import json
from datetime import datetime, timezone
import os
import sys
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision, BinaryPrecision, BinaryRecall
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
existing_pythonpath = os.environ.get("PYTHONPATH")
if existing_pythonpath:
    if str(REPO_ROOT) not in existing_pythonpath.split(os.pathsep):
        os.environ["PYTHONPATH"] = f"{REPO_ROOT}{os.pathsep}{existing_pythonpath}"
else:
    os.environ["PYTHONPATH"] = str(REPO_ROOT)

if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("medium")

try:  # optional dependency
    import mlflow
except ModuleNotFoundError:  # pragma: no cover - optional
    mlflow = None

META_COLUMNS = {
    "secid",
    "timeframe",
    "feature_set",
    "label_set",
    "signal_time",
    "horizon_minutes",
    "forward_return_pct",
    "max_runup_pct",
    "max_drawdown_pct",
    "label_long",
    "label_short",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train baseline temporal models on the prepared dataset")
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=Path("data/training/baseline_dataset.csv"),
        help="Path to the merged features+labels CSV",
    )
    parser.add_argument("--seq-len", type=int, default=32, help="Sliding window size in steps")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size")
    parser.add_argument("--max-epochs", type=int, default=15, help="Training epochs")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Adam learning rate")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Portion of rows for training")
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Portion of rows for validation (rest is test)",
    )
    parser.add_argument("--num-workers", type=int, default=0, help="Dataloader workers")
    parser.add_argument("--gradient-clip", type=float, default=1.0, help="Gradient clipping value")
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Seed for torch / numpy to keep chronological splits reproducible",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("logs/temporal_cnn"),
        help="Directory to store TensorBoard / CSV logs and checkpoints",
    )
    parser.add_argument(
        "--logger",
        choices=["csv", "tensorboard", "both"],
        default="both",
        help="Monitoring backend to enable",
    )
    parser.add_argument(
        "--accelerator",
        type=str,
        default="auto",
        help="Lightning accelerator (gpu, cpu, auto, etc.)",
    )
    parser.add_argument("--devices", type=int, default=1, help="Number of devices to use")
    parser.add_argument(
        "--precision",
        type=str,
        default="32-true",
        help="Floating point precision setting such as '16-mixed' or '32-true'",
    )
    parser.add_argument("--model-type", choices=["tcn", "tft"], default="tcn", help="Which architecture to train")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden size for transformer/gru blocks")
    parser.add_argument("--attn-heads", type=int, default=4, help="Attention heads for TFT model")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout applied inside the model")
    parser.add_argument(
        "--plot-roc",
        action="store_true",
        help="Save ROC curve using test predictions after training",
    )
    parser.add_argument("--mlflow-tracking-uri", help="Optional MLflow tracking URI")
    parser.add_argument("--mlflow-experiment", help="MLflow experiment name (default: TemporalCNN)")
    parser.add_argument("--mlflow-run-name", help="Custom MLflow run name")
    parser.add_argument(
        "--mlflow-tags",
        nargs="*",
        help="MLflow tags in key=value format",
    )
    parser.add_argument(
        "--walk-forward-json",
        type=Path,
        help="Optional JSON file describing walk-forward splits (list of train/val/test ranges)",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("docs/modeling/train_runs"),
        help="Directory where JSON training reports will be stored",
    )
    return parser.parse_args()


class SlidingWindowDataset(Dataset):
    def __init__(self, sequences: torch.Tensor, labels: torch.Tensor) -> None:
        self.sequences = sequences
        self.labels = labels

    def __len__(self) -> int:  # noqa: D401 - tiny helper
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.sequences[idx], self.labels[idx]


class BaselineDataModule(pl.LightningDataModule):
    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        seq_len: int,
        batch_size: int,
        train_ratio: float,
        val_ratio: float,
        num_workers: int,
        split_config: Optional[dict[str, tuple[pd.Timestamp, pd.Timestamp]]] = None,
        min_val_fraction: float = 0.1,
    ) -> None:
        super().__init__()
        self.df = df
        self.feature_cols = feature_cols
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.num_workers = num_workers
        self.split_config = split_config
        self.min_val_fraction = min_val_fraction
        self.train_ds: Optional[SlidingWindowDataset] = None
        self.val_ds: Optional[SlidingWindowDataset] = None
        self.test_ds: Optional[SlidingWindowDataset] = None
        self.train_pos_weight: float = 1.0
        self.train_df: Optional[pd.DataFrame] = None
        self.val_df: Optional[pd.DataFrame] = None
        self.test_df: Optional[pd.DataFrame] = None

    def setup(self, stage: str | None = None) -> None:  # noqa: D401 - Lightning hook
        df_sorted = self.df.sort_values("signal_time").reset_index(drop=True)
        if self.split_config:
            train_df = self._slice_range(df_sorted, self.split_config.get("train"))
            val_window = self.split_config.get("val")
            val_df = self._slice_range(df_sorted, val_window)
            if val_df.empty:
                tail = max(1, int(len(train_df) * self.min_val_fraction))
                val_df = train_df.iloc[-tail:].copy()
                train_df = train_df.iloc[:-tail].copy()
            test_df = self._slice_range(df_sorted, self.split_config.get("test"))
            if test_df.empty:
                raise ValueError("Walk-forward split produced empty test set. Check split ranges.")
        else:
            total = len(df_sorted)
            train_end = max(self.seq_len, int(total * self.train_ratio))
            val_end = max(train_end + self.seq_len, int(total * (self.train_ratio + self.val_ratio)))
            train_df = df_sorted.iloc[:train_end].copy()
            val_df = df_sorted.iloc[train_end:val_end].copy()
            test_df = df_sorted.iloc[val_end:].copy()
            if test_df.empty:
                test_df = df_sorted.iloc[-math.ceil(total * 0.15) :].copy()
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.train_ds = self._build_dataset(train_df)
        self.val_ds = self._build_dataset(val_df)
        self.test_ds = self._build_dataset(test_df)
        if len(self.train_ds) == 0:
            raise ValueError("Training split produced zero sequences. Reduce --seq-len or widen train window.")
        pos = float(self.train_ds.labels.sum().item())
        neg = float(len(self.train_ds) - pos)
        self.train_pos_weight = neg / pos if pos > 0 else 1.0

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def _build_dataset(self, df_subset: pd.DataFrame) -> SlidingWindowDataset:
        sequences: list[np.ndarray] = []
        labels: list[float] = []
        grouped = df_subset.groupby("secid")
        for _, group in grouped:
            group = group.sort_values("signal_time")
            if len(group) < self.seq_len:
                continue
            features = group[self.feature_cols].to_numpy(dtype=np.float32)
            target = group["label_long"].to_numpy(dtype=np.float32)
            for idx in range(self.seq_len - 1, len(group)):
                window = features[idx - self.seq_len + 1 : idx + 1]
                sequences.append(window)
                labels.append(target[idx])
        if not sequences:
            return SlidingWindowDataset(torch.empty(0), torch.empty(0))
        seq_tensor = torch.from_numpy(np.stack(sequences))
        label_tensor = torch.from_numpy(np.array(labels, dtype=np.float32))
        return SlidingWindowDataset(seq_tensor, label_tensor)

    @staticmethod
    def _slice_range(
        df_sorted: pd.DataFrame,
        window: Optional[tuple[pd.Timestamp, pd.Timestamp]],
    ) -> pd.DataFrame:
        if window is None:
            return df_sorted.copy()
        start, end = window
        mask = (df_sorted["signal_time"] >= start) & (df_sorted["signal_time"] <= end)
        subset = df_sorted.loc[mask].copy()
        if subset.empty:
            return pd.DataFrame(columns=df_sorted.columns)
        return subset


class BaseTemporalClassifier(pl.LightningModule):
    def __init__(self, lr: float, pos_weight: float) -> None:
        super().__init__()
        self.lr = lr
        self.register_buffer("pos_weight", torch.tensor(pos_weight, dtype=torch.float32))
        self.val_auc = BinaryAUROC()
        self.val_ap = BinaryAveragePrecision()
        self.val_precision = BinaryPrecision()
        self.val_recall = BinaryRecall()

    def _step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        inputs, targets = batch
        logits = self(inputs).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=self.pos_weight)
        probs = torch.sigmoid(logits)
        return loss, probs, targets

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss, _, _ = self._step(batch)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss, probs, targets = self._step(batch)
        try:
            self.val_auc.update(probs, targets.int())
            self.val_ap.update(probs, targets.int())
            self.val_precision.update(probs, targets.int())
            self.val_recall.update(probs, targets.int())
        except RuntimeError:
            pass  # metric update can fail if batch only has one class
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def on_validation_epoch_end(self) -> None:
        def _safe_compute(metric, name: str) -> None:
            try:
                value = metric.compute()
                self.log(name, value, prog_bar=True, on_step=False, on_epoch=True)
            except (RuntimeError, ValueError):
                self.log(name, torch.tensor(float("nan")), prog_bar=True, on_step=False, on_epoch=True)
            finally:
                metric.reset()

        _safe_compute(self.val_auc, "val_roc_auc")
        _safe_compute(self.val_ap, "val_pr_auc")
        _safe_compute(self.val_precision, "val_precision")
        _safe_compute(self.val_recall, "val_recall")

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.lr)


class TemporalCNN(BaseTemporalClassifier):
    def __init__(self, feature_dim: int, seq_len: int, lr: float, pos_weight: float, dropout: float = 0.2) -> None:
        super().__init__(lr, pos_weight)
        self.save_hyperparameters(
            {
                "model_type": "tcn",
                "feature_dim": feature_dim,
                "seq_len": seq_len,
                "dropout": dropout,
            }
        )
        self.feature_dim = feature_dim
        self.seq_len = seq_len
        self.dropout = dropout
        self.conv = nn.Sequential(
            nn.Conv1d(feature_dim, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401 - inference hook
        if x.dim() != 3:
            raise ValueError(f"Expected [batch, seq_len, features], got {tuple(x.shape)}")
        x = x.permute(0, 2, 1)
        feats = self.conv(x)
        return self.head(feats)


class TemporalFusionTransformer(BaseTemporalClassifier):
    def __init__(
        self,
        feature_dim: int,
        seq_len: int,
        lr: float,
        pos_weight: float,
        hidden_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.2,
    ) -> None:
        super().__init__(lr, pos_weight)
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by attn heads")
        self.save_hyperparameters(
            {
                "model_type": "tft",
                "feature_dim": feature_dim,
                "seq_len": seq_len,
                "hidden_dim": hidden_dim,
                "num_heads": num_heads,
                "dropout": dropout,
            }
        )
        self.feature_dim = feature_dim
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.input_proj = nn.Linear(feature_dim, hidden_dim)
        self.temporal_gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.gate = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.context = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        head_hidden = max(32, hidden_dim // 2)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, head_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"Expected [batch, seq_len, features], got {tuple(x.shape)}")
        proj = self.input_proj(x)
        temporal, _ = self.temporal_gru(proj)
        attn_out, _ = self.attn(temporal, temporal, temporal)
        gated = self.gate(attn_out) * attn_out
        fused = temporal + gated
        context = self.context(fused[:, -1, :])
        return self.head(context)


def evaluate_model(
    model: pl.LightningModule, dataloader: DataLoader, device: torch.device, plot_dir: Optional[Path] = None
) -> dict[str, float]:
    model.eval()
    probabilities: list[float] = []
    targets: list[float] = []
    with torch.no_grad():
        for batch in dataloader:
            inputs, labels = batch
            inputs = inputs.to(device)
            logits = model(inputs).squeeze(-1)
            probs = torch.sigmoid(logits).cpu().numpy()
            probabilities.extend(probs.tolist())
            targets.extend(labels.numpy().tolist())
    if not probabilities:
        print("Test dataloader returned zero batches - nothing to evaluate.")
        return {}
    prob_arr = np.array(probabilities)
    target_arr = np.array(targets)
    preds = (prob_arr >= 0.5).astype(int)
    report_text = classification_report(target_arr, preds, digits=4)
    print("\n=== Classification report ===")
    print(report_text)
    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(target_arr, preds)),
        "precision": float(precision_score(target_arr, preds, zero_division=0)),
        "recall": float(recall_score(target_arr, preds, zero_division=0)),
        "f1": float(f1_score(target_arr, preds, zero_division=0)),
        "support": int(len(target_arr)),
        "classification_report": report_text,
    }
    try:
        roc_auc = roc_auc_score(target_arr, prob_arr)
        pr_auc = average_precision_score(target_arr, prob_arr)
        print(f"ROC-AUC: {roc_auc:.4f}")
        print(f"PR-AUC:  {pr_auc:.4f}")
        metrics["roc_auc"] = float(roc_auc)
        metrics["pr_auc"] = float(pr_auc)
        if plot_dir is not None:
            fpr, tpr, _ = roc_curve(target_arr, prob_arr)
            plot_dir.mkdir(parents=True, exist_ok=True)
            plt.figure(figsize=(6, 4))
            plt.plot(fpr, tpr, label=f"ROC (AUC={roc_auc:.3f})")
            plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("Temporal CNN ROC Curve")
            plt.legend()
            plt.tight_layout()
            out_path = plot_dir / "roc_curve.png"
            plt.savefig(out_path, dpi=150)
            plt.close()
            print(f"ROC curve saved to {out_path}")
            metrics["roc_curve_path"] = str(out_path)
    except ValueError as exc:
        print(f"Skipped AUC metrics: {exc}")
    return metrics


def prepare_dataframe(dataset_path: Path) -> tuple[pd.DataFrame, list[str]]:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset {dataset_path} not found. Re-run prepare_baseline_dataset first.")
    df = pd.read_csv(dataset_path, parse_dates=["signal_time"])
    if df.empty:
        raise ValueError("Loaded dataset is empty — nothing to train on.")
    feature_cols = [col for col in df.columns if col not in META_COLUMNS]
    if not feature_cols:
        raise ValueError("No feature columns detected in dataset.")
    return df, feature_cols


def _auto_num_workers(requested: int) -> int:
    if requested and requested > 0:
        return requested
    cpu_total = os.cpu_count() or 1
    return max(1, min(4, cpu_total))


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    pl.seed_everything(args.seed, workers=True)
    args.num_workers = _auto_num_workers(getattr(args, "num_workers", 0))
    df, feature_cols = prepare_dataframe(args.dataset_path)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    splits = _load_walk_forward_splits(
        args.walk_forward_json,
        getattr(args, "walk_forward_splits", None),
    )
    if not splits:
        splits = [
            {
                "name": "full_window",
                "windows": {},
            }
        ]
    results: list[dict[str, Any]] = []
    for split in splits:
        split_name = split.get("name", "full_window")
        split_windows = split.get("windows") or {}
        metrics, datamodule, log_dir, best_ckpt = _train_single_run(
            args,
            df,
            feature_cols,
            split_name,
            split_windows,
        )
        run_entry = {
            "name": split_name,
            "metrics": metrics,
            "log_dir": str(log_dir),
            "best_checkpoint": best_ckpt,
            "split_windows": {k: _format_window(v) for k, v in split_windows.items() if v},
            "counts": {
                "train_rows": int(len(datamodule.train_df) if datamodule.train_df is not None else 0),
                "val_rows": int(len(datamodule.val_df) if datamodule.val_df is not None else 0),
                "test_rows": int(len(datamodule.test_df) if datamodule.test_df is not None else 0),
                "train_sequences": int(len(datamodule.train_ds) if datamodule.train_ds else 0),
                "val_sequences": int(len(datamodule.val_ds) if datamodule.val_ds else 0),
                "test_sequences": int(len(datamodule.test_ds) if datamodule.test_ds else 0),
            },
        }
        if metrics.get("roc_curve_path"):
            run_entry["roc_curve_path"] = metrics["roc_curve_path"]
        results.append(run_entry)
    report_path = _write_training_report(args, results)
    summary = {
        "model_type": args.model_type,
        "dataset": str(args.dataset_path),
        "runs": results,
        "report_path": str(report_path) if report_path else None,
    }
    return summary


def _train_single_run(
    args: argparse.Namespace,
    df: pd.DataFrame,
    feature_cols: list[str],
    split_name: str,
    split_windows: dict[str, tuple[pd.Timestamp, pd.Timestamp]],
) -> tuple[dict[str, Any], BaselineDataModule, Path, Optional[str]]:
    datamodule = BaselineDataModule(
        df=df,
        feature_cols=feature_cols,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        num_workers=args.num_workers,
        split_config=split_windows or None,
    )
    datamodule.setup()
    has_val_sequences = datamodule.val_ds is not None and len(datamodule.val_ds) > 0
    model = _build_model(args, len(feature_cols), datamodule.train_pos_weight)
    safe_split = split_name.replace(" ", "_")
    run_log_dir = Path(args.log_dir) / f"{args.model_type}_{safe_split}"
    run_log_dir.mkdir(parents=True, exist_ok=True)
    loggers: list[pl.loggers.Logger] = []
    if args.logger in {"csv", "both"}:
        loggers.append(CSVLogger(save_dir=str(run_log_dir), name="csv"))
    if args.logger in {"tensorboard", "both"}:
        try:
            loggers.append(TensorBoardLogger(save_dir=str(run_log_dir), name="tensorboard"))
        except ModuleNotFoundError:
            print("TensorBoard is not installed; falling back to CSV logging only.")
    checkpoint_dir = run_log_dir / "checkpoints"
    filename_tpl = f"{args.model_type}-{safe_split}-{{epoch:02d}}"
    if has_val_sequences:
        checkpoint_cb = ModelCheckpoint(
            dirpath=str(checkpoint_dir),
            filename=filename_tpl,
            monitor="val_loss",
            save_top_k=1,
            mode="min",
        )
    else:
        checkpoint_cb = ModelCheckpoint(
            dirpath=str(checkpoint_dir),
            filename=filename_tpl,
            save_last=True,
            save_top_k=0,
        )
    callbacks = [
        LearningRateMonitor(logging_interval="epoch"),
        checkpoint_cb,
    ]
    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        max_epochs=args.max_epochs,
        gradient_clip_val=args.gradient_clip,
        log_every_n_steps=20,
        deterministic=True,
        logger=loggers if loggers else None,
        callbacks=callbacks,
        precision=args.precision,
    )
    mlflow_run = None
    if args.mlflow_tracking_uri:
        if mlflow is None:
            raise ModuleNotFoundError("mlflow is not installed. Add it to requirements or disable MLflow logging.")
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)
        experiment = args.mlflow_experiment or "TemporalModels"
        mlflow.set_experiment(experiment)
        tags = _parse_tags(args.mlflow_tags) | {"split": split_name, "model_type": args.model_type}
        run_name = args.mlflow_run_name or f"{args.model_type}-{split_name}"
        mlflow_run = mlflow.start_run(run_name=run_name, tags=tags)
        mlflow.log_params(
            {
                "dataset_path": str(args.dataset_path),
                "seq_len": args.seq_len,
                "batch_size": args.batch_size,
                "max_epochs": args.max_epochs,
                "learning_rate": args.learning_rate,
                "train_ratio": args.train_ratio,
                "val_ratio": args.val_ratio,
                "model_type": args.model_type,
                "split_name": split_name,
            }
        )
    trainer.fit(model, datamodule=datamodule)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    plot_dir = run_log_dir / "plots" if args.plot_roc else None
    metrics = evaluate_model(model, datamodule.test_dataloader(), device=device, plot_dir=plot_dir)
    numeric_metrics = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
    if mlflow_run:
        assert mlflow is not None
        if numeric_metrics:
            mlflow.log_metrics(numeric_metrics)
        mlflow.log_artifacts(str(run_log_dir))
        mlflow.end_run()
    best_ckpt = checkpoint_cb.best_model_path or None
    if best_ckpt:
        metrics.setdefault("best_checkpoint", best_ckpt)
    return metrics, datamodule, run_log_dir, best_ckpt


def _build_model(args: argparse.Namespace, feature_dim: int, pos_weight: float) -> pl.LightningModule:
    if args.model_type == "tft":
        return TemporalFusionTransformer(
            feature_dim=feature_dim,
            seq_len=args.seq_len,
            lr=args.learning_rate,
            pos_weight=pos_weight,
            hidden_dim=args.hidden_dim,
            num_heads=args.attn_heads,
            dropout=args.dropout,
        )
    return TemporalCNN(
        feature_dim=feature_dim,
        seq_len=args.seq_len,
        lr=args.learning_rate,
        pos_weight=pos_weight,
        dropout=args.dropout,
    )


def _load_walk_forward_splits(
    json_path: Optional[Path],
    predefined: Optional[list[dict[str, Any]]] = None,
) -> list[dict[str, Any]]:
    raw = predefined
    if raw is None and json_path:
        if not json_path.exists():
            raise FileNotFoundError(json_path)
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            raw = [payload]
        else:
            raw = payload
    if not raw:
        return []
    normalized: list[dict[str, Any]] = []
    for idx, entry in enumerate(raw):
        windows: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {}
        for key in ("train", "val", "test"):
            window = entry.get(key)
            if window:
                windows[key] = _parse_window(window)
        normalized.append(
            {
                "name": entry.get("name") or f"split_{idx + 1}",
                "windows": windows,
            }
        )
    return normalized


def _parse_window(window: Any) -> tuple[pd.Timestamp, pd.Timestamp]:
    if not isinstance(window, (list, tuple)) or len(window) != 2:
        raise ValueError(f"Split window must be [start, end], got {window!r}")
    start = _coerce_timestamp(window[0])
    end = _coerce_timestamp(window[1])
    if start >= end:
        raise ValueError(f"Split start {start} must be before end {end}")
    return start, end


def _coerce_timestamp(value: Any) -> pd.Timestamp:
    if isinstance(value, pd.Timestamp):
        return value.tz_convert(timezone.utc) if value.tzinfo else value.tz_localize(timezone.utc)
    if isinstance(value, datetime):
        ts = pd.Timestamp(value)
        return ts.tz_convert(timezone.utc) if ts.tzinfo else ts.tz_localize(timezone.utc)
    parsed = pd.to_datetime(value, utc=True)
    if isinstance(parsed, pd.Series):
        parsed = parsed.iloc[0]
    if isinstance(parsed, pd.Timestamp):
        return parsed.tz_convert(timezone.utc) if parsed.tzinfo else parsed.tz_localize(timezone.utc)
    timestamp = pd.Timestamp(parsed)
    return timestamp.tz_convert(timezone.utc) if timestamp.tzinfo else timestamp.tz_localize(timezone.utc)


def _format_window(window: tuple[pd.Timestamp, pd.Timestamp]) -> list[str]:
    start, end = window
    return [start.isoformat(), end.isoformat()]


def _write_training_report(args: argparse.Namespace, runs: list[dict[str, Any]]) -> Optional[Path]:
    if not runs:
        return None
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    filename = args.report_dir / f"{timestamp}_{args.model_type}_{args.dataset_path.stem}.json"
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(args.dataset_path),
        "model_type": args.model_type,
        "seq_len": args.seq_len,
        "batch_size": args.batch_size,
        "max_epochs": args.max_epochs,
        "learning_rate": args.learning_rate,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "walk_forward": runs,
    }
    filename.parent.mkdir(parents=True, exist_ok=True)
    filename.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return filename


def main() -> None:
    args = parse_args()
    summary = run_training(args)
    runs = summary.get("runs", [])
    if runs:
        print("\n=== Training runs summary ===")
        for run in runs:
            metrics = run.get("metrics", {})
            acc = metrics.get("accuracy")
            roc = metrics.get("roc_auc")
            pr = metrics.get("pr_auc")
            if isinstance(acc, (int, float)):
                acc_str = f"{acc:.4f}"
                roc_str = f"{roc:.4f}" if isinstance(roc, (int, float)) else "n/a"
                pr_str = f"{pr:.4f}" if isinstance(pr, (int, float)) else "n/a"
                print(f"- {run.get('name')}: accuracy={acc_str} roc_auc={roc_str} pr_auc={pr_str}")
            else:
                print(f"- {run.get('name')} metrics recorded")
    report_path = summary.get("report_path")
    if report_path:
        print(f"Training report saved to {report_path}")


if __name__ == "__main__":
    main()


def _parse_tags(raw: Optional[list[str]]) -> dict[str, str]:
    if not raw:
        return {}
    tags: dict[str, str] = {}
    for item in raw:
        if "=" in item:
            key, value = item.split("=", 1)
            tags[key.strip()] = value.strip()
    return tags
