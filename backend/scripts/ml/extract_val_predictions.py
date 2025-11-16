"""
Extract validation predictions from trained PyTorch Lightning checkpoint.

This script loads a trained TCN/TFT model checkpoint, runs inference on the
validation dataset, and saves predictions (y_true, y_pred_proba) to CSV for
threshold calibration.

Usage:
    python backend/scripts/ml/extract_val_predictions.py \\
        --checkpoint path/to/checkpoint.ckpt \\
        --dataset-path data/training/dataset.csv \\
        --output data/modeling/val_predictions.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.scripts.training.train_temporal_cnn import (
    BaselineDataModule,
    TemporalCNN,
    TemporalFusionTransformer,
    META_COLUMNS,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract validation predictions from trained checkpoint"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to PyTorch Lightning checkpoint (.ckpt file)",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        required=True,
        help="Path to the CSV dataset used for training",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/modeling/val_predictions.csv"),
        help="Output path for validation predictions CSV",
    )
    parser.add_argument(
        "--model-type",
        choices=["tcn", "tft"],
        default="tcn",
        help="Model architecture type (default: tcn)",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=32,
        help="Sequence length used during training (default: 32)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for inference (default: 64)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Train split ratio used during training (default: 0.7)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Val split ratio used during training (default: 0.15)",
    )
    return parser.parse_args()


def load_model(checkpoint_path: Path, model_type: str):
    """Load trained model from checkpoint."""
    # Load checkpoint to extract hyperparameters
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    hparams = ckpt.get("hyper_parameters", {})
    
    # Extract required parameters with defaults
    lr = hparams.get("lr", 0.001)
    pos_weight = hparams.get("pos_weight", 1.0)
    
    if model_type == "tcn":
        model = TemporalCNN.load_from_checkpoint(
            str(checkpoint_path),
            lr=lr,
            pos_weight=pos_weight,
        )
    else:
        model = TemporalFusionTransformer.load_from_checkpoint(
            str(checkpoint_path),
            lr=lr,
            pos_weight=pos_weight,
        )
    model.eval()
    model.freeze()
    return model


def extract_predictions(
    model,
    val_loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Run inference on validation set and extract predictions."""
    all_y_true = []
    all_y_pred_proba = []

    with torch.no_grad():
        for batch in val_loader:
            x, y = batch
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            proba = torch.sigmoid(logits).cpu().numpy().flatten()

            all_y_true.append(y.cpu().numpy().flatten())
            all_y_pred_proba.append(proba)

    y_true = np.concatenate(all_y_true)
    y_pred_proba = np.concatenate(all_y_pred_proba)
    return y_true, y_pred_proba


def main() -> None:
    args = parse_args()

    # Load dataset
    print(f"Loading dataset: {args.dataset_path}")
    df = pd.read_csv(args.dataset_path)
    print(f"Dataset shape: {df.shape}")

    # Prepare feature columns (exclude metadata)
    feature_cols = [col for col in df.columns if col not in META_COLUMNS]
    print(f"Feature columns: {len(feature_cols)}")

    # Create data module to get validation split
    data_module = BaselineDataModule(
        df=df,
        feature_cols=feature_cols,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        num_workers=0,
    )
    data_module.setup("fit")
    val_loader = data_module.val_dataloader()
    print(f"Validation samples: {len(data_module.val_ds)}")

    # Load model
    print(f"Loading model from: {args.checkpoint}")
    model = load_model(args.checkpoint, args.model_type)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Model loaded on device: {device}")

    # Extract predictions
    print("Running inference on validation set...")
    y_true, y_pred_proba = extract_predictions(model, val_loader, device)
    print(f"Extracted {len(y_true)} predictions")

    # Save to CSV
    output_df = pd.DataFrame({
        "y_true": y_true,
        "y_pred_proba": y_pred_proba,
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(args.output, index=False)
    print(f"Saved validation predictions to: {args.output}")

    # Print summary statistics
    print(f"\\nSummary:")
    print(f"  Positive samples: {np.sum(y_true)} ({100 * np.mean(y_true):.2f}%)")
    print(f"  Mean predicted probability: {np.mean(y_pred_proba):.4f}")
    print(f"  Std predicted probability: {np.std(y_pred_proba):.4f}")


if __name__ == "__main__":
    main()
