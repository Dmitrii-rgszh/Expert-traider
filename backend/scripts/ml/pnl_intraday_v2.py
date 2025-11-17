from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.scripts.training.train_temporal_cnn import (
    BaselineDataModule,
    META_COLUMNS,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply probability thresholds to intraday_v2 predictions and compute trading metrics",
    )
    parser.add_argument("--dataset-csv", type=Path, required=True, help="Path to merged features+labels CSV")
    parser.add_argument("--pred-csv", type=Path, required=True, help="CSV with y_true and y_pred_proba (test split)")
    parser.add_argument(
        "--seq-len",
        type=int,
        default=32,
        help="Sequence length used during training (needed to align metadata)",
    )
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Train split ratio used for training")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation split ratio used for training")
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="*",
        default=[0.03194498, 0.04493976],
        help="Probability thresholds to evaluate (default: approx precision 0.8 and 0.9)",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Optional path to dump metrics as JSON",
    )
    return parser.parse_args()


def _aligned_rows(df_subset: pd.DataFrame, seq_len: int) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for _, group in df_subset.groupby("secid"):
        group = group.sort_values("signal_time")
        if len(group) < seq_len:
            continue
        rows.append(group.iloc[seq_len - 1 :].copy())
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, axis=0).reset_index(drop=True)


def _max_drawdown(series: pd.Series) -> float:
    if series.empty:
        return 0.0
    cumulative = series.cumsum()
    running_max = cumulative.cummax()
    drawdown = running_max - cumulative
    return float(drawdown.max())


def main() -> None:
    args = parse_args()

    if not args.dataset_csv.is_file():
        raise FileNotFoundError(args.dataset_csv)
    if not args.pred_csv.is_file():
        raise FileNotFoundError(args.pred_csv)

    df = pd.read_csv(args.dataset_csv)
    feature_cols = [col for col in df.columns if col not in META_COLUMNS]

    data_module = BaselineDataModule(
        df=df,
        feature_cols=feature_cols,
        seq_len=args.seq_len,
        batch_size=1024,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        num_workers=0,
    )
    data_module.setup("fit")
    test_df = data_module.test_df
    if test_df is None or test_df.empty:
        raise RuntimeError("Test split is empty; check ratios or dataset window")

    aligned = _aligned_rows(test_df, args.seq_len)
    preds = pd.read_csv(args.pred_csv)
    if len(preds) != len(aligned):
        raise ValueError(
            f"Predictions ({len(preds)}) do not match aligned rows ({len(aligned)})."
            " Ensure you pass the same dataset split as used during training."
        )
    aligned = aligned.reset_index(drop=True)
    aligned["y_true"] = preds["y_true"].values
    aligned["y_pred_proba"] = preds["y_pred_proba"].values

    results: dict[str, dict[str, object]] = {}

    for thr in args.thresholds:
        selected = aligned[aligned["y_pred_proba"] >= thr].copy()
        key = f"thr_{thr:.6f}"
        if selected.empty:
            results[key] = {"signals": 0}
            print(f"Threshold {thr:.6f}: no signals")
            continue

        win_rate = float(selected["y_true"].mean())
        avg_r = float(selected["r_multiple"].mean())
        median_r = float(selected["r_multiple"].median())
        pnl_std = float(selected["r_multiple"].std(ddof=0))
        expectancy = avg_r
        max_dd = _max_drawdown(selected["r_multiple"])

        per_secid = (
            selected.groupby("secid")
            .agg(
                signals=("r_multiple", "count"),
                win_rate=("y_true", "mean"),
                avg_r=("r_multiple", "mean"),
                median_r=("r_multiple", "median"),
            )
            .sort_index()
        )
        selected["date"] = pd.to_datetime(selected["signal_time"]).dt.date
        per_day = (
            selected.groupby("date")
            .agg(
                signals=("r_multiple", "count"),
                win_rate=("y_true", "mean"),
                avg_r=("r_multiple", "mean"),
                pnl_sum=("r_multiple", "sum"),
            )
            .sort_index()
        )
        equity_curve = per_day["pnl_sum"].cumsum().tolist()
        per_day_reset = per_day.reset_index()
        per_day_reset["date"] = per_day_reset["date"].astype(str)

        results[key] = {
            "threshold": thr,
            "signals": int(len(selected)),
            "win_rate": win_rate,
            "avg_r": avg_r,
            "median_r": median_r,
            "expectancy": expectancy,
            "pnl_std": pnl_std,
            "max_drawdown": max_dd,
            "by_secid": per_secid.reset_index().to_dict(orient="records"),
            "by_day": per_day_reset.to_dict(orient="records"),
            "equity_curve": equity_curve,
        }

        print(
            f"Threshold {thr:.6f}: signals={len(selected)}, win_rate={win_rate:.3f}, "
            f"avg_r={avg_r:.4f}, median_r={median_r:.4f}, max_dd={max_dd:.4f}"
        )

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(results, indent=2))
        print(f"Saved summary to {args.output_json}")


if __name__ == "__main__":
    main()
