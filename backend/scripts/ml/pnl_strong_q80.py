from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.scripts.training.train_temporal_cnn import (  # noqa: E402
    BaselineDataModule,
    META_COLUMNS,
)
from backend.scripts.ml.pnl_intraday_v2 import _aligned_rows  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Apply probability thresholds to strong_q80 predictions and "
            "compute P&L / winrate stats with simple regime breakdowns."
        ),
    )
    parser.add_argument(
        "--dataset-csv",
        type=Path,
        required=True,
        help="Path to enriched features+labels CSV (e.g. *_enriched_strong_q80.csv)",
    )
    parser.add_argument(
        "--pred-csv",
        type=Path,
        required=True,
        help="CSV with y_true and y_pred_proba for the test split",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=32,
        help="Sequence length used during training (for alignment)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Train split ratio used for training",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Validation split ratio used for training",
    )
    parser.add_argument(
        "--target-column",
        type=str,
        default="label_long_strong",
        help="Target column used during training (default: label_long_strong)",
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="*",
        default=[0.0020, 0.0035, 0.0050, 0.0100],
        help="Probability thresholds to evaluate",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Optional path to dump metrics (per-threshold + regime breakdowns) as JSON",
    )
    parser.add_argument(
        "--direction",
        type=str,
        choices=["long", "short"],
        default="long",
        help="Direction for P&L: long=forward_return_pct, short=short_pnl_pct",
    )
    return parser.parse_args()


def _regime_buckets(df: pd.DataFrame) -> pd.DataFrame:
    """Attach simple regime buckets for volatility, time-of-day and liquidity."""
    out = df.copy()

    # Volatility bucket from intraday volatility_20
    if "volatility_20" in out.columns:
        try:
            out["vol_bucket"] = pd.qcut(
                out["volatility_20"].astype(float),
                q=3,
                labels=["low_vol", "mid_vol", "high_vol"],
            )
        except ValueError:
            out["vol_bucket"] = "unknown"
    else:
        out["vol_bucket"] = "unknown"

    # Liquidity bucket from volume_zscore_20
    if "volume_zscore_20" in out.columns:
        try:
            out["liq_bucket"] = pd.qcut(
                out["volume_zscore_20"].astype(float),
                q=3,
                labels=["low_liq", "mid_liq", "high_liq"],
            )
        except ValueError:
            out["liq_bucket"] = "unknown"
    else:
        out["liq_bucket"] = "unknown"

    # Time-of-day bucket from signal_time (assumed local exchange time or UTC-consistent)
    if "signal_time" in out.columns:
        dt = pd.to_datetime(out["signal_time"])
        hour = dt.dt.hour
        # Rough MOEX-style buckets; exact timezone is less important than consistency
        conditions = [
            (hour >= 10) & (hour < 12),
            (hour >= 12) & (hour < 15),
            (hour >= 15) & (hour <= 19),
        ]
        choices = ["open", "midday", "close"]
        out["tod_bucket"] = np.select(conditions, choices, default="other")
    else:
        out["tod_bucket"] = "unknown"

    return out


def _regime_summary(df: pd.DataFrame, bucket_col: str, pnl_col: str) -> list[dict[str, Any]]:
    if bucket_col not in df.columns:
        return []
    grouped = (
        df.groupby(bucket_col)
        .agg(
            signals=(pnl_col, "count"),
            strong_label_rate=("y_true", "mean"),
            win_rate=(pnl_col, lambda s: float((s > 0).mean())),
            mean_forward_return_pct=(pnl_col, "mean"),
            median_forward_return_pct=(pnl_col, "median"),
            sum_forward_return_pct=(pnl_col, "sum"),
        )
        .reset_index()
    )
    return grouped.to_dict(orient="records")


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
        target_column=args.target_column,
    )
    data_module.setup("fit")
    test_df = data_module.test_df
    if test_df is None or test_df.empty:
        raise RuntimeError("Test split is empty; check ratios or dataset window")

    aligned = _aligned_rows(test_df, args.seq_len)
    preds = pd.read_csv(args.pred_csv)
    if len(preds) != len(aligned):
        raise ValueError(
            f"Predictions ({len(preds)}) do not match aligned rows ({len(aligned)}). "
            "Ensure you pass the same dataset and splits as used during training/extraction.",
        )

    aligned = aligned.reset_index(drop=True)
    aligned["y_true"] = preds["y_true"].values
    aligned["y_pred_proba"] = preds["y_pred_proba"].values

    if args.direction == "long":
        pnl_col = "forward_return_pct"
    else:
        pnl_col = "short_pnl_pct"

    if pnl_col not in aligned.columns:
        raise KeyError(f"{pnl_col} column is required in dataset for P&L computation")

    results: dict[str, dict[str, Any]] = {}

    for thr in args.thresholds:
        selected = aligned[aligned["y_pred_proba"] >= thr].copy()
        key = f"thr_{thr:.6f}"
        if selected.empty:
            results[key] = {"threshold": thr, "signals": 0}
            print(f"Threshold {thr:.6f}: no signals")
            continue

        selected = _regime_buckets(selected)

        signals = int(len(selected))
        strong_label_rate = float(selected["y_true"].mean())
        win_rate = float((selected[pnl_col] > 0).mean())
        mean_fwd = float(selected[pnl_col].mean())
        median_fwd = float(selected[pnl_col].median())
        sum_fwd = float(selected[pnl_col].sum())

        results[key] = {
            "threshold": thr,
            "signals": signals,
            "strong_label_rate": strong_label_rate,
            "win_rate_forward_return_gt_0": win_rate,
            "mean_forward_return_pct": mean_fwd,
            "median_forward_return_pct": median_fwd,
            "sum_forward_return_pct": sum_fwd,
            "regimes": {
                "vol_bucket": _regime_summary(selected, "vol_bucket", pnl_col),
                "liq_bucket": _regime_summary(selected, "liq_bucket", pnl_col),
                "tod_bucket": _regime_summary(selected, "tod_bucket", pnl_col),
            },
        }

        print(
            f"Threshold {thr:.6f}: "
            f"signals={signals}, "
            f"strong_label_rate={strong_label_rate:.3f}, "
            f"winrate(fwd>0)={win_rate:.3f}, "
            f"mean_fwd={mean_fwd:.5f}, "
            f"sum_fwd={sum_fwd:.4f}",
        )

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(results, indent=2))
        print(f"Saved summary to {args.output_json}")


if __name__ == "__main__":
    main()
