from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import requests

from backend.scripts.training.train_temporal_cnn import (
    BaselineDataModule,
    META_COLUMNS,
)
from backend.scripts.ml.pnl_intraday_v2 import _aligned_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay strong_q80 super-trader policy on a historical dataset "
            "via FastAPI decision endpoint."
        ),
    )
    parser.add_argument(
        "--dataset-csv",
        type=Path,
        required=True,
        help="Path to enriched features+labels CSV (e.g. *_enriched_strong_q80.csv)",
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default="http://localhost:8000/api/trader/strong_q80/decision",
        help="URL of the strong_q80 decision endpoint",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=32,
        help="Sequence length used during training",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Train split ratio used during training",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Val split ratio used during training",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Optional limit on number of test rows to replay (0 = all)",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Optional path to dump detailed per-trade log as JSON",
    )
    return parser.parse_args()


def _row_to_payload(row: pd.Series, feature_cols: list[str]) -> Dict[str, Any]:
    # Include both meta and feature columns; backend service will pick what it needs
    payload: Dict[str, Any] = {}
    for col in row.index:
        val = row[col]
        if isinstance(val, (np.floating, np.integer)):
            payload[col] = float(val)
        else:
            payload[col] = val
    # Ensure feature columns exist (fill missing with 0.0)
    for col in feature_cols:
        if col not in payload:
            payload[col] = 0.0
    return payload


def main() -> None:
    args = parse_args()

    if not args.dataset_csv.is_file():
        raise FileNotFoundError(args.dataset_csv)

    df = pd.read_csv(args.dataset_csv)
    feature_cols = [c for c in df.columns if c not in META_COLUMNS]

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
    if args.max_rows > 0:
        aligned = aligned.head(args.max_rows)

    trades: list[dict[str, Any]] = []
    total = len(aligned)
    print(f"Replaying {total} rows via {args.api_url}")

    for idx, row in aligned.reset_index(drop=True).iterrows():
        payload = _row_to_payload(row, feature_cols)
        try:
            resp = requests.post(args.api_url, json=payload, timeout=5)
            resp.raise_for_status()
        except Exception as exc:
            print(f"[{idx+1}/{total}] API error: {exc}")
            continue

        decision = resp.json()
        action = decision.get("action", "HOLD")
        p_long = float(decision.get("p_long", 0.0))
        p_short = float(decision.get("p_short", 0.0))
        regime = decision.get("regime", {})

        # Realized P&L proxy
        long_pnl = float(row.get("forward_return_pct", 0.0))
        short_pnl = float(row.get("short_pnl_pct", 0.0))
        if action == "OPEN_LONG":
            realized_pnl = long_pnl
        elif action == "OPEN_SHORT":
            realized_pnl = short_pnl
        else:
            realized_pnl = 0.0

        trades.append(
            {
                "index": int(idx),
                "secid": row.get("secid"),
                "signal_time": row.get("signal_time"),
                "action": action,
                "p_long": p_long,
                "p_short": p_short,
                "forward_return_pct": long_pnl,
                "short_pnl_pct": short_pnl,
                "realized_pnl": realized_pnl,
                "regime": regime,
            },
        )

        if (idx + 1) % 1000 == 0:
            print(f"Processed {idx+1}/{total} rows")

    if not trades:
        print("No trades recorded (maybe all actions were HOLD or API failed).")
        return

    df_trades = pd.DataFrame(trades)
    signals = df_trades[df_trades["action"] != "HOLD"]
    if signals.empty:
        print("No OPEN_LONG/OPEN_SHORT actions produced.")
        return

    win_rate = float((signals["realized_pnl"] > 0).mean())
    mean_pnl = float(signals["realized_pnl"].mean())
    sum_pnl = float(signals["realized_pnl"].sum())

    print(
        f"\nReplay summary:\n"
        f"  total_rows={total}\n"
        f"  signals={len(signals)}\n"
        f"  win_rate(realized_pnl>0)={win_rate:.3f}\n"
        f"  mean_realized_pnl={mean_pnl:.5f}\n"
        f"  sum_realized_pnl={sum_pnl:.4f}\n",
    )

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(trades, indent=2))
        print(f"Saved detailed trades to {args.output_json}")


if __name__ == "__main__":
    main()

