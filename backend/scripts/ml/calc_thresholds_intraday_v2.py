from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute precision/recall thresholds for intraday_v2 predictions",
    )
    parser.add_argument(
        "--pred-csv",
        type=Path,
        required=True,
        help="CSV with columns y_true, y_pred_proba (validation or test predictions)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.pred_csv)
    y = df["y_true"].values
    p = df["y_pred_proba"].values

    prec, rec, thr = precision_recall_curve(y, p)

    print("pos_rate", y.mean())
    for target_p in [0.6, 0.7, 0.8, 0.9]:
        idx = np.where(prec >= target_p)[0]
        if len(idx) == 0:
            print("target", target_p, "unreachable")
            continue
        i = idx[0]
        n_signals = int(rec[i] * y.sum())
        print(
            "target", target_p,
            "thr", float(thr[i]),
            "prec", float(prec[i]),
            "rec", float(rec[i]),
            "n_signals", n_signals,
        )


if __name__ == "__main__":
    main()
