"""EDA helper for intraday_v2 labels.

This script is meant to answer:
- насколько равномерно метки распределены по времени и тикерам;
- как выглядит профиль r_multiple (R-множители TP/SL);
- нет ли аномалий (дни без меток, тикеры с странной статистикой).

It prints a few aggregated tables and can dump JSON summaries
for notebooks/dashboards.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EDA for intraday_v2 labels")
    p.add_argument("--labels-csv", type=Path, required=True)
    p.add_argument("--summary-json", type=Path, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.labels_csv)

    # Ensure timestamps are datetime
    if not np.issubdtype(df["timestamp"].dtype, np.datetime64):
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    df["date"] = df["timestamp"].dt.date

    print("=== Overall ===")
    print("rows:", len(df))
    print("positive rate:", df["label_long"].mean())
    print("r_multiple mean/median:", df["r_multiple"].mean(), df["r_multiple"].median())

    print("\n=== By date (top 10) ===")
    by_date = df.groupby("date").agg(
        rows=("label_long", "size"),
        positives=("label_long", "sum"),
        pos_rate=("label_long", "mean"),
        r_mean=("r_multiple", "mean"),
    ).sort_index()
    print(by_date.head(10))

    print("\n=== By secid ===")
    by_secid = df.groupby("secid").agg(
        rows=("label_long", "size"),
        positives=("label_long", "sum"),
        pos_rate=("label_long", "mean"),
        r_mean=("r_multiple", "mean"),
        r_median=("r_multiple", "median"),
    ).sort_values("pos_rate", ascending=False)
    print(by_secid)

    # Convert date index to string so JSON can serialize it
    by_date_reset = by_date.reset_index()
    by_date_reset["date"] = by_date_reset["date"].astype(str)

    summary = {
        "rows": int(len(df)),
        "positive_rate": float(df["label_long"].mean()),
        "r_multiple_mean": float(df["r_multiple"].mean()),
        "r_multiple_median": float(df["r_multiple"].median()),
        "by_date": by_date_reset.to_dict(orient="records"),
        "by_secid": by_secid.reset_index().to_dict(orient="records"),
    }

    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
