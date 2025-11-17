from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare intraday_v2 baseline dataset by joining precomputed technical "
            "features with intraday_v2 labels CSV."
        )
    )
    parser.add_argument(
        "--labels-csv",
        type=Path,
        default=Path("data/training/intraday_v2_labels_1m_2025q4.csv"),
        help="Path to intraday_v2 labels CSV (built from candles)",
    )
    parser.add_argument(
        "--features-csv",
        type=Path,
        required=True,
        help=(
            "Path to technical feature windows CSV (e.g. dataset with one row per "
            "secid+timestamp containing tech_v2/tech_v3 features)."
        ),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("data/training/dataset_intraday_v2_1m_2025q4.csv"),
        help="Where to write the merged dataset (CSV)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    labels_path = args.labels_csv
    features_path = args.features_csv
    output_path = args.output_csv

    if not labels_path.is_file():
        raise FileNotFoundError(f"Labels CSV not found: {labels_path}")
    if not features_path.is_file():
        raise FileNotFoundError(f"Features CSV not found: {features_path}")

    labels = pd.read_csv(labels_path)
    features = pd.read_csv(features_path)

    if "timestamp" not in labels.columns:
        raise ValueError("Labels CSV must contain 'timestamp' column (signal time).")
    if "secid" not in labels.columns:
        raise ValueError("Labels CSV must contain 'secid' column.")

    # Normalise column names and types for joining
    labels["secid"] = labels["secid"].astype(str).str.upper()
    features["secid"] = features["secid"].astype(str).str.upper()

    # We treat labels.timestamp as the signal_time for the model
    labels = labels.rename(columns={"timestamp": "signal_time"})

    # Ensure signal_time is comparable across both tables
    labels["signal_time"] = pd.to_datetime(labels["signal_time"], utc=True)
    if "signal_time" in features.columns:
        features["signal_time"] = pd.to_datetime(features["signal_time"], utc=True)
    elif "window_end" in features.columns:
        features["signal_time"] = pd.to_datetime(features["window_end"], utc=True)
    else:
        raise ValueError(
            "Features CSV must contain either 'signal_time' or 'window_end' column"
            " to align with label timestamps."
        )

    # Basic sanity filter: keep only overlapping interval
    min_time = max(labels["signal_time"].min(), features["signal_time"].min())
    max_time = min(labels["signal_time"].max(), features["signal_time"].max())
    labels = labels[(labels["signal_time"] >= min_time) & (labels["signal_time"] <= max_time)].copy()

    merged = pd.merge(
        labels,
        features,
        on=["secid", "signal_time"],
        suffixes=("_label", "_feat"),
        how="inner",
    )

    if merged.empty:
        raise RuntimeError("Join of labels and features produced an empty dataset.")

    # Align with existing training scripts expectations where possible
    # Rename label columns to the names expected by train_temporal_cnn.py
    if "label_long_label" in merged.columns:
        merged.rename(columns={"label_long_label": "label_long"}, inplace=True)
    if "horizon_minutes_label" in merged.columns:
        merged.rename(columns={"horizon_minutes_label": "horizon_minutes"}, inplace=True)

    merged["timeframe"] = merged.get("timeframe", "1m")
    merged["feature_set"] = merged.get("feature_set", "tech_v2")
    merged["label_set"] = "intraday_v2"

    # Ensure signal_time is stored as ISO string
    merged["signal_time"] = merged["signal_time"].dt.strftime("%Y-%m-%dT%H:%M:%S%z")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)

    print(
        f"Saved intraday_v2 dataset with {len(merged)} rows and "
        f"{len(merged.columns)} columns to {output_path}."
    )


if __name__ == "__main__":
    main()
