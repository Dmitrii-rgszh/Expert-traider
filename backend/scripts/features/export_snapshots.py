from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from backend.app.db.session import SessionLocal
from backend.app.models import FeatureWindow
from backend.app.models.operations import TrainDataSnapshot

DEFAULT_OUTPUT_DIR = Path("data/processed/features")
SUPPORTED_FORMATS = {"parquet", "feather"}


def _parse_datetime(value: str) -> datetime:
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _normalize_secids(secids: Iterable[str] | None) -> list[str]:
    if not secids:
        return []
    return [sec.upper() for sec in secids if sec.strip()]


def fetch_feature_windows(
    secids: Sequence[str] | None,
    timeframe: str,
    feature_set: str,
    start_dt: datetime,
    end_dt: datetime,
) -> list[FeatureWindow]:
    with SessionLocal() as session:
        stmt = (
            select(FeatureWindow)
            .options(selectinload(FeatureWindow.numeric_features))
            .where(
                FeatureWindow.feature_set == feature_set,
                FeatureWindow.timeframe == timeframe,
                FeatureWindow.window_end >= start_dt,
                FeatureWindow.window_end <= end_dt,
            )
            .order_by(FeatureWindow.window_end.asc())
        )
        if secids:
            stmt = stmt.where(FeatureWindow.secid.in_(list(secids)))
        return session.execute(stmt).scalars().all()


def build_dataframe(windows: Sequence[FeatureWindow]) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for window in windows:
        record: dict[str, object] = {
            "secid": window.secid,
            "timeframe": window.timeframe,
            "feature_set": window.feature_set,
            "window_start": window.window_start,
            "window_end": window.window_end,
            "generated_at": window.generated_at,
        }
        for feature in window.numeric_features:
            record[feature.feature_name] = feature.value_numeric
        records.append(record)
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame.from_records(records)
    df["window_end"] = pd.to_datetime(df["window_end"], utc=True)
    df["window_start"] = pd.to_datetime(df["window_start"], utc=True)
    df["generated_at"] = pd.to_datetime(df["generated_at"], utc=True)
    return df


def export_feature_snapshots(
    secids: Sequence[str] | None,
    timeframe: str,
    feature_set: str,
    start_dt: datetime,
    end_dt: datetime,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    file_format: str = "parquet",
    partition_daily: bool = True,
) -> list[Path]:
    file_format = file_format.lower()
    if file_format not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported format '{file_format}' (allowed: {', '.join(sorted(SUPPORTED_FORMATS))})")

    windows = fetch_feature_windows(secids, timeframe, feature_set, start_dt, end_dt)
    df = build_dataframe(windows)
    if df.empty:
        return []

    df.sort_values(["window_end", "secid"], inplace=True)
    df["snapshot_date"] = df["window_end"].dt.strftime("%Y%m%d")

    root = Path(output_dir) / feature_set / timeframe
    root.mkdir(parents=True, exist_ok=True)

    written_files: list[Path] = []
    if partition_daily:
        grouped = df.groupby("snapshot_date")
        for snapshot_date, frame in grouped:
            path = root / f"{snapshot_date}.{file_format}"
            _write_dataframe(frame.drop(columns=["snapshot_date"]), path, file_format)
            _register_snapshot_metadata(frame, timeframe, feature_set)
            written_files.append(path)
    else:
        suffix = f"{start_dt:%Y%m%d}_{end_dt:%Y%m%d}"
        path = root / f"snapshot_{suffix}.{file_format}"
        trimmed = df.drop(columns=["snapshot_date"])
        _write_dataframe(trimmed, path, file_format)
        _register_snapshot_metadata(trimmed, timeframe, feature_set)
        written_files.append(path)
    return written_files


def _write_dataframe(df: pd.DataFrame, path: Path, file_format: str) -> None:
    if file_format == "parquet":
        df.to_parquet(path, index=False)
    elif file_format == "feather":
        df.to_feather(path)
    else:  # pragma: no cover - guarded above
        raise ValueError(file_format)


def _register_snapshot_metadata(frame: pd.DataFrame, timeframe: str, feature_set: str) -> None:
    if frame.empty:
        return
    with SessionLocal() as session:
        for secid, chunk in frame.groupby("secid"):
            start = chunk["window_start"].min().to_pydatetime()
            end = chunk["window_end"].max().to_pydatetime()
            rows = int(len(chunk))
            session.merge(
                TrainDataSnapshot(
                    secid=secid,
                    timeframe=timeframe,
                    feature_set=feature_set,
                    snapshot_start=start,
                    snapshot_end=end,
                    rows_count=rows,
                )
            )
        session.commit()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export feature windows snapshots into parquet/feather files")
    parser.add_argument("--secids", nargs="*", help="Optional list of tickers to export")
    parser.add_argument("--timeframe", default="1m", help="Timeframe filter (default: 1m)")
    parser.add_argument("--feature-set", default="tech_v1", help="Feature set to export (default: tech_v1)")
    parser.add_argument("--start-date", required=True, help="Start datetime (ISO)")
    parser.add_argument("--end-date", required=True, help="End datetime (ISO)")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory root (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--format",
        choices=sorted(SUPPORTED_FORMATS),
        default="parquet",
        help="Snapshot file format (default: parquet)",
    )
    parser.add_argument(
        "--daily-partition",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Partition snapshots by day (default: true)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    secids = _normalize_secids(args.secids)
    start_dt = _parse_datetime(args.start_date)
    end_dt = _parse_datetime(args.end_date)
    files = export_feature_snapshots(
        secids=secids,
        timeframe=args.timeframe,
        feature_set=args.feature_set,
        start_dt=start_dt,
        end_dt=end_dt,
        output_dir=args.output_dir,
        file_format=args.format,
        partition_daily=args.daily_partition,
    )
    if not files:
        print("No feature windows found for the requested interval.")
        return
    print(f"Exported {len(files)} snapshot file(s) to {args.output_dir}")
    for file in files:
        print(f" - {file}")


if __name__ == "__main__":
    main()
