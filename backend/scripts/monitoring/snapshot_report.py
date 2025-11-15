from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
from sqlalchemy import select

from backend.app.db.session import SessionLocal
from backend.app.models.operations import TrainDataSnapshot

DEFAULT_OUTPUT = Path("docs/data_quality/train_snapshots.json")


def _parse_date(value: Optional[str], default: datetime) -> datetime:
    if not value:
        return default
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def load_snapshots(feature_set: str, timeframe: str, since: datetime) -> pd.DataFrame:
    with SessionLocal() as session:
        stmt = (
            select(TrainDataSnapshot)
            .where(
                TrainDataSnapshot.feature_set == feature_set,
                TrainDataSnapshot.timeframe == timeframe,
                TrainDataSnapshot.snapshot_end >= since,
            )
            .order_by(TrainDataSnapshot.snapshot_end.asc())
        )
        rows = session.execute(stmt).scalars().all()
    payload = [
        {
            "secid": row.secid,
            "timeframe": row.timeframe,
            "feature_set": row.feature_set,
            "snapshot_start": row.snapshot_start,
            "snapshot_end": row.snapshot_end,
            "rows_count": row.rows_count,
        }
        for row in rows
    ]
    if not payload:
        return pd.DataFrame()
    df = pd.DataFrame(payload)
    df["snapshot_start"] = pd.to_datetime(df["snapshot_start"], utc=True)
    df["snapshot_end"] = pd.to_datetime(df["snapshot_end"], utc=True)
    df["date"] = df["snapshot_end"].dt.date
    return df


def summarize(df: pd.DataFrame, feature_set: str, timeframe: str) -> dict[str, object]:
    summary: dict[str, object] = {
        "feature_set": feature_set,
        "timeframe": timeframe,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "secids": {},
        "text_summary": "No snapshots in the requested window.",
    }
    if df.empty:
        return summary
    lines = [
        f"Snapshot summary for {feature_set}/{timeframe}",
        f"- total rows: {int(df['rows_count'].sum())}",
        f"- secids: {', '.join(sorted(df['secid'].unique()))}",
        f"- last snapshot: {df['snapshot_end'].max().isoformat()}",
    ]
    for secid, chunk in df.groupby("secid"):
        lines.append(
            f"  · {secid}: {len(chunk)} snapshots, rows={int(chunk['rows_count'].sum())}, "
            f"latest={chunk['snapshot_end'].max().isoformat()}"
        )
        summary["secids"][secid] = {
            "snapshots": int(len(chunk)),
            "total_rows": int(chunk["rows_count"].sum()),
            "last_snapshot_end": chunk["snapshot_end"].max().isoformat(),
        }
    summary["text_summary"] = "\n".join(lines)
    return summary


def write_summary(summary: dict[str, object], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate summary for train_data_snapshots")
    parser.add_argument("--feature-set", default="tech_v1", help="Feature set to analyze")
    parser.add_argument("--timeframe", default="1m", help="Timeframe filter (default: 1m)")
    parser.add_argument(
        "--since",
        help="ISO datetime (UTC) from which to include snapshots (default: now-7d)",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output JSON path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    default_since = datetime.now(timezone.utc) - timedelta(days=7)
    since = _parse_date(args.since, default_since)
    df = load_snapshots(args.feature_set, args.timeframe, since)
    summary = summarize(df, args.feature_set, args.timeframe)
    output_path = write_summary(summary, args.output)
    print(summary["text_summary"])
    print(f"Snapshot report saved to {output_path}")


if __name__ == "__main__":
    main()


def generate_snapshot_report(
    feature_set: str,
    timeframe: str,
    since: Optional[str],
    output: Path = DEFAULT_OUTPUT,
) -> Path:
    default_since = datetime.now(timezone.utc) - timedelta(days=7)
    since_dt = _parse_date(since, default_since)
    df = load_snapshots(feature_set, timeframe, since_dt)
    summary = summarize(df, feature_set, timeframe)
    return write_summary(summary, output)
