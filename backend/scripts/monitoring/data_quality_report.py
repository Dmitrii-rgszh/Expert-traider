from __future__ import annotations

import argparse
import json
from datetime import date, datetime, time, timezone
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
from sqlalchemy import select

from backend.app.db.session import SessionLocal
from backend.app.models import Candle
from backend.scripts.ingestion.candles import INTERVAL_MAP

DEFAULT_OUTPUT_DIR = Path("docs/data_quality")


def _parse_date(value: str, end_of_day: bool = False) -> datetime:
    dt = datetime.fromisoformat(value)
    if isinstance(dt, datetime):
        base_date = dt.date()
    else:  # pragma: no cover - defensive
        base_date = dt
    if end_of_day:
        return datetime.combine(base_date, time.max, tzinfo=timezone.utc)
    return datetime.combine(base_date, time.min, tzinfo=timezone.utc)


def _normalize_secids(secids: Iterable[str] | None) -> list[str]:
    if not secids:
        return []
    return [sec.strip().upper() for sec in secids if sec.strip()]


def load_candles(secids: list[str], timeframe: str, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    with SessionLocal() as session:
        stmt = (
            select(Candle.secid, Candle.timestamp)
            .where(
                Candle.timeframe == timeframe,
                Candle.timestamp >= start_dt,
                Candle.timestamp <= end_dt,
            )
            .order_by(Candle.secid.asc(), Candle.timestamp.asc())
        )
        if secids:
            stmt = stmt.where(Candle.secid.in_(secids))
        rows = session.execute(stmt).all()
    if not rows:
        return pd.DataFrame(columns=["secid", "timestamp"])
    df = pd.DataFrame(rows, columns=["secid", "timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["date"] = df["timestamp"].dt.date
    return df


def _expected_business_days(start_dt: datetime, end_dt: datetime) -> list[date]:
    business_range = pd.bdate_range(start_dt.date(), end_dt.date())
    return [idx.date() for idx in business_range]


def _timeframe_minutes(timeframe: str) -> int:
    interval = INTERVAL_MAP.get(timeframe)
    if not interval:
        raise ValueError(f"Unsupported timeframe '{timeframe}'")
    return interval


def evaluate_quality(
    df: pd.DataFrame,
    timeframe: str,
    start_dt: datetime,
    end_dt: datetime,
) -> dict[str, object]:
    generated_at = datetime.now(timezone.utc)
    summary: dict[str, object] = {
        "timeframe": timeframe,
        "start": start_dt.isoformat(),
        "end": end_dt.isoformat(),
        "generated_at": generated_at.isoformat(),
        "secids": {},
    }
    if df.empty:
        summary["text_summary"] = "No candles found for the requested window."
        return summary

    expected_days = _expected_business_days(start_dt, end_dt)
    timeframe_minutes = _timeframe_minutes(timeframe)
    lines = [
        f"Candles quality report timeframe={timeframe} {start_dt.date()}..{end_dt.date()}",
    ]

    for secid, chunk in df.groupby("secid"):
        chunk = chunk.sort_values("timestamp")
        per_day = chunk.groupby("date")["timestamp"].count()
        trading_minutes_per_day = 390  # Approximate MOEX equities session (6.5h)
        expected_rows = len(expected_days) * (trading_minutes_per_day // timeframe_minutes) if timeframe_minutes else len(chunk)
        coverage_ratio = float(len(chunk)) / expected_rows if expected_rows else 0.0
        missing_days = sorted(set(expected_days) - set(per_day.index))
        gaps = chunk["timestamp"].diff().dt.total_seconds().div(60).dropna()
        metrics = {
            "secid": secid,
            "rows": int(len(chunk)),
            "days_observed": int(per_day.index.nunique()),
            "expected_business_days": len(expected_days),
            "coverage_ratio": coverage_ratio,
            "min_bars_per_day": int(per_day.min()) if not per_day.empty else 0,
            "median_bars_per_day": float(per_day.median()) if not per_day.empty else 0.0,
            "max_bars_per_day": int(per_day.max()) if not per_day.empty else 0,
            "max_gap_minutes": float(gaps.max()) if not gaps.empty else 0.0,
            "first_timestamp": chunk["timestamp"].min().isoformat(),
            "last_timestamp": chunk["timestamp"].max().isoformat(),
            "missing_days": [d.isoformat() for d in missing_days[:10]],
            "missing_days_count": len(missing_days),
        }
        summary["secids"][secid] = metrics
        lines.append(
            f"- {secid}: rows={metrics['rows']} coverage={coverage_ratio:.2%} "
            f"median/day={metrics['median_bars_per_day']:.0f} missing_days={metrics['missing_days_count']}"
        )
    summary["text_summary"] = "\n".join(lines)
    return summary


def write_report(summary: dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate candle data quality report")
    parser.add_argument("--secids", nargs="*", help="Optional tickers to include (default: all)")
    parser.add_argument("--timeframe", default="1m", help="Timeframe to validate (default: 1m)")
    parser.add_argument("--start-date", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "--output",
        type=Path,
        help="Destination JSON report path (default: docs/data_quality/candles_<tf>_<start>_<end>.json)",
    )
    return parser.parse_args()


def default_output_path(timeframe: str, start: datetime, end: datetime) -> Path:
    filename = f"candles_{timeframe}_{start.date()}_{end.date()}.json"
    return DEFAULT_OUTPUT_DIR / filename


def main() -> None:
    args = parse_args()
    secids = _normalize_secids(args.secids)
    start_dt = _parse_date(args.start_date, end_of_day=False)
    end_dt = _parse_date(args.end_date, end_of_day=True)
    df = load_candles(secids, args.timeframe, start_dt, end_dt)
    summary = evaluate_quality(df, args.timeframe, start_dt, end_dt)
    output_path = args.output or default_output_path(args.timeframe, start_dt, end_dt)
    write_report(summary, output_path)
    print(summary["text_summary"])
    print(f"Report written to {output_path}")


if __name__ == "__main__":
    main()
