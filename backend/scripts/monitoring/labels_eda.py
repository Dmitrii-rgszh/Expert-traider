from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd
from sqlalchemy import select

from backend.app.db.session import SessionLocal
from backend.app.models.market_data import TradeLabel

DEFAULT_OUTPUT_DIR = Path("docs/data_quality")
TIMEFRAME_MINUTES = {
    "1m": 1,
    "5m": 5,
    "15m": 15,
    "1h": 60,
    "1d": 1440,
}


def _parse_datetime(value: str) -> datetime:
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def load_labels(
    label_set: str,
    timeframe: str,
    start: datetime,
    end: datetime,
    secids: Optional[Sequence[str]],
) -> pd.DataFrame:
    with SessionLocal() as session:
        stmt = (
            select(TradeLabel)
            .where(
                TradeLabel.label_set == label_set,
                TradeLabel.timeframe == timeframe,
                TradeLabel.signal_time >= start,
                TradeLabel.signal_time <= end,
            )
            .order_by(TradeLabel.signal_time.asc())
        )
        if secids:
            stmt = stmt.where(TradeLabel.secid.in_(list(secids)))
        rows = session.execute(stmt).scalars().all()
    payload = [
        {
            "secid": row.secid,
            "signal_time": row.signal_time,
            "horizon_minutes": row.horizon_minutes,
            "label_long": row.label_long,
            "label_short": row.label_short,
            "long_pnl_pct": row.long_pnl_pct,
            "short_pnl_pct": row.short_pnl_pct,
        }
        for row in rows
    ]
    if not payload:
        return pd.DataFrame()
    df = pd.DataFrame(payload)
    df["signal_time"] = pd.to_datetime(df["signal_time"], utc=True)
    df["date"] = df["signal_time"].dt.date
    return df


def summarize_labels(
    df: pd.DataFrame,
    label_set: str,
    timeframe: str,
    start: datetime,
    end: datetime,
    expected_secids: int,
) -> dict[str, object]:
    summary = {
        "label_set": label_set,
        "timeframe": timeframe,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "horizons": {},
    }
    if df.empty:
        summary["text_summary"] = "No labels in the requested window."
        return summary

    days = (end.date() - start.date()).days + 1
    interval = TIMEFRAME_MINUTES.get(timeframe, 1)
    bars_per_day = (24 * 60) / interval
    expected_rows_per_day = bars_per_day * (expected_secids or df["secid"].nunique())
    lines = [
        f"Label EDA {label_set}/{timeframe} rows={len(df)} secids={df['secid'].nunique()} window={start.date()}..{end.date()}",
    ]

    for horizon, chunk in df.groupby("horizon_minutes"):
        rows = len(chunk)
        coverage = rows / (max(1, days) * max(1, expected_rows_per_day))
        long_ratio = float(chunk["label_long"].mean())
        short_ratio = float(chunk["label_short"].mean())
        long_expectancy = float(chunk["long_pnl_pct"].mean())
        short_expectancy = float(chunk["short_pnl_pct"].mean())
        summary["horizons"][int(horizon)] = {
            "rows": rows,
            "coverage_per_day": coverage,
            "long_positive_ratio": long_ratio,
            "short_positive_ratio": short_ratio,
            "long_expectancy": long_expectancy,
            "short_expectancy": short_expectancy,
        }
        lines.append(
            f"- H={horizon}m rows={rows} coverage/day={coverage:.2f} "
            f"long+= {long_ratio:.2%} short+= {short_ratio:.2%} "
            f"long_exp={long_expectancy:.4f} short_exp={short_expectancy:.4f}"
        )
    summary["text_summary"] = "\n".join(lines)
    return summary


def write_summary(summary: dict[str, object], output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate EDA summary for trade labels")
    parser.add_argument("--label-set", required=True, help="Label set identifier")
    parser.add_argument("--timeframe", required=True, help="Timeframe filter (e.g. 1m, 1h)")
    parser.add_argument("--start-date", required=True, help="Start ISO datetime")
    parser.add_argument("--end-date", required=True, help="End ISO datetime")
    parser.add_argument("--secids", nargs="*", help="Optional list of tickers to include")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "labels_eda.json",
        help="Destination JSON path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = _parse_datetime(args.start_date)
    end = _parse_datetime(args.end_date)
    df = load_labels(args.label_set, args.timeframe, start, end, args.secids)
    summary = summarize_labels(df, args.label_set, args.timeframe, start, end, len(args.secids or []))
    path = write_summary(summary, args.output)
    print(summary["text_summary"])
    print(f"Label EDA saved to {path}")


if __name__ == "__main__":
    main()
