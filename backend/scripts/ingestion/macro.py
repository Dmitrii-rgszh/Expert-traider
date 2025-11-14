from __future__ import annotations

import argparse
import csv
from datetime import date, datetime
from pathlib import Path
from typing import Iterable, Optional

from backend.app.models import MacroSeries, PolicyRate
from sqlalchemy import select

from .base import IngestionStats, session_scope, logger


def parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def parse_datetime(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    return datetime.fromisoformat(value)


def load_csv_rows(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            yield row


class MacroSeriesIngestor:
    def run(self, csv_path: Optional[Path], dry_run: bool = False) -> IngestionStats:
        stats = IngestionStats()
        if not csv_path or not csv_path.exists():
            logger.warning("Macro series CSV not provided or missing: %s", csv_path)
            return stats

        with session_scope() as session:
            for row in load_csv_rows(csv_path):
                stats.processed += 1
                if dry_run:
                    continue
                payload = {
                    "series_code": row["series_code"],
                    "period_start": parse_date(row["period_start"]),
                    "period_end": parse_date(row["period_end"]),
                    "value": float(row["value"]),
                    "revision": row.get("revision") or None,
                    "source": row.get("source") or None,
                }
                stmt = select(MacroSeries).where(
                    MacroSeries.series_code == payload["series_code"],
                    MacroSeries.period_start == payload["period_start"],
                    MacroSeries.period_end == payload["period_end"],
                )
                existing = session.execute(stmt).scalar_one_or_none()
                if existing:
                    for key, value in payload.items():
                        setattr(existing, key, value)
                    stats.updated += 1
                else:
                    session.add(MacroSeries(**payload))
                    stats.inserted += 1
        return stats


class PolicyRateIngestor:
    def run(self, csv_path: Optional[Path], dry_run: bool = False) -> IngestionStats:
        stats = IngestionStats()
        if not csv_path or not csv_path.exists():
            logger.warning("Policy rate CSV not provided or missing: %s", csv_path)
            return stats

        with session_scope() as session:
            for row in load_csv_rows(csv_path):
                stats.processed += 1
                if dry_run:
                    continue
                payload = {
                    "rate_code": row["rate_code"],
                    "date": parse_date(row["date"]),
                    "value": float(row["value"]),
                    "announced_at": parse_datetime(row.get("announced_at")),
                    "effective_from": parse_date(row["effective_from"]) if row.get("effective_from") else None,
                }
                stmt = select(PolicyRate).where(
                    PolicyRate.rate_code == payload["rate_code"],
                    PolicyRate.date == payload["date"],
                )
                existing = session.execute(stmt).scalar_one_or_none()
                if existing:
                    existing.value = payload["value"]
                    stats.updated += 1
                else:
                    session.add(PolicyRate(**payload))
                    stats.inserted += 1
        return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Load macro series and policy rates from CSV files")
    parser.add_argument("--macro-csv", type=Path, help="Path to macro series CSV")
    parser.add_argument("--policy-csv", type=Path, help="Path to policy rate CSV")
    parser.add_argument("--dry-run", action="store_true", help="Do not write to DB")
    args = parser.parse_args()

    macro_stats = MacroSeriesIngestor().run(args.macro_csv, dry_run=args.dry_run)
    policy_stats = PolicyRateIngestor().run(args.policy_csv, dry_run=args.dry_run)
    logger.info("Macro ingestion: %s", macro_stats)
    logger.info("Policy rate ingestion: %s", policy_stats)


if __name__ == "__main__":
    main()
