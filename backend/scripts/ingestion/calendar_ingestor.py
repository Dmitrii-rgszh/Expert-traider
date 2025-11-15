from __future__ import annotations

import argparse
from datetime import date, datetime, time
from typing import Iterable, Optional

from sqlalchemy import select

from backend.app.models import ExchangeCalendar, ScheduleChange
from .base import HttpSource, IngestionStats, fetch_json, session_scope, logger

CALENDAR_SOURCE = HttpSource(
    url="https://iss.moex.com/iss/engines/stock/markets/shares/sessions.json",
    params={"lang": "ru"},
)


def parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def parse_time(value: Optional[str]) -> Optional[time]:
    if not value:
        return None
    try:
        return datetime.strptime(value, "%H:%M:%S").time()
    except ValueError:
        return None


class CalendarIngestor:
    def __init__(self, source: HttpSource = CALENDAR_SOURCE) -> None:
        self.source = source

    def run(self, start: date, end: date, dry_run: bool = False) -> IngestionStats:
        payload = fetch_json(self.source)
        stats = IngestionStats()
        if not payload:
            logger.warning("Empty payload for exchange calendar")
            return stats

        records = self._extract(payload, start, end)
        with session_scope() as session:
            for record in records:
                stats.processed += 1
                if dry_run:
                    continue
                existing = session.get(ExchangeCalendar, record["date"])
                if existing:
                    for key, value in record.items():
                        setattr(existing, key, value)
                    stats.updated += 1
                else:
                    session.add(ExchangeCalendar(**record))
                    stats.inserted += 1
        return stats

    def _extract(self, payload: dict, start: date, end: date) -> Iterable[dict]:
        data = payload.get("calendar", {}).get("data") or []
        columns = payload.get("calendar", {}).get("columns") or []
        idx = {name: i for i, name in enumerate(columns)}
        for row in data:
            raw_date = row[idx.get("date", 0)]
            if not raw_date:
                continue
            session_date = datetime.strptime(raw_date, "%Y-%m-%d").date()
            if session_date < start or session_date > end:
                continue
            yield {
                "date": session_date,
                "is_trading_day": str(row[idx.get("is_trading_day", 1)] or "0") == "1",
                "session_open": parse_time(row[idx.get("begin", 2)]),
                "session_close": parse_time(row[idx.get("end", 3)]),
                "notes": row[idx.get("reason", 4)] or None,
            }


class ScheduleChangeIngestor:
    def __init__(self) -> None:
        self.source = HttpSource(
            url="https://iss.moex.com/iss/statistics/engines/stock/schedule.json",
            params={"lang": "ru"},
        )

    def run(self, dry_run: bool = False) -> IngestionStats:
        payload = fetch_json(self.source)
        stats = IngestionStats()
        if not payload:
            logger.warning("Empty payload for schedule changes")
            return stats
        data = payload.get("schedule", {}).get("data") or []
        columns = payload.get("schedule", {}).get("columns") or []
        idx = {name: i for i, name in enumerate(columns)}
        with session_scope() as session:
            for row in data:
                stats.processed += 1
                if dry_run:
                    continue
                effective_date = datetime.strptime(row[idx.get("date", 0)], "%Y-%m-%d").date()
                change_type = row[idx.get("type", 1)] or "unknown"
                existing = session.execute(
                    select(ScheduleChange).where(
                        ScheduleChange.effective_date == effective_date,
                        ScheduleChange.change_type == change_type,
                    )
                ).scalar_one_or_none()

                if existing:
                    existing.details_json = {"raw": row}
                    existing.source_ref = row[idx.get("link", 2)]
                    stats.updated += 1
                else:
                    session.add(
                        ScheduleChange(
                            effective_date=effective_date,
                            change_type=change_type,
                            details_json={"raw": row},
                            source_ref=row[idx.get("link", 2)],
                        )
                    )
                    stats.inserted += 1
        return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest MOEX exchange calendar")
    parser.add_argument("--start-date", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, help="End date YYYY-MM-DD")
    parser.add_argument("--dry-run", action="store_true", help="Do not write to DB")
    args = parser.parse_args()

    ingestor = CalendarIngestor()
    stats = ingestor.run(parse_date(args.start_date), parse_date(args.end_date), args.dry_run)
    logger.info("Calendar ingestion done: %s", stats)

    changes = ScheduleChangeIngestor()
    change_stats = changes.run(args.dry_run)
    logger.info("Schedule changes ingestion done: %s", change_stats)


if __name__ == "__main__":
    main()
