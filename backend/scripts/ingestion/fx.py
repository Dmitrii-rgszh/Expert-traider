from __future__ import annotations

import argparse
from datetime import datetime, timezone
from typing import Iterable, Sequence

from backend.app.models import FxRate
from sqlalchemy import select
from .base import HttpSource, IngestionStats, fetch_json, session_scope, logger

CBR_SOURCE = HttpSource(url="https://www.cbr-xml-daily.ru/daily_json.js")


class FxIngestor:
    def __init__(self, source: HttpSource = CBR_SOURCE) -> None:
        self.source = source

    def run(self, pairs: Sequence[str], dry_run: bool = False) -> IngestionStats:
        payload = fetch_json(self.source)
        stats = IngestionStats()
        if not payload:
            logger.warning("FX payload empty")
            return stats
        series = self._extract(payload, pairs)
        with session_scope() as session:
            for item in series:
                stats.processed += 1
                if dry_run:
                    continue
                stmt = select(FxRate).where(FxRate.pair == item["pair"], FxRate.timestamp == item["timestamp"])
                record = session.execute(stmt).scalar_one_or_none()
                if record:
                    record.rate = item["rate"]
                    record.provider = item.get("provider")
                    stats.updated += 1
                else:
                    session.add(FxRate(**item))
                    stats.inserted += 1
        return stats

    def _extract(self, payload: dict, pairs: Sequence[str]) -> Iterable[dict]:
        valutes = payload.get("Valute") or {}
        timestamp = (
            datetime.fromisoformat(payload.get("Timestamp"))
            if payload.get("Timestamp")
            else datetime.now(timezone.utc)
        )
        for pair in pairs:
            base = pair[:3]
            quote = pair[3:]
            valute = valutes.get(base)
            if not valute:
                continue
            value = float(valute["Value"])
            nominal = float(valute["Nominal"])
            if quote.upper() != "RUB":
                continue
            yield {
                "pair": pair.upper(),
                "timestamp": timestamp,
                "rate": value / nominal,
                "provider": "cbr",
            }


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest FX rates from CBR JSON feed")
    parser.add_argument(
        "pairs",
        nargs="*",
        default=["USDRUB", "EURRUB", "CNYRUB"],
        help="Pairs to ingest (default: major RUB)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Do not write to DB")
    args = parser.parse_args()

    stats = FxIngestor().run(args.pairs, dry_run=args.dry_run)
    logger.info("FX ingestion done: %s", stats)


if __name__ == "__main__":
    main()
