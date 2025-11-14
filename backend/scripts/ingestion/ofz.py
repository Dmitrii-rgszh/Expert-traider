from __future__ import annotations

import argparse
from datetime import date, datetime
from typing import Iterable

from sqlalchemy import select

from backend.app.models import OfzAuction, OfzYield
from .base import HttpSource, IngestionStats, fetch_json, session_scope, logger

YIELDS_SOURCE = HttpSource(
    url="https://iss.moex.com/iss/statistics/engines/stock/ofz/yields.json",
    params={"lang": "ru"},
)
AUCTIONS_SOURCE = HttpSource(
    url="https://iss.moex.com/iss/statistics/engines/stock/ofz/auctions.json",
    params={"lang": "ru"},
)


def parse_date_safe(value: str | None) -> date | None:
    if not value:
        return None
    return datetime.strptime(value, "%Y-%m-%d").date()


class OfzYieldIngestor:
    def __init__(self, source: HttpSource = YIELDS_SOURCE) -> None:
        self.source = source

    def run(self, dry_run: bool = False) -> IngestionStats:
        stats = IngestionStats()
        payload = fetch_json(self.source)
        if not payload:
            logger.warning("Empty OFZ yields payload")
            return stats
        rows = self._extract(payload)
        with session_scope() as session:
            for row in rows:
                stats.processed += 1
                if dry_run:
                    continue
                stmt = select(OfzYield).where(OfzYield.isin == row["isin"], OfzYield.date == row["date"])
                existing = session.execute(stmt).scalar_one_or_none()
                if existing:
                    for key, value in row.items():
                        setattr(existing, key, value)
                    stats.updated += 1
                else:
                    session.add(OfzYield(**row))
                    stats.inserted += 1
        return stats

    def _extract(self, payload: dict) -> Iterable[dict]:
        block = payload.get("yields") or {}
        columns = block.get("columns") or []
        data = block.get("data") or []
        idx = {name: i for i, name in enumerate(columns)}
        for row in data:
            if not row:
                continue
            trade_date = parse_date_safe(row[idx.get("TRADEDATE", 2)])
            if not trade_date:
                continue
            yield {
                "isin": row[idx.get("ISIN", 0)],
                "maturity_date": parse_date_safe(row[idx.get("MATDATE", 1)]),
                "date": trade_date,
                "ytm": float(row[idx.get("YIELD", 3)]) if row[idx.get("YIELD", 3)] else None,
                "dirty_price": float(row[idx.get("DIRTYPRICE", 4)]) if row[idx.get("DIRTYPRICE", 4)] else None,
                "duration": float(row[idx.get("DURATION", 5)]) if row[idx.get("DURATION", 5)] else None,
                "convexity": float(row[idx.get("CONVEXITY", 6)]) if row[idx.get("CONVEXITY", 6)] else None,
                "coupon": float(row[idx.get("COUPONPERCENT", 7)]) if row[idx.get("COUPONPERCENT", 7)] else None,
                "next_coupon_date": parse_date_safe(row[idx.get("NEXTCOUPON", 8)]),
            }


class OfzAuctionIngestor:
    def __init__(self, source: HttpSource = AUCTIONS_SOURCE) -> None:
        self.source = source

    def run(self, dry_run: bool = False) -> IngestionStats:
        stats = IngestionStats()
        payload = fetch_json(self.source)
        if not payload:
            logger.warning("Empty OFZ auction payload")
            return stats
        rows = self._extract(payload)
        with session_scope() as session:
            for row in rows:
                stats.processed += 1
                if dry_run:
                    continue
                stmt = select(OfzAuction).where(
                    OfzAuction.isin == row["isin"],
                    OfzAuction.auction_date == row["auction_date"],
                )
                existing = session.execute(stmt).scalar_one_or_none()
                if existing:
                    for key, value in row.items():
                        setattr(existing, key, value)
                    stats.updated += 1
                else:
                    session.add(OfzAuction(**row))
                    stats.inserted += 1
        return stats

    def _extract(self, payload: dict) -> Iterable[dict]:
        block = payload.get("auctions") or {}
        columns = block.get("columns") or []
        data = block.get("data") or []
        idx = {name: i for i, name in enumerate(columns)}
        for row in data:
            if not row:
                continue
            trade_date = parse_date_safe(row[idx.get("TRADEDATE", 0)])
            if not trade_date:
                continue
            yield {
                "auction_date": trade_date,
                "isin": row[idx.get("ISIN", 1)],
                "offered": float(row[idx.get("VOLUME", 2)]) if row[idx.get("VOLUME", 2)] else None,
                "placed": float(row[idx.get("PLACED", 3)]) if row[idx.get("PLACED", 3)] else None,
                "yield_min": float(row[idx.get("YIELD_MIN", 4)]) if row[idx.get("YIELD_MIN", 4)] else None,
                "yield_avg": float(row[idx.get("YIELD_AVG", 5)]) if row[idx.get("YIELD_AVG", 5)] else None,
                "yield_max": float(row[idx.get("YIELD_MAX", 6)]) if row[idx.get("YIELD_MAX", 6)] else None,
                "bid_cover": float(row[idx.get("BIDCOVER", 7)]) if row[idx.get("BIDCOVER", 7)] else None,
                "notes": row[idx.get("COMMENTS", 8)] or None,
            }


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest OFZ yields and auctions")
    parser.add_argument("--skip-yields", action="store_true", help="Skip yields ingestion")
    parser.add_argument("--skip-auctions", action="store_true", help="Skip auction ingestion")
    parser.add_argument("--dry-run", action="store_true", help="Do not write to DB")
    args = parser.parse_args()

    if not args.skip_yields:
        yield_stats = OfzYieldIngestor().run(dry_run=args.dry_run)
        logger.info("OFZ yields ingestion: %s", yield_stats)
    if not args.skip_auctions:
        auction_stats = OfzAuctionIngestor().run(dry_run=args.dry_run)
        logger.info("OFZ auctions ingestion: %s", auction_stats)


if __name__ == "__main__":
    main()
