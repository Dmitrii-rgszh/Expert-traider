from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from backend.app.models.fundamental import FundamentalMetric
from .base import IngestionStats, logger, session_scope

SUPPORTED_METRICS = {
    "earnings_yoy",
    "dividend_yield",
    "net_debt_to_ebitda",
    "sanction_score",
    "revenue_yoy",
}


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _load_payloads(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        return payload.get("companies", []) or []
    return []


def _records_from_entry(entry: dict[str, Any]) -> Iterable[dict[str, Any]]:
    secid = str(entry.get("secid", "")).upper().strip()
    if not secid:
        return []
    metrics = entry.get("metrics") or []
    for metric in metrics:
        metric_type = str(metric.get("metric_type", "")).strip().lower()
        if metric_type and metric_type not in SUPPORTED_METRICS:
            continue
        metric_date = _parse_dt(metric.get("metric_date")) or _parse_dt(metric.get("reported_at"))
        if metric_date is None:
            continue
        yield {
            "secid": secid,
            "metric_type": metric_type,
            "metric_date": metric_date,
            "period_end": _parse_dt(metric.get("period_end")),
            "metric_value": float(metric.get("value")),
            "currency": metric.get("currency"),
            "source": metric.get("source"),
            "reliability_score": metric.get("reliability_score"),
            "metadata_json": metric,
        }


def ingest_metrics(json_file: Path, dry_run: bool = False) -> IngestionStats:
    stats = IngestionStats()
    payloads = _load_payloads(json_file)
    if not payloads:
        logger.warning("No fundamental payloads found in %s", json_file)
        return stats

    with session_scope() as session:
        for entry in payloads:
            for record in _records_from_entry(entry):
                stats.processed += 1
                if dry_run:
                    continue
                existing = (
                    session.query(FundamentalMetric)
                    .filter(
                        FundamentalMetric.secid == record["secid"],
                        FundamentalMetric.metric_type == record["metric_type"],
                        FundamentalMetric.metric_date == record["metric_date"],
                    )
                    .one_or_none()
                )
                if existing:
                    for key, value in record.items():
                        setattr(existing, key, value)
                    stats.updated += 1
                else:
                    session.add(FundamentalMetric(**record))
                    stats.inserted += 1
    return stats


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ingest structured fundamentals from JSON payload")
    parser.add_argument("--json-file", type=Path, required=True, help="Path to JSON payload with metrics")
    parser.add_argument("--dry-run", action="store_true", help="Parse payload without writing to DB")
    return parser


def main() -> None:
    parser = build_cli()
    args = parser.parse_args()
    stats = ingest_metrics(args.json_file, dry_run=args.dry_run)
    logger.info("Fundamental ingestion stats: processed=%s inserted=%s updated=%s", stats.processed, stats.inserted, stats.updated)


if __name__ == "__main__":
    main()
