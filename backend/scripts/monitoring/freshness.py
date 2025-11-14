from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from typing import Optional

from sqlalchemy import func, select, text, inspect

from backend.app.models import (
    DataQualityAlert,
    FxRate,
    MacroSeries,
    MarketRegime,
    NewsEvent,
    OfzYield,
    PolicyRate,
    RiskAlert,
    SanctionLink,
)
from ..ingestion.base import session_scope, logger


@dataclass
class ModelSpec:
    column: str
    model: Optional[type] = None
    table_name: Optional[str] = None


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "fx_rates": ModelSpec(model=FxRate, column="timestamp"),
    "news_events": ModelSpec(model=NewsEvent, column="published_at"),
    "market_regimes": ModelSpec(model=MarketRegime, column="timestamp"),
    "macro_series": ModelSpec(model=MacroSeries, column="period_end"),
    "policy_rates": ModelSpec(model=PolicyRate, column="date"),
    "ofz_yields": ModelSpec(model=OfzYield, column="date"),
    "risk_alerts": ModelSpec(model=RiskAlert, column="timestamp"),
    "sanction_links": ModelSpec(model=SanctionLink, column="linked_at"),
    "candles": ModelSpec(table_name="candles", column="datetime"),
    "index_candles": ModelSpec(table_name="index_candles", column="datetime"),
}


class FreshnessMonitor:
    def __init__(self) -> None:
        self.registry = MODEL_REGISTRY

    def run(self, table_name: str, max_lag_minutes: int, severity: str = "medium") -> Optional[datetime]:
        if table_name not in self.registry:
            logger.warning("Table %s not registered for freshness monitoring", table_name)
            return None
        with session_scope() as session:
            last_ts = self._fetch_last_timestamp(session, table_name)
            if not last_ts:
                self._record_alert(session, table_name, "missing-data", severity, details={"reason": "no rows"})
                return None
            lag = datetime.now(timezone.utc) - last_ts
            if lag > timedelta(minutes=max_lag_minutes):
                self._record_alert(
                    session,
                    table_name,
                    "stale-data",
                    severity,
                    details={"lag_minutes": lag.total_seconds() / 60, "threshold": max_lag_minutes},
                )
            return last_ts

    def _fetch_last_timestamp(self, session, table_name: str) -> Optional[datetime]:
        spec = self.registry[table_name]
        if spec.model:
            column = getattr(spec.model, spec.column)
            stmt = select(func.max(column))
            raw_value = session.execute(stmt).scalar_one_or_none()
            return self._normalize_ts(raw_value)
        if not spec.table_name:
            logger.warning("No table configured for %s", table_name)
            return None
        bind = session.get_bind()
        inspector = inspect(bind)
        if not inspector.has_table(spec.table_name):
            logger.warning("Table %s is missing in the current database", spec.table_name)
            return None
        stmt = text(f"SELECT MAX({spec.column}) AS max_ts FROM {spec.table_name}")
        raw_value = session.execute(stmt).scalar_one_or_none()
        return self._normalize_ts(raw_value)

    def _record_alert(
        self,
        session,
        table_name: str,
        alert_type: str,
        severity: str,
        details: dict,
    ) -> None:
        session.add(
            DataQualityAlert(
                alert_type=alert_type,
                severity=severity,
                details_json={"table": table_name, **details},
            )
        )

    @staticmethod
    def _normalize_ts(value: Optional[object]) -> Optional[datetime]:
        if value is None:
            return None
        if isinstance(value, datetime):
            return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        if isinstance(value, date):
            return datetime.combine(value, time.min, tzinfo=timezone.utc)
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            return None
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def main() -> None:
    parser = argparse.ArgumentParser(description="Check data freshness")
    parser.add_argument("table", choices=MODEL_REGISTRY.keys())
    parser.add_argument("--max-lag", type=int, default=60, help="Allowed lag in minutes")
    parser.add_argument("--severity", default="medium")
    args = parser.parse_args()

    FreshnessMonitor().run(args.table, args.max_lag, severity=args.severity)


if __name__ == "__main__":
    main()
