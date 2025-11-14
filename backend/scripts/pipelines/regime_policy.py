from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from statistics import mean
from typing import Iterable

from sqlalchemy import desc, select

from backend.app.models import (
    FxRate,
    MarketRegime,
    MarketRegimeDetail,
    PolicyFeedback,
    PolicyRun,
)
from ..ingestion.base import IngestionStats, session_scope, logger


class MarketRegimePipeline:
    def __init__(self, lookback_points: int = 5) -> None:
        self.lookback_points = lookback_points

    def run(self, dry_run: bool = False) -> IngestionStats:
        stats = IngestionStats()
        with session_scope() as session:
            stmt = (
                select(FxRate)
                .where(FxRate.pair == "USDRUB")
                .order_by(desc(FxRate.timestamp))
                .limit(self.lookback_points)
            )
            rates = [row[0].rate for row in session.execute(stmt).all() if row[0] and row[0].rate]
            if not rates:
                logger.warning("No FX data to derive market regime")
                return stats
            avg_rate = float(mean([float(r) for r in rates]))
            regime = self._classify(avg_rate)
            payload = MarketRegime(
                timestamp=datetime.now(timezone.utc),
                scope="market",
                scope_value="MOEX",
                value=regime,
                probabilities_json={"panic": float(avg_rate > 100), "normal": float(avg_rate <= 90)},
            )
            if dry_run:
                logger.info("Derived regime (dry-run): %s", regime)
                return stats
            session.add(payload)
            session.flush()
            detail = MarketRegimeDetail(
                timestamp=payload.timestamp,
                scope="market",
                scope_value="MOEX",
                base_regime_id=payload.id,
                liquidity_regime="tight" if avg_rate > 95 else "stable",
                spread_regime="wide" if avg_rate > 100 else "normal",
                news_burst_level=2,
                derived_from="fx_rates",
            )
            session.add(detail)
            stats.inserted += 2
        return stats

    @staticmethod
    def _classify(avg_rate: float) -> str:
        if avg_rate >= 100:
            return "panic"
        if avg_rate >= 90:
            return "stress"
        return "normal"


class PolicyFeedbackPipeline:
    def __init__(self, policy_name: str, version: str) -> None:
        self.policy_name = policy_name
        self.version = version

    def run(self, events: Iterable[dict], dry_run: bool = False) -> IngestionStats:
        stats = IngestionStats()
        with session_scope() as session:
            policy_run = self._ensure_run(session)
            for event in events:
                stats.processed += 1
                if dry_run:
                    continue
                session.add(
                    PolicyFeedback(
                        policy_run_id=policy_run.id,
                        timestamp=self._parse_timestamp(event.get("timestamp")),
                        user_id=event.get("user_id"),
                        secid=event.get("secid"),
                        context_hash=event.get("context_hash", "unknown"),
                        chosen_action=event.get("action", "hold"),
                        reward=event.get("reward"),
                        reward_components_json=event.get("reward_components"),
                    )
                )
                stats.inserted += 1
        return stats

    def _ensure_run(self, session) -> PolicyRun:
        stmt = select(PolicyRun).where(
            PolicyRun.policy_name == self.policy_name,
            PolicyRun.version == self.version,
        )
        policy_run = session.execute(stmt).scalar_one_or_none()
        if policy_run:
            return policy_run
        policy_run = PolicyRun(
            policy_name=self.policy_name,
            version=self.version,
            start_at=datetime.now(timezone.utc),
            config_json={"source": "pipeline"},
            status="running",
        )
        session.add(policy_run)
        session.flush()
        return policy_run

    @staticmethod
    def _parse_timestamp(value: str | None) -> datetime:
        if not value:
            return datetime.now(timezone.utc)
        ts = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return ts


def main() -> None:
    parser = argparse.ArgumentParser(description="Run market regime and policy feedback pipelines")
    parser.add_argument("--policy-name", default="contextual_bandit", help="Policy name")
    parser.add_argument("--policy-version", default="v1", help="Policy version")
    parser.add_argument("--events-json", type=argparse.FileType("r"), help="Policy feedback events JSON")
    parser.add_argument("--dry-run", action="store_true", help="Do not write to DB")
    args = parser.parse_args()

    regime_stats = MarketRegimePipeline().run(dry_run=args.dry_run)
    logger.info("Market regime pipeline stats: %s", regime_stats)

    events = []
    if args.events_json:
        events = json.load(args.events_json)
    policy_stats = PolicyFeedbackPipeline(args.policy_name, args.policy_version).run(events, dry_run=args.dry_run)
    logger.info("Policy feedback pipeline stats: %s", policy_stats)


if __name__ == "__main__":
    main()
