from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

from prefect import flow, task

from backend.scripts.ingestion.base import logger
from backend.scripts.ingestion.calendar import CalendarIngestor, parse_date
from backend.scripts.ingestion.fx import FxIngestor
from backend.scripts.ingestion.macro import MacroSeriesIngestor, PolicyRateIngestor
from backend.scripts.ingestion.news import MoexNewsClient, NewsIngestor
from backend.scripts.ingestion.ofz import OfzAuctionIngestor, OfzYieldIngestor
from backend.scripts.ingestion.sanctions import SanctionIngestor, SanctionsApiClient
from backend.scripts.monitoring.freshness import FreshnessMonitor
from backend.scripts.pipelines.regime_policy import MarketRegimePipeline, PolicyFeedbackPipeline


@task(name="ingest-calendar")
def ingest_calendar(start: date, end: date) -> None:
    stats = CalendarIngestor().run(start, end)
    logger.info("Calendar stats: %s", stats)


@task(name="ingest-fx")
def ingest_fx() -> None:
    FxIngestor().run(["USDRUB", "EURRUB", "CNYRUB"])


@task(name="ingest-news")
def ingest_news(json_file: Optional[Path] = None) -> None:
    file_arg = json_file if json_file and json_file.exists() else None
    moex_client = None if file_arg else MoexNewsClient()
    NewsIngestor(file_path=file_arg, moex_client=moex_client).run()


@task(name="ingest-sanctions")
def ingest_sanctions(file_path: Optional[Path], use_api: bool = True) -> None:
    file_arg = file_path if file_path and file_path.exists() else None
    api_client = SanctionsApiClient() if use_api or not file_arg else None
    SanctionIngestor(file_path=file_arg, api_client=api_client).run()


@task(name="ingest-ofz")
def ingest_ofz() -> None:
    OfzYieldIngestor().run()
    OfzAuctionIngestor().run()


@task(name="ingest-macro")
def ingest_macro(macro_csv: Optional[Path], policy_csv: Optional[Path]) -> None:
    MacroSeriesIngestor().run(macro_csv)
    PolicyRateIngestor().run(policy_csv)


@task(name="run-regime-pipeline")
def run_regime_pipeline() -> None:
    MarketRegimePipeline().run()


@task(name="run-policy-feedback")
def run_policy_feedback(events_path: Optional[Path]) -> None:
    events: list[dict] = []
    if events_path and events_path.exists():
        events = json.loads(events_path.read_text(encoding="utf-8"))
    PolicyFeedbackPipeline("contextual_bandit", "v1").run(events)


@task(name="freshness-check")
def freshness_check(table: str, max_lag_minutes: int, severity: str = "medium") -> None:
    FreshnessMonitor().run(table, max_lag_minutes, severity=severity)


@flow(name="daily-ingestion-flow")
def daily_ingestion_flow(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    news_json: Optional[str] = None,
    sanctions_file: Optional[str] = None,
    sanctions_use_api: bool = True,
    macro_csv: Optional[str] = None,
    policy_csv: Optional[str] = None,
) -> None:
    today = date.today()
    start_dt = parse_date(start_date) if start_date else today - timedelta(days=1)
    end_dt = parse_date(end_date) if end_date else today
    ingest_calendar(start_dt, end_dt)
    ingest_fx()
    ingest_news(Path(news_json) if news_json else None)
    ingest_sanctions(Path(sanctions_file) if sanctions_file else None, sanctions_use_api)
    ingest_ofz()
    ingest_macro(Path(macro_csv) if macro_csv else None, Path(policy_csv) if policy_csv else None)
    freshness_check("fx_rates", 60, "high")
    freshness_check("news_events", 30, "high")
    freshness_check("sanction_links", 1440, "medium")


@flow(name="regime-policy-flow")
def regime_policy_flow(events_json: Optional[str] = None) -> None:
    run_regime_pipeline()
    run_policy_feedback(Path(events_json) if events_json else None)
    freshness_check("market_regimes", 120, "medium")


if __name__ == "__main__":
    today = date.today()
    yesterday = today - timedelta(days=1)
    daily_ingestion_flow(start_date=str(yesterday), end_date=str(today))
    regime_policy_flow()
