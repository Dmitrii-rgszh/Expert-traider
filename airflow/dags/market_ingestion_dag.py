from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG  # type: ignore
from airflow.operators.python import PythonOperator  # type: ignore

from backend.scripts.ingestion.calendar import CalendarIngestor
from backend.scripts.ingestion.fx import FxIngestor
from backend.scripts.ingestion.news import MoexNewsClient, NewsIngestor
from backend.scripts.ingestion.sanctions import SanctionIngestor, SanctionsApiClient
from backend.scripts.monitoring.freshness import FreshnessMonitor

DEFAULT_ARGS = {
    "owner": "auto-trader",
    "depends_on_past": False,
    "email_on_failure": True,
    "email": ["alerts@glazok.site"],
    "retries": 2,
    "retry_delay": timedelta(minutes=10),
}


def run_calendar() -> None:
    today = datetime.utcnow().date()
    start = today - timedelta(days=1)
    CalendarIngestor().run(start, today)


def run_fx() -> None:
    FxIngestor().run(["USDRUB", "EURRUB", "CNYRUB"])


def run_news() -> None:
    NewsIngestor(moex_client=MoexNewsClient()).run()


def run_sanctions() -> None:
    SanctionIngestor(api_client=SanctionsApiClient()).run()


def check_freshness(table: str, threshold_minutes: int) -> None:
    FreshnessMonitor().run(table, threshold_minutes, severity="high")


with DAG(
    dag_id="market_ingestion",
    default_args=DEFAULT_ARGS,
    start_date=datetime(2024, 1, 1),
    schedule="0 4 * * 1-5",
    catchup=False,
    max_active_runs=1,
    tags=["market-data", "moex"],
) as dag:
    calendar_task = PythonOperator(task_id="calendar", python_callable=run_calendar)
    fx_task = PythonOperator(task_id="fx", python_callable=run_fx)
    news_task = PythonOperator(task_id="news", python_callable=run_news)
    sanctions_task = PythonOperator(task_id="sanctions", python_callable=run_sanctions)
    fx_freshness = PythonOperator(
        task_id="fx_freshness",
        python_callable=check_freshness,
        op_kwargs={"table": "fx_rates", "threshold_minutes": 60},
    )
    news_freshness = PythonOperator(
        task_id="news_freshness",
        python_callable=check_freshness,
        op_kwargs={"table": "news_events", "threshold_minutes": 30},
    )

    [calendar_task, fx_task, news_task, sanctions_task]
    fx_task >> fx_freshness
    news_task >> news_freshness
