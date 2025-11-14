from __future__ import annotations

from datetime import timedelta

from prefect.client.schemas.schedules import CronSchedule, IntervalSchedule

from .prefect_flows import daily_ingestion_flow, regime_policy_flow


def deploy_flows() -> None:
    """Register Prefect deployments using Flow.deploy API."""
    daily_ingestion_flow.deploy(
        name="daily-ingestion-prod",
        work_pool_name="default-agent-pool",
        schedules=[
            CronSchedule(cron="5 4 * * 1-5", timezone="Europe/Moscow"),
        ],
        parameters={
            "sanctions_use_api": True,
        },
        description="Daily MOEX/macro ingestion with freshness checks",
        tags=["ingestion", "prod"],
        paused=False,
    )

    regime_policy_flow.deploy(
        name="regime-policy-hourly",
        work_pool_name="default-agent-pool",
        schedules=[
            IntervalSchedule(interval=timedelta(hours=1)),
        ],
        description="Market regime derivation + bandit feedback",
        tags=["signals", "bandits"],
        paused=False,
    )


if __name__ == "__main__":
    deploy_flows()
