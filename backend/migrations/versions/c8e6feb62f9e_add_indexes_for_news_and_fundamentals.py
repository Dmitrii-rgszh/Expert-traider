"""Add indexes for news/fundamental lookups

Revision ID: c8e6feb62f9e
Revises: ba5b3f97d9f0
Create Date: 2025-11-17 12:30:00.000000
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "c8e6feb62f9e"
down_revision: Union[str, Sequence[str], None] = "ba5b3f97d9f0"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_index(
        "ix_news_events_published_at",
        "news_events",
        ["published_at"],
        unique=False,
    )
    op.create_index(
        "ix_news_events_secid_published_at",
        "news_events",
        ["secid", "published_at"],
        unique=False,
    )
    op.create_index(
        "ix_risk_alerts_timestamp",
        "risk_alerts",
        ["timestamp"],
        unique=False,
    )
    op.create_index(
        "ix_risk_alerts_secid_timestamp",
        "risk_alerts",
        ["secid", "timestamp"],
        unique=False,
    )
    op.create_index(
        "ix_fundamental_metrics_metric_date",
        "fundamental_metrics",
        ["metric_date"],
        unique=False,
    )
    op.create_index(
        "ix_fundamental_metrics_metric_date_secid",
        "fundamental_metrics",
        ["metric_date", "secid"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_fundamental_metrics_metric_date_secid", table_name="fundamental_metrics")
    op.drop_index("ix_fundamental_metrics_metric_date", table_name="fundamental_metrics")
    op.drop_index("ix_risk_alerts_secid_timestamp", table_name="risk_alerts")
    op.drop_index("ix_risk_alerts_timestamp", table_name="risk_alerts")
    op.drop_index("ix_news_events_secid_published_at", table_name="news_events")
    op.drop_index("ix_news_events_published_at", table_name="news_events")
