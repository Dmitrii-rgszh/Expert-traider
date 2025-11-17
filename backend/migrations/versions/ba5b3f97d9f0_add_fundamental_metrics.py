"""add fundamental metrics table"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "ba5b3f97d9f0"
down_revision = "3c2d843c3956"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "fundamental_metrics",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("secid", sa.String(length=32), nullable=False),
        sa.Column("metric_type", sa.String(length=64), nullable=False),
        sa.Column("metric_date", sa.DateTime(timezone=True), nullable=False),
        sa.Column("period_end", sa.DateTime(timezone=True), nullable=True),
        sa.Column("metric_value", sa.Float(), nullable=False),
        sa.Column("currency", sa.String(length=8), nullable=True),
        sa.Column("source", sa.String(length=64), nullable=True),
        sa.Column("reliability_score", sa.Float(), nullable=True),
        sa.Column("metadata_json", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("secid", "metric_type", "metric_date", name="uq_fundamental_metric_snapshot"),
    )
    op.create_index("ix_fundamental_metrics_secid", "fundamental_metrics", ["secid"])
    op.create_index("ix_fundamental_metrics_metric_type", "fundamental_metrics", ["metric_type"])


def downgrade() -> None:
    op.drop_index("ix_fundamental_metrics_metric_type", table_name="fundamental_metrics")
    op.drop_index("ix_fundamental_metrics_secid", table_name="fundamental_metrics")
    op.drop_table("fundamental_metrics")
