"""Add trade labels table

Revision ID: 9a4dbbc5c5ac
Revises: 8e6e0a9a8c9f
Create Date: 2025-11-15 12:30:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "9a4dbbc5c5ac"
down_revision: Union[str, Sequence[str], None] = "8e6e0a9a8c9f"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "trade_labels",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("secid", sa.String(length=32), nullable=False),
        sa.Column("timeframe", sa.String(length=16), nullable=False),
        sa.Column(
            "label_set",
            sa.String(length=64),
            nullable=False,
            server_default=sa.text("'basic_v1'"),
        ),
        sa.Column("signal_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("horizon_minutes", sa.Integer(), nullable=False),
        sa.Column("take_profit_pct", sa.Float(), nullable=False),
        sa.Column("stop_loss_pct", sa.Float(), nullable=False),
        sa.Column("entry_price", sa.Numeric(18, 6), nullable=False),
        sa.Column("horizon_close", sa.Numeric(18, 6), nullable=False),
        sa.Column("forward_return_pct", sa.Float(), nullable=False),
        sa.Column("max_runup_pct", sa.Float(), nullable=False),
        sa.Column("max_drawdown_pct", sa.Float(), nullable=False),
        sa.Column("label_long", sa.Boolean(), nullable=False),
        sa.Column("label_short", sa.Boolean(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.UniqueConstraint(
            "secid",
            "timeframe",
            "signal_time",
            "horizon_minutes",
            "label_set",
            name="uq_trade_labels_unique",
        ),
    )
    op.create_index("ix_trade_labels_secid", "trade_labels", ["secid"])
    op.create_index("ix_trade_labels_timeframe", "trade_labels", ["timeframe"])
    op.create_index("ix_trade_labels_signal_time", "trade_labels", ["signal_time"])


def downgrade() -> None:
    op.drop_index("ix_trade_labels_signal_time", table_name="trade_labels")
    op.drop_index("ix_trade_labels_timeframe", table_name="trade_labels")
    op.drop_index("ix_trade_labels_secid", table_name="trade_labels")
    op.drop_table("trade_labels")
