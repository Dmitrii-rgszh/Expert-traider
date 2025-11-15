"""Add candles and feature store tables

Revision ID: 8e6e0a9a8c9f
Revises: 6f6c1b244f4e
Create Date: 2025-11-15 10:05:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "8e6e0a9a8c9f"
down_revision: Union[str, Sequence[str], None] = "6f6c1b244f4e"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "candles",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("secid", sa.String(length=32), nullable=False),
        sa.Column("board", sa.String(length=16), nullable=False, server_default=sa.text("'TQBR'")),
        sa.Column("timeframe", sa.String(length=16), nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("open", sa.Numeric(18, 6), nullable=False),
        sa.Column("high", sa.Numeric(18, 6), nullable=False),
        sa.Column("low", sa.Numeric(18, 6), nullable=False),
        sa.Column("close", sa.Numeric(18, 6), nullable=False),
        sa.Column("volume", sa.BigInteger(), nullable=True),
        sa.Column("value", sa.Numeric(20, 2), nullable=True),
        sa.Column("trades", sa.Integer(), nullable=True),
        sa.Column(
            "ingested_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.UniqueConstraint("secid", "board", "timeframe", "timestamp", name="uq_candles_series"),
    )

    op.create_table(
        "index_candles",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("index_code", sa.String(length=32), nullable=False),
        sa.Column("timeframe", sa.String(length=16), nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("open", sa.Numeric(18, 6), nullable=False),
        sa.Column("high", sa.Numeric(18, 6), nullable=False),
        sa.Column("low", sa.Numeric(18, 6), nullable=False),
        sa.Column("close", sa.Numeric(18, 6), nullable=False),
        sa.Column("volume", sa.BigInteger(), nullable=True),
        sa.Column("value", sa.Numeric(20, 2), nullable=True),
        sa.Column(
            "ingested_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.UniqueConstraint("index_code", "timeframe", "timestamp", name="uq_index_candles_series"),
    )

    op.create_table(
        "feature_windows",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("secid", sa.String(length=32), nullable=False),
        sa.Column("timeframe", sa.String(length=16), nullable=False),
        sa.Column("window_start", sa.DateTime(timezone=True), nullable=False),
        sa.Column("window_end", sa.DateTime(timezone=True), nullable=False),
        sa.Column("feature_set", sa.String(length=64), nullable=False),
        sa.Column("generated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("checksum", sa.String(length=64), nullable=True),
        sa.UniqueConstraint(
            "secid",
            "timeframe",
            "window_start",
            "window_end",
            "feature_set",
            name="uq_feature_windows_range",
        ),
    )

    op.create_table(
        "feature_numeric",
        sa.Column("feature_window_id", sa.Integer(), nullable=False),
        sa.Column("feature_name", sa.String(length=128), nullable=False),
        sa.Column("value_numeric", sa.Float(), nullable=False),
        sa.ForeignKeyConstraint(
            ["feature_window_id"],
            ["feature_windows.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("feature_window_id", "feature_name"),
    )

    op.create_table(
        "feature_categorical",
        sa.Column("feature_window_id", sa.Integer(), nullable=False),
        sa.Column("feature_name", sa.String(length=128), nullable=False),
        sa.Column("value_text", sa.String(length=256), nullable=False),
        sa.ForeignKeyConstraint(
            ["feature_window_id"],
            ["feature_windows.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("feature_window_id", "feature_name"),
    )


def downgrade() -> None:
    op.drop_table("feature_categorical")
    op.drop_table("feature_numeric")
    op.drop_table("feature_windows")
    op.drop_table("index_candles")
    op.drop_table("candles")
