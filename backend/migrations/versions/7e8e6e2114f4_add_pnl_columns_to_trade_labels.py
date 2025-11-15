"""Add PnL columns to trade_labels

Revision ID: 7e8e6e2114f4
Revises: 1f2f8d3f1c2b
Create Date: 2025-11-16 12:45:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "7e8e6e2114f4"
down_revision: Union[str, Sequence[str], None] = "1f2f8d3f1c2b"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("trade_labels") as batch_op:
        batch_op.add_column(sa.Column("long_pnl_pct", sa.Float(), nullable=False, server_default="0"))
        batch_op.add_column(sa.Column("short_pnl_pct", sa.Float(), nullable=False, server_default="0"))


def downgrade() -> None:
    with op.batch_alter_table("trade_labels") as batch_op:
        batch_op.drop_column("short_pnl_pct")
        batch_op.drop_column("long_pnl_pct")
