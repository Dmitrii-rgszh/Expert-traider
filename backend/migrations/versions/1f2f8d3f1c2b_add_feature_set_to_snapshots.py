"""Add feature_set column to train_data_snapshots

Revision ID: 1f2f8d3f1c2b
Revises: 0f68bb77f483
Create Date: 2025-11-16 12:30:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "1f2f8d3f1c2b"
down_revision: Union[str, Sequence[str], None] = "0f68bb77f483"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("train_data_snapshots") as batch_op:
        batch_op.add_column(sa.Column("feature_set", sa.String(length=64), nullable=False, server_default="tech_v1"))
        batch_op.drop_constraint("uq_train_data_snapshots_range", type_="unique")
        batch_op.create_unique_constraint(
            "uq_train_data_snapshots_range",
            ["secid", "timeframe", "feature_set", "snapshot_start", "snapshot_end"],
        )
    op.execute("UPDATE train_data_snapshots SET feature_set='tech_v1'")


def downgrade() -> None:
    with op.batch_alter_table("train_data_snapshots") as batch_op:
        batch_op.drop_constraint("uq_train_data_snapshots_range", type_="unique")
        batch_op.create_unique_constraint(
            "uq_train_data_snapshots_range",
            ["secid", "timeframe", "snapshot_start", "snapshot_end"],
        )
        batch_op.drop_column("feature_set")
