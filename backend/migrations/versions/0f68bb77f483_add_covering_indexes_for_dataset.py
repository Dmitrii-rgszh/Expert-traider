"""Add covering indexes for dataset joins

Revision ID: 0f68bb77f483
Revises: 9a4dbbc5c5ac
Create Date: 2025-11-16 11:20:00.000000

"""
from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = "0f68bb77f483"
down_revision: Union[str, Sequence[str], None] = "9a4dbbc5c5ac"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_index(
        "ix_feature_windows_timeframe_feature_set_window_end_secid",
        "feature_windows",
        ["timeframe", "feature_set", "window_end", "secid"],
        unique=False,
    )
    op.create_index(
        "ix_trade_labels_timeframe_label_set_signal_time_secid",
        "trade_labels",
        ["timeframe", "label_set", "signal_time", "secid"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_trade_labels_timeframe_label_set_signal_time_secid",
        table_name="trade_labels",
    )
    op.drop_index(
        "ix_feature_windows_timeframe_feature_set_window_end_secid",
        table_name="feature_windows",
    )
