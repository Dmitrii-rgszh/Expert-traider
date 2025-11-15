"""Add analysis feedback table

Revision ID: c3d2f9f4cfec
Revises: 9a4dbbc5c5ac
Create Date: 2025-11-15 15:30:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "c3d2f9f4cfec"
down_revision: Union[str, Sequence[str], None] = "9a4dbbc5c5ac"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "analysis_feedback",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("analysis_id", sa.Integer(), sa.ForeignKey("analysis_results.id"), nullable=False),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id"), nullable=True),
        sa.Column("telegram_id", sa.String(length=64), nullable=True),
        sa.Column("verdict", sa.String(length=32), nullable=False),
        sa.Column("comment", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
            server_onupdate=sa.func.now(),
        ),
        sa.UniqueConstraint("analysis_id", "user_id", name="uq_analysis_feedback_user"),
        sa.UniqueConstraint("analysis_id", "telegram_id", name="uq_analysis_feedback_telegram"),
    )
    op.create_index("ix_analysis_feedback_analysis_id", "analysis_feedback", ["analysis_id"])
    op.create_index("ix_analysis_feedback_user_id", "analysis_feedback", ["user_id"])
    op.create_index("ix_analysis_feedback_telegram_id", "analysis_feedback", ["telegram_id"])


def downgrade() -> None:
    op.drop_index("ix_analysis_feedback_telegram_id", table_name="analysis_feedback")
    op.drop_index("ix_analysis_feedback_user_id", table_name="analysis_feedback")
    op.drop_index("ix_analysis_feedback_analysis_id", table_name="analysis_feedback")
    op.drop_table("analysis_feedback")
