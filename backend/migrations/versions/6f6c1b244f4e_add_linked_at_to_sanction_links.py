"""Add linked_at to sanction_links

Revision ID: 6f6c1b244f4e
Revises: 3c2d843c3956
Create Date: 2025-11-14 21:10:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '6f6c1b244f4e'
down_revision: Union[str, Sequence[str], None] = '3c2d843c3956'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        'sanction_links',
        sa.Column(
            'linked_at',
            sa.DateTime(timezone=True),
            server_default=sa.text('CURRENT_TIMESTAMP'),
            nullable=False,
        ),
    )


def downgrade() -> None:
    op.drop_column('sanction_links', 'linked_at')
