"""add user_email to user_templates

Revision ID: 008_add_user_email_to_user_templates
Revises: 007_add_user_email_to_projects
Create Date: 2026-04-01 10:30:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect


# revision identifiers, used by Alembic.
revision = '008_add_user_email_to_user_templates'
down_revision = '007_add_user_email_to_projects'
branch_labels = None
depends_on = None


def _column_exists(table_name: str, column_name: str) -> bool:
    """Check if column exists."""
    bind = op.get_bind()
    inspector = inspect(bind)
    columns = [col['name'] for col in inspector.get_columns(table_name)]
    return column_name in columns


def upgrade() -> None:
    """Add nullable user_email column to user_templates table."""
    if not _column_exists('user_templates', 'user_email'):
        op.add_column('user_templates', sa.Column('user_email', sa.String(length=255), nullable=True))


def downgrade() -> None:
    op.drop_column('user_templates', 'user_email')
