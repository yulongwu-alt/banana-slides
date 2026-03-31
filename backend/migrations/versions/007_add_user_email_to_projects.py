"""add user_email to projects

Revision ID: 007_add_user_email_to_projects
Revises: 006_add_export_settings
Create Date: 2026-03-31 15:40:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect


# revision identifiers, used by Alembic.
revision = '007_add_user_email_to_projects'
down_revision = '006_add_export_settings'
branch_labels = None
depends_on = None


def _column_exists(table_name: str, column_name: str) -> bool:
    """Check if column exists."""
    bind = op.get_bind()
    inspector = inspect(bind)
    columns = [col['name'] for col in inspector.get_columns(table_name)]
    return column_name in columns


def upgrade() -> None:
    """Add nullable user_email column to projects table."""
    if not _column_exists('projects', 'user_email'):
        op.add_column('projects', sa.Column('user_email', sa.String(length=255), nullable=True))


def downgrade() -> None:
    op.drop_column('projects', 'user_email')
