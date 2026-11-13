"""Prevent duplicate form writes and enforce normalized account identities."""

import sqlalchemy as sa
from alembic import op

revision = "0003_submission"
down_revision = "0002_integrity"
branch_labels = None
depends_on = None


def upgrade():
    connection = op.get_bind()
    conflict = connection.execute(
        sa.text(
            "SELECT lower(trim(email)) FROM users GROUP BY lower(trim(email)) HAVING count(*) > 1"
        )
    ).first()
    if conflict:
        raise RuntimeError("Conflicting normalized emails: reconcile accounts before migration")
    connection.execute(sa.text("UPDATE users SET email = lower(trim(email))"))
    op.create_check_constraint("ck_users_email_normalized", "users", "email = lower(trim(email))")
    op.add_column("orders", sa.Column("submission_key", sa.String(36), nullable=True))
    op.create_unique_constraint("uq_orders_submission_key", "orders", ["submission_key"])


def downgrade():
    op.drop_constraint("uq_orders_submission_key", "orders", type_="unique")
    op.drop_column("orders", "submission_key")
    op.drop_constraint("ck_users_email_normalized", "users", type_="check")
