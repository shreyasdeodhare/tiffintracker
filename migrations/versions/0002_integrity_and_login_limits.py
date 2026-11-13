import sqlalchemy as sa
from alembic import op

revision = "0002_integrity"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "login_attempts",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("key", sa.String(300), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_login_attempts_key", "login_attempts", ["key"])
    op.create_index("ix_login_attempts_created_at", "login_attempts", ["created_at"])
    op.create_index("ix_orders_member_date", "orders", ["member_id", "order_date"])
    op.create_index("ix_payments_week", "payments", ["week_start", "week_end"])
    op.create_index(
        "ix_email_logs_week_member", "email_logs", ["week_start", "week_end", "member_id"]
    )
    op.create_check_constraint("ck_users_role", "users", "role IN ('admin', 'member')")
    op.create_check_constraint("ck_orders_quantity", "orders", "quantity BETWEEN 1 AND 20")
    op.create_check_constraint("ck_orders_price", "orders", "unit_price > 0")
    op.create_check_constraint("ck_orders_status", "orders", "status IN ('active', 'cancelled')")
    op.create_check_constraint("ck_payments_amount", "payments", "amount >= 0")
    op.create_check_constraint("ck_payments_status", "payments", "status IN ('paid', 'pending')")
    op.create_check_constraint(
        "ck_payments_week",
        "payments",
        "week_end = week_start + 6 AND EXTRACT(ISODOW FROM week_start) = 1",
    )
    op.create_check_constraint(
        "ck_email_logs_status", "email_logs", "status IN ('sending', 'sent', 'failed', 'unknown')"
    )
    op.create_index(
        "uq_email_logs_sent_once",
        "email_logs",
        ["member_id", "week_start", "week_end"],
        unique=True,
        postgresql_where=sa.text("status IN ('sending', 'sent', 'unknown')"),
    )


def downgrade():
    op.drop_index("uq_email_logs_sent_once", table_name="email_logs")
    for name, table in [
        ("ck_email_logs_status", "email_logs"),
        ("ck_payments_week", "payments"),
        ("ck_payments_status", "payments"),
        ("ck_payments_amount", "payments"),
        ("ck_orders_status", "orders"),
        ("ck_orders_price", "orders"),
        ("ck_orders_quantity", "orders"),
        ("ck_users_role", "users"),
    ]:
        op.drop_constraint(name, table, type_="check")
    op.drop_index("ix_email_logs_week_member", table_name="email_logs")
    op.drop_index("ix_payments_week", table_name="payments")
    op.drop_index("ix_orders_member_date", table_name="orders")
    op.drop_index("ix_login_attempts_created_at", table_name="login_attempts")
    op.drop_index("ix_login_attempts_key", table_name="login_attempts")
    op.drop_table("login_attempts")
