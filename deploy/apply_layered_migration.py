"""Host-only approved migration; never print environment values or private rows."""

import os

from dotenv import load_dotenv
from flask_migrate import upgrade
from sqlalchemy import text
from sqlalchemy.engine import make_url

from app import create_app
from app.extensions import db


def main():
    if os.getenv("TIFFIN_MIGRATION_APPROVED") != "yes":
        raise RuntimeError("Run only through the explicitly approved release procedure")
    load_dotenv("/home/support/tiffintrack/.env", override=True)
    os.environ["DATABASE_URL"] = (
        make_url(os.environ["DATABASE_URL"])
        .set(host="127.0.0.1")
        .render_as_string(hide_password=False)
    )
    with create_app().app_context():
        revision = db.session.execute(text("SELECT version_num FROM alembic_version")).scalar()
        if revision != "0004_whatsapp":
            raise RuntimeError("Expected reviewed baseline revision 0004_whatsapp")
        queries = {
            "users": "SELECT id,name,email,password_hash,role,is_active,must_change_password,billing_email_enabled FROM users ORDER BY id",
            "orders": "SELECT id,member_id,order_date,quantity,unit_price,status,source FROM orders ORDER BY id",
            "payments": "SELECT * FROM payments ORDER BY id",
            "emails": "SELECT * FROM email_logs ORDER BY id",
            "participants": "SELECT id,member_id,display_name,automatically_created FROM wa_participants ORDER BY id",
            "identities": "SELECT * FROM wa_identities ORDER BY jid",
            "events": "SELECT * FROM wa_events ORDER BY id",
            "auth": "SELECT key,encrypted_value FROM wa_auth ORDER BY key",
        }

        def state():
            return {key: db.session.execute(text(sql)).all() for key, sql in queries.items()}

        before = state()
        db.session.rollback()
        upgrade()
        if before != state():
            raise RuntimeError(
                "Historical rows changed; retain backup and investigate before opening traffic"
            )
        if db.session.execute(
            text("SELECT count(*) FROM orders WHERE participant_id IS NULL")
        ).scalar():
            raise RuntimeError("Order participant backfill incomplete")
        print("MIGRATION_COMPLETE_HISTORICAL_ROWS_AND_ENCRYPTED_AUTH_PRESERVED")
        print("preserved_counts", {key: len(value) for key, value in before.items()})


if __name__ == "__main__":
    main()
