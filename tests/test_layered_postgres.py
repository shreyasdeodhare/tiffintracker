"""Forward migration preservation and concurrent weekly claims on disposable PG."""

import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest
from flask_migrate import upgrade
from sqlalchemy import text
from sqlalchemy.engine import make_url

from app import create_app
from app.application.publications import prepare_weekly
from app.extensions import db
from app.services import local_today
from app.whatsapp_models import SummaryPublication, WAConnection


@pytest.mark.skipif(
    not os.getenv("TEST_MIGRATION_URL"), reason="Disposable migration database not configured"
)
def test_forward_migration_preserves_accounts_prices_orders_and_auth():
    url = os.environ["TEST_MIGRATION_URL"]
    assert make_url(url).database.endswith("_migration_test")
    app = create_app({"SECRET_KEY": "m" * 48, "SQLALCHEMY_DATABASE_URI": url})
    with app.app_context():
        upgrade(revision="0004_whatsapp")
        db.session.execute(
            text(
                "INSERT INTO users(id,name,email,password_hash,role,is_active,must_change_password,created_at,billing_email_enabled) VALUES(1,'Mapped','mapped@test.invalid','fixture','member',true,false,now(),true),(2,'Legacy','wa-legacy@members.tiffintrack.invalid','fixture','member',true,true,now(),false)"
            )
        )
        db.session.execute(
            text(
                "INSERT INTO wa_participants(id,member_id,display_name,automatically_created) VALUES(1,1,'Mapped',false)"
            )
        )
        db.session.execute(text("SELECT setval(pg_get_serial_sequence('wa_participants','id'),1)"))
        db.session.execute(
            text(
                "INSERT INTO orders(member_id,order_date,quantity,unit_price,status,created_at,source) VALUES(1,current_date,2,80.25,'active',now(),'manual'),(1,current_date,3,90.10,'cancelled',now(),'manual'),(2,current_date,1,70.00,'active',now(),'manual')"
            )
        )
        db.session.execute(
            text(
                "INSERT INTO wa_auth(key,encrypted_value) VALUES('fixture','preserved-ciphertext')"
            )
        )
        before = db.session.execute(
            text("SELECT id,member_id,quantity,unit_price,status,source FROM orders ORDER BY id")
        ).all()
        db.session.commit()
        upgrade()
        after = db.session.execute(
            text("SELECT id,member_id,quantity,unit_price,status,source FROM orders ORDER BY id")
        ).all()
        assert before == after
        assert (
            db.session.execute(
                text("SELECT count(*) FROM orders WHERE participant_id IS NULL")
            ).scalar()
            == 0
        )
        assert (
            db.session.execute(text("SELECT member_id FROM wa_participants WHERE id=1")).scalar()
            == 1
        )
        assert db.session.execute(text("SELECT count(*) FROM users")).scalar() == 2
        assert (
            db.session.execute(
                text("SELECT encrypted_value FROM wa_auth WHERE key='fixture'")
            ).scalar()
            == "preserved-ciphertext"
        )
        db.session.execute(
            text(
                "INSERT INTO wa_participants(display_name,automatically_created) VALUES('Guest',true)"
            )
        )
        db.session.commit()
        assert db.session.execute(text("SELECT count(*) FROM users")).scalar() == 2


@pytest.mark.skipif(not os.getenv("TEST_POSTGRES_URL"), reason="TEST_POSTGRES_URL not set")
def test_concurrent_preparation_claims_one_snapshot():
    url = os.environ["TEST_POSTGRES_URL"]
    assert make_url(url).database.endswith("_test")
    app = create_app({"SECRET_KEY": "c" * 48, "SQLALCHEMY_DATABASE_URI": url})
    with app.app_context():
        upgrade()
        config = db.session.get(WAConnection, 1)
        config.group_jid = "120363222222222@g.us"
        config.status = "connected"
        config.enabled_since = datetime.now(ZoneInfo("Asia/Kolkata")) - timedelta(days=15)
        db.session.commit()
        day = local_today()
        friday = day + timedelta(days=(4 - day.weekday()) % 7)
        now = datetime.combine(friday, datetime.min.time(), ZoneInfo("Asia/Kolkata")) + timedelta(
            hours=14
        )

    def prepare(_):
        with app.app_context():
            return prepare_weekly(now)

    with ThreadPoolExecutor(max_workers=4) as executor:
        identities = list(executor.map(prepare, range(4)))
    assert len(set(identities)) == 1
    with app.app_context():
        assert SummaryPublication.query.filter_by(group_jid="120363222222222@g.us").count() == 1
        from sqlalchemy.exc import IntegrityError

        with pytest.raises(IntegrityError):
            with db.session.begin_nested():
                db.session.execute(
                    text("UPDATE summary_publications SET snapshot_hash='tampered' WHERE id=:id"),
                    {"id": identities[0]},
                )
        # Publication is deliberately retained in this disposable DB for the
        # separate Node PostgreSQL outbox/concurrency test. No transport runs.


@pytest.mark.skipif(not os.getenv("TEST_POSTGRES_URL"), reason="TEST_POSTGRES_URL not set")
def test_connection_lock_refreshes_cached_configuration():
    from app.infrastructure.inbox import InboxRepository

    url = os.environ["TEST_POSTGRES_URL"]
    assert make_url(url).database.endswith("_test")
    app = create_app({"SECRET_KEY": "g" * 48, "SQLALCHEMY_DATABASE_URI": url})
    with app.app_context():
        upgrade()
        config = db.session.get(WAConnection, 1)
        previous = config.group_name
        try:
            with db.engine.begin() as connection:
                connection.execute(
                    text("UPDATE wa_connection SET group_name='Fresh fixture metadata' WHERE id=1")
                )
            assert config.group_name == previous
            assert (
                InboxRepository(db.session).connection_lock().group_name == "Fresh fixture metadata"
            )
        finally:
            db.session.rollback()
            with db.engine.begin() as connection:
                connection.execute(
                    text("UPDATE wa_connection SET group_name=:name WHERE id=1"), {"name": previous}
                )
