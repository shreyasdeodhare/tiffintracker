"""Exercise per-recipient PostgreSQL commits across the old 100-row cursor boundary."""

import os
from unittest.mock import patch
from uuid import uuid4

import pytest
from conftest import cleanup_member_participants, cleanup_order_audits
from flask_migrate import upgrade
from sqlalchemy.engine import make_url

from app import create_app
from app.extensions import db
from app.models import EmailLog, Order, User
from app.services import local_today, previous_week
from scripts.weekly_billing import run


@pytest.mark.skipif(not os.getenv("TEST_POSTGRES_URL"), reason="TEST_POSTGRES_URL not set")
def test_postgres_mail_batch_commit_and_failure_isolation():
    url = os.environ["TEST_POSTGRES_URL"]
    assert make_url(url).database.endswith("_test")
    app = create_app({"TESTING": True, "SECRET_KEY": "t" * 48, "SQLALCHEMY_DATABASE_URI": url})
    prefix = str(uuid4())
    with app.app_context():
        assert db.engine.dialect.name == "postgresql"
        upgrade()
        users = [
            User(
                name="Mail Test",
                email=f"{prefix}-{i}@test.invalid",
                password_hash="unused-no-login",
                role="member",
                is_active=True,
            )
            for i in range(101)
        ]
        db.session.add_all(users)
        db.session.flush()
        ids = [user.id for user in users]
        start, _end = previous_week(local_today())
        db.session.add_all(
            [
                Order(
                    member_id=user.id, order_date=start, quantity=1, unit_price=80, status="active"
                )
                for user in users
            ]
        )
        db.session.commit()

        def delivery(recipient, _subject, _body):
            if recipient == f"{prefix}-0@test.invalid":
                raise OSError("ambiguous timeout")

        try:
            with patch("scripts.weekly_billing.send_email", side_effect=delivery) as send:
                result = run(send=True)
                assert result["sent"] == 100 and result["failed"] == 1
                assert send.call_count == 101
                assert run(send=True)["skipped"] == 101
                assert send.call_count == 101
        finally:
            EmailLog.query.filter(EmailLog.member_id.in_(ids)).delete(synchronize_session=False)
            cleanup_order_audits(ids)
            Order.query.filter(Order.member_id.in_(ids)).delete(synchronize_session=False)
            cleanup_member_participants(ids)
            User.query.filter(User.id.in_(ids)).delete(synchronize_session=False)
            db.session.commit()
