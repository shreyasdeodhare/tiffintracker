"""Requires an EMPTY, disposable PostgreSQL database named *_test."""

import os
from urllib.parse import urlsplit

import pytest
from conftest import cleanup_member_participants, cleanup_order_audits
from flask_migrate import upgrade
from sqlalchemy.exc import IntegrityError

from app import create_app
from app.extensions import db
from app.models import Order, User


@pytest.mark.skipif(not os.getenv("TEST_POSTGRES_URL"), reason="TEST_POSTGRES_URL not set")
def test_real_postgres_migrations_and_constraints():
    url = os.environ.get("TEST_POSTGRES_URL", "")
    db_name = urlsplit(url).path.rsplit("/", 1)[-1]
    if not db_name.endswith("_test"):
        pytest.fail("Refusing a database not named *_test")
    app = create_app(
        {
            "TESTING": True,
            "SECRET_KEY": "t" * 48,
            "SQLALCHEMY_DATABASE_URI": url,
            "SQLALCHEMY_ENGINE_OPTIONS": {},
        }
    )
    with app.app_context():
        if db.engine.dialect.name != "postgresql":
            pytest.fail("Real PostgreSQL required")
        upgrade()
        user = User(name="Integration", email="integration-unique@test.invalid", role="member")
        user.set_password("integration-password-123")
        db.session.add(user)
        db.session.commit()
        try:
            db.session.add(
                Order(
                    member_id=user.id,
                    order_date=__import__("datetime").date.today(),
                    quantity=0,
                    unit_price=80,
                    status="active",
                )
            )
            with pytest.raises(IntegrityError):
                db.session.commit()
            db.session.rollback()
        finally:
            cleanup_order_audits([user.id])
            db.session.query(Order).filter_by(member_id=user.id).delete()
            cleanup_member_participants([user.id])
            db.session.delete(user)
            db.session.commit()
