"""Run only on a disposable real PostgreSQL database."""

import os
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

import pytest
from conftest import cleanup_member_participants, cleanup_order_audits
from flask_migrate import upgrade

from app import create_app
from app.extensions import db
from app.models import Order, User
from app.services import local_today


@pytest.mark.skipif(not os.getenv("TEST_POSTGRES_URL"), reason="TEST_POSTGRES_URL not set")
def test_concurrent_form_submissions():
    url = os.environ["TEST_POSTGRES_URL"]
    from sqlalchemy.engine import make_url

    assert make_url(url).database.endswith("_test")
    app = create_app(
        {
            "TESTING": True,
            "SECRET_KEY": "t" * 48,
            "SQLALCHEMY_DATABASE_URI": url,
            "WTF_CSRF_ENABLED": False,
            "SESSION_COOKIE_SECURE": False,
        }
    )
    email = f"concurrent-{uuid4()}@test.invalid"
    with app.app_context():
        assert db.engine.dialect.name == "postgresql"
        upgrade()
        user = User(name="Concurrent", email=email, role="member", is_active=True)
        user.set_password("test-password-123")
        db.session.add(user)
        db.session.commit()
        member_id = user.id
        day = local_today().isoformat()
    clients = [app.test_client() for _ in range(8)]
    for client in clients:
        assert (
            client.post(
                "/auth/login", data={"email": email, "password": "test-password-123"}
            ).status_code
            == 302
        )
    form = {"order_date": day, "quantity": 2, "submission_key": str(uuid4())}
    try:
        with ThreadPoolExecutor(max_workers=8) as executor:
            responses = list(
                executor.map(lambda client: client.post("/orders", data=form).status_code, clients)
            )
        assert responses == [302] * 8
        with app.app_context():
            assert Order.query.filter_by(member_id=member_id).count() == 1
    finally:
        with app.app_context():
            cleanup_order_audits([member_id])
            Order.query.filter_by(member_id=member_id).delete()
            cleanup_member_participants([member_id])
            db.session.delete(db.session.get(User, member_id))
            db.session.commit()
