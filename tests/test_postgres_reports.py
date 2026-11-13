"""Real PostgreSQL report isolation while a concurrent order commits."""

import os
from io import BytesIO
from uuid import uuid4

import pytest
from conftest import cleanup_member_participants, cleanup_order_audits
from flask_migrate import upgrade
from openpyxl import load_workbook
from sqlalchemy.engine import make_url

from app import create_app, reports
from app.extensions import db
from app.models import Order, User
from app.services import local_today, lock_member, week_bounds


@pytest.mark.skipif(not os.getenv("TEST_POSTGRES_URL"), reason="TEST_POSTGRES_URL not set")
def test_export_snapshot_during_concurrent_commit(monkeypatch):
    url = os.environ["TEST_POSTGRES_URL"]
    assert make_url(url).database.endswith("_test")
    app = create_app({"TESTING": True, "SECRET_KEY": "t" * 48, "SQLALCHEMY_DATABASE_URI": url})
    with app.app_context():
        upgrade()
        user = User(name="Snapshot", email=f"snapshot-{uuid4()}@test.invalid", role="member")
        user.set_password("test-password-123")
        db.session.add(user)
        db.session.commit()
        member_id = user.id
        # A locking SELECT must refresh an already-loaded identity-map object.
        with db.engine.begin() as connection:
            connection.execute(db.update(User).where(User.id == member_id).values(is_active=False))
        assert user.is_active is True
        assert lock_member(member_id).is_active is False
        db.session.rollback()
        day = local_today()
        db.session.add(Order(member_id=member_id, order_date=day, quantity=2, unit_price="80.25"))
        db.session.commit()
        original = reports.weekly_totals

        def concurrent_insert(*args, **kwargs):
            rows = original(*args, **kwargs)
            # report_snapshot has its own REPEATABLE READ connection. This is
            # another transaction, committed between summary and detail reads.
            db.session.add(
                Order(member_id=member_id, order_date=day, quantity=3, unit_price="80.25")
            )
            db.session.commit()
            return rows

        monkeypatch.setattr(reports, "weekly_totals", concurrent_insert)
        try:
            book = load_workbook(BytesIO(reports.weekly_workbook(*week_bounds(day)).read()))
            summary = [
                r for r in book["Weekly Summary"].iter_rows(values_only=True) if r[0] == "Snapshot"
            ]
            detail = [
                r for r in book["Order Details"].iter_rows(values_only=True) if r[1] == "Snapshot"
            ]
            assert summary[0][2:4] == (2, 160.5)
            assert len(detail) == 1 and detail[0][4] == 2
            assert Order.query.filter_by(member_id=member_id).count() == 2
        finally:
            db.session.rollback()
            cleanup_order_audits([member_id])
            Order.query.filter_by(member_id=member_id).delete()
            cleanup_member_participants([member_id])
            db.session.delete(db.session.get(User, member_id))
            db.session.commit()
