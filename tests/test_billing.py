from unittest.mock import patch

from app.extensions import db
from app.models import EmailLog, Order, User
from app.services import local_today, previous_week
from scripts.weekly_billing import run


def seed(app):
    with app.app_context():
        start, end = previous_week(local_today())
        u = User.query.filter_by(email="a@example.com").first()
        db.session.add(
            Order(member_id=u.id, order_date=start, quantity=2, unit_price=85, status="active")
        )
        db.session.commit()


def test_dry_run_does_not_send(app):
    seed(app)
    with app.app_context(), patch("scripts.weekly_billing.send_email") as send:
        assert run(send=False)["dry_run"] == 1
        assert EmailLog.query.count() == 0
        send.assert_not_called()


def test_success_and_idempotency(app):
    seed(app)
    with app.app_context(), patch("scripts.weekly_billing.send_email") as send:
        assert run(send=True)["sent"] == 1
        assert run(send=True)["skipped"] == 1
        assert EmailLog.query.count() == 1
        send.assert_called_once()


def test_ambiguous_failure_not_resent(app):
    seed(app)
    with (
        app.app_context(),
        patch("scripts.weekly_billing.send_email", side_effect=OSError("timeout")) as send,
    ):
        assert run(send=True)["failed"] == 1
        assert EmailLog.query.first().status == "unknown"
        assert run(send=True)["skipped"] == 1
        send.assert_called_once()
