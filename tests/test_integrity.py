from io import BytesIO
from unittest.mock import patch
from uuid import uuid4

from conftest import login
from openpyxl import load_workbook

from app.extensions import db
from app.models import EmailLog, Order, Payment, User
from app.services import local_today, previous_week, week_bounds
from scripts.weekly_billing import run


def seed_previous():
    member = User.query.filter_by(email="a@example.com").one()
    start, end = previous_week(local_today())
    order = Order(
        member_id=member.id, order_date=start, quantity=2, unit_price="80.25", status="active"
    )
    db.session.add(order)
    db.session.commit()
    return member.id, order.id, start, end


def test_paid_period_cannot_gain_or_lose_orders(client, app):
    with app.app_context():
        member_id, order_id, start, end = seed_previous()
    login(client, "admin@example.com")
    for _ in range(2):
        assert (
            client.post(
                f"/admin/payments/{member_id}/paid", data={"week_start": start, "week_end": end}
            ).status_code
            == 302
        )
    with app.app_context():
        assert Payment.query.count() == 1
        assert str(Payment.query.one().amount) == "160.50"
    assert (
        client.post(
            "/orders", data={"member_id": member_id, "order_date": start, "quantity": 1}
        ).status_code
        == 409
    )
    assert client.post(f"/orders/{order_id}/cancel").status_code == 409


def test_issued_invoice_freezes_period(client, app):
    with app.app_context():
        member_id, order_id, start, end = seed_previous()
    login(client)
    with app.app_context(), patch("scripts.weekly_billing.send_email"):
        assert run(send=True)["sent"] == 1
    assert client.post("/orders", data={"order_date": start, "quantity": 1}).status_code == 409
    assert client.post(f"/orders/{order_id}/cancel").status_code == 409


def test_submission_idempotency(client, app):
    login(client)
    with app.app_context():
        day = local_today()
    form = {"order_date": day, "quantity": 2, "submission_key": str(uuid4())}
    client.post("/orders", data=form)
    client.post("/orders", data=form)
    with app.app_context():
        assert Order.query.count() == 1


def test_reset_password_revokes_session(client, app):
    login(client)
    with app.app_context():
        member = User.query.filter_by(email="a@example.com").one()
        member.set_password("new-admin-reset-password")
        db.session.commit()
    assert client.get("/").status_code == 302


def test_disabled_existing_session(client, app):
    login(client)
    with app.app_context():
        User.query.filter_by(email="a@example.com").one().is_active = False
        db.session.commit()
    assert client.get("/").status_code == 302


def test_export_contents_and_formula_safety(client, app):
    with app.app_context():
        member = User.query.filter_by(email="a@example.com").one()
        member.name = "=1+1"
        db.session.add(
            Order(
                member_id=member.id,
                order_date=local_today(),
                quantity=2,
                unit_price="80.25",
                status="active",
            )
        )
        db.session.commit()
    login(client, "admin@example.com")
    workbook = load_workbook(BytesIO(client.get("/admin/export.xlsx").data))
    sheet = workbook["Weekly Summary"]
    assert sheet["A2"].value == "'=1+1"
    assert sheet["D2"].value == 160.5
    assert workbook["Order Details"].max_row == 2


def test_history_pagination(client, app):
    with app.app_context():
        member = User.query.filter_by(email="a@example.com").one()
        db.session.add_all(
            [
                Order(
                    member_id=member.id,
                    order_date=local_today(),
                    quantity=1,
                    unit_price=80,
                    status="active",
                )
                for _ in range(51)
            ]
        )
        db.session.commit()
    login(client)
    assert b"Next page" in client.get("/").data
    assert b"Previous page" in client.get("/?page=2").data
    assert client.get("/?page=0").status_code == 400


def test_failed_configuration_retry_is_safe(app):
    with app.app_context():
        seed_previous()
        with patch("scripts.weekly_billing.send_email", side_effect=ValueError("config")):
            assert run(send=True)["failed"] == 1
        assert EmailLog.query.one().status == "failed"
        with patch("scripts.weekly_billing.send_email") as send:
            assert run(send=True, retry_failed=True)["sent"] == 1
            send.assert_called_once()


def test_week_boundaries():
    from datetime import date

    assert week_bounds(date(2026, 11, 8)) == (date(2026, 11, 2), date(2026, 11, 8))
    assert previous_week(date(2026, 11, 9)) == (date(2026, 11, 2), date(2026, 11, 8))
