from datetime import timedelta
from io import BytesIO
from unittest.mock import patch

from conftest import login
from openpyxl import load_workbook

from app.extensions import db
from app.models import EmailLog, Order, Payment, User, utcnow
from app.reconciliation import reconcile_week
from app.reports import excel_safe
from app.services import local_today, previous_week
from scripts import weekly_billing


def seed():
    member = User.query.filter_by(email="a@example.com").one()
    start, end = previous_week(local_today())
    db.session.add_all(
        [
            Order(
                member_id=member.id,
                order_date=start,
                quantity=2,
                unit_price="80.25",
                status="active",
            ),
            Order(
                member_id=member.id,
                order_date=start,
                quantity=2,
                unit_price="80.25",
                status="active",
            ),
            Order(
                member_id=member.id,
                order_date=start,
                quantity=20,
                unit_price="99.99",
                status="cancelled",
            ),
        ]
    )
    db.session.commit()
    return member.id, start, end


def test_reconciliation_detects_discrepancies_without_changing_orders(app):
    with app.app_context():
        member_id, start, end = seed()
        db.session.add(
            Payment(
                member_id=member_id, week_start=start, week_end=end, amount="1.00", status="paid"
            )
        )
        db.session.commit()
        before = [(o.id, o.status, o.quantity) for o in Order.query.order_by(Order.id)]
        result = reconcile_week(start, end)
        assert result["active_records"] == 2
        assert result["tiffin_quantity"] == 4
        assert result["amount"] == "321.00"
        assert result["mismatches"] == [{"member_id": member_id, "type": "payment_amount"}]
        assert result["potential_duplicate_groups"] == 1
        assert before == [(o.id, o.status, o.quantity) for o in Order.query.order_by(Order.id)]


def test_retry_records_actual_recipient(app):
    with app.app_context():
        member_id, start, end = seed()
        db.session.add(
            EmailLog(
                member_id=member_id,
                week_start=start,
                week_end=end,
                recipient="old@example.com",
                status="failed",
            )
        )
        db.session.commit()
        member = db.session.get(User, member_id)
        member.email = "corrected@example.com"
        db.session.commit()
        with patch("scripts.weekly_billing.send_email") as send:
            assert weekly_billing.run(send=True, retry_failed=True)["sent"] == 1
        assert send.call_args.args[0] == "corrected@example.com"
        assert EmailLog.query.one().recipient == send.call_args.args[0]


def test_billing_rechecks_eligibility_under_lock(app):
    with app.app_context():
        seed()
        original = weekly_billing.lock_member

        def deactivate(member_id):
            member = original(member_id)
            member.billing_email_enabled = False
            db.session.flush()
            return member

        with (
            patch("scripts.weekly_billing.lock_member", side_effect=deactivate),
            patch("scripts.weekly_billing.send_email") as send,
        ):
            assert weekly_billing.run(send=True)["skipped"] == 1
            send.assert_not_called()
        assert EmailLog.query.count() == 0


def test_operations_is_admin_only_and_does_not_expose_member_details(client, app):
    with app.app_context():
        _, start, _ = seed()
    login(client)
    assert client.get("/admin/operations").status_code == 403
    client.post("/auth/logout")
    login(client, "admin@example.com")
    response = client.get(f"/admin/operations?week={start}")
    assert response.status_code == 200
    assert b"321.00" in response.data
    assert b"a@example.com" not in response.data
    assert client.get("/admin/operations?week=invalid").status_code == 400


def test_counts_excel_numeric_format_and_cancelled_exclusion(client, app):
    with app.app_context():
        _, start, _ = seed()
    login(client, "admin@example.com")
    response = client.get(f"/admin/whatsapp/export.xlsx?day={start}&source=manual")
    book = load_workbook(BytesIO(response.data))
    assert book.sheetnames == ["Daily", "Weekly", "Monthly"]
    for sheet in book:
        assert sheet["C2"].value == 2
        assert sheet["D2"].value == 4
        assert sheet["E2"].value == 321
        assert sheet["E2"].number_format == "#,##0.00"


def test_cancellation_rechecks_ownership_after_lock(client, app):
    with app.app_context():
        _, _, _ = seed()
        order_id = Order.query.filter_by(status="active").first().id
        other_id = User.query.filter_by(email="b@example.com").one().id
    login(client)

    def reassign(_):
        order = db.session.get(Order, order_id)
        order.member_id = other_id
        db.session.flush()

    with patch("app.application.orders.lock_member", side_effect=reassign):
        assert client.post(f"/orders/{order_id}/cancel").status_code == 403


def test_unknown_delivery_is_reported_not_retried(app):
    with app.app_context():
        member_id, start, end = seed()
        db.session.add(
            EmailLog(
                member_id=member_id,
                week_start=start,
                week_end=end,
                recipient="a@example.com",
                status="sending",
                created_at=utcnow() - timedelta(hours=2),
            )
        )
        db.session.commit()
        assert reconcile_week(start, end)["email_reconciliation_count"] == 1
        with patch("scripts.weekly_billing.send_email") as send:
            assert weekly_billing.run(send=True, retry_failed=True)["skipped"] == 1
            send.assert_not_called()


def test_export_untrusted_control_characters():
    assert excel_safe("\x0b=1+1") == "'=1+1"
    assert excel_safe("  =1+1") == "'  =1+1"
    assert excel_safe("नाम\x0b") == "नाम"
    assert excel_safe(42) == 42
