from datetime import timedelta

from conftest import login

from app.extensions import db
from app.models import Order, User
from app.services import local_today, week_bounds, weekly_totals


def test_login_logout_and_invalid(client):
    assert client.get("/").status_code == 302
    assert (
        client.post("/auth/login", data={"email": "a@example.com", "password": "wrong"}).status_code
        == 200
    )
    assert login(client).status_code == 302
    assert client.get("/").status_code == 200
    assert client.post("/auth/logout").status_code == 302
    assert client.get("/").status_code == 302


def test_login_rate_limit(client):
    for _ in range(8):
        client.post("/auth/login", data={"email": "a@example.com", "password": "wrong"})
    assert (
        client.post("/auth/login", data={"email": "a@example.com", "password": "wrong"}).status_code
        == 429
    )


def test_open_redirect_block(client):
    response = client.post(
        "/auth/login?next=//attacker.test",
        data={"email": "a@example.com", "password": "correct-password-123"},
    )
    assert response.headers["Location"].endswith("/")


def test_member_cannot_be_admin(client, app):
    login(client)
    assert (
        client.post(
            "/admin/members",
            data={
                "name": "bad",
                "email": "x@example.com",
                "temporary_password": "12-characters-yes",
            },
        ).status_code
        == 403
    )
    assert client.get("/admin/export.xlsx").status_code == 403
    assert (
        client.post(
            "/admin/payments/2/paid", data={"week_start": "2026-11-02", "week_end": "2026-11-08"}
        ).status_code
        == 403
    )


def test_member_cannot_assign_other_member(client, app):
    login(client)
    with app.app_context():
        other_id = User.query.filter_by(email="b@example.com").first().id
        day = local_today().isoformat()
    assert (
        client.post(
            "/orders", data={"member_id": other_id, "order_date": day, "quantity": "1"}
        ).status_code
        == 302
    )
    with app.app_context():
        assert Order.query.count() == 1
        assert Order.query.first().member.email == "a@example.com"


def test_member_cannot_cancel_other_order(client, app):
    with app.app_context():
        member = User.query.filter_by(email="b@example.com").first()
        order = Order(
            member_id=member.id,
            order_date=local_today(),
            quantity=2,
            unit_price=80,
            status="active",
        )
        db.session.add(order)
        db.session.commit()
        order_id = order.id
    login(client)
    assert client.post(f"/orders/{order_id}/cancel").status_code == 403
    with app.app_context():
        assert db.session.get(Order, order_id).status == "active"


def test_retired_member_creation_and_historical_export(client, app):
    login(client, "admin@example.com")
    response = client.post(
        "/admin/members",
        data={
            "name": "New",
            "email": "new@example.com",
            "temporary_password": "temporary-pass-123",
        },
    )
    assert response.status_code == 410
    assert client.get("/admin/export.xlsx").status_code == 200
    assert (
        client.post(
            "/admin/members",
            data={
                "name": "Duplicate",
                "email": "new@example.com",
                "temporary_password": "temporary-pass-123",
            },
        ).status_code
        == 410
    )

    with app.app_context():
        assert User.query.filter_by(email="new@example.com").first() is None


def test_order_history_price_and_cancel(client, app):
    login(client)
    with app.app_context():
        today = local_today()
    for _ in range(2):
        assert (
            client.post(
                "/orders", data={"order_date": today.isoformat(), "quantity": "2"}
            ).status_code
            == 302
        )
    with app.app_context():
        start, end = week_bounds(today)
        result = weekly_totals(start, end)
        assert len(result) == 1 and result[0]["orders"] == 4 and result[0]["amount"] == 320
        order_id = Order.query.first().id
    client.post(f"/orders/{order_id}/cancel")
    with app.app_context():
        assert weekly_totals(start, end)[0]["amount"] == 160


def test_password_change_and_forced_password_change(client, app):
    with app.app_context():
        user = User.query.filter_by(email="a@example.com").first()
        user.must_change_password = True
        db.session.commit()
    login(client)
    assert client.get("/").headers["Location"].endswith("/auth/change-password")
    response = client.post(
        "/auth/change-password",
        data={
            "current_password": "correct-password-123",
            "new_password": "a-new-safe-password-123",
            "confirm_password": "a-new-safe-password-123",
        },
    )
    assert response.status_code == 302
    assert client.get("/").status_code == 200


def test_invalid_orders(client, app):
    login(client)
    with app.app_context():
        today = local_today()
    for day, qty in [
        (today.isoformat(), "0"),
        ((today + timedelta(days=1)).isoformat(), "1"),
        (today.isoformat(), "100"),
    ]:
        assert client.post("/orders", data={"order_date": day, "quantity": qty}).status_code == 302
    with app.app_context():
        assert Order.query.count() == 0


def test_deactivated_account_cannot_login(client, app):
    with app.app_context():
        user = User.query.filter_by(email="a@example.com").first()
        user.is_active = False
        db.session.commit()
    assert login(client).status_code == 200
    assert client.get("/").status_code == 302


def test_weekly_bill_includes_only_active(app):
    with app.app_context():
        today = local_today()
        u = User.query.filter_by(email="a@example.com").first()
        db.session.add_all(
            [
                Order(member_id=u.id, order_date=today, quantity=2, unit_price=70, status="active"),
                Order(
                    member_id=u.id, order_date=today, quantity=3, unit_price=80, status="cancelled"
                ),
            ]
        )
        db.session.commit()
        start, end = week_bounds(today)
        assert weekly_totals(start, end, u.id)[0]["amount"] == 140


def test_csrf_enforced(client, app):
    app.config["WTF_CSRF_ENABLED"] = True
    response = client.post(
        "/auth/login", data={"email": "a@example.com", "password": "correct-password-123"}
    )
    assert response.status_code == 400
