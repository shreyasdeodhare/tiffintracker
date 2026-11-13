import hashlib
from datetime import datetime, timedelta, timezone
from io import BytesIO
from uuid import uuid4

import pytest
from conftest import login
from openpyxl import load_workbook

from app.extensions import db
from app.models import Order, Payment, User
from app.services import local_today, previous_week, week_bounds, weekly_totals
from app.whatsapp_models import WAAudit, WAConnection, WAEvent, WAIdentity, WAInbox, WAParticipant
from app.whatsapp_parser import classify
from app.whatsapp_service import ingest, period_totals, process_next

GROUP = "120363000000001@g.us"
SENDER = "919000000001@s.whatsapp.net"


@pytest.fixture()
def wa(app):
    with app.app_context():
        db.session.add(
            WAConnection(
                id=1,
                group_jid=GROUP,
                group_name="Test group",
                enabled_since=datetime.now(timezone.utc) - timedelta(days=15),
                status="connected",
            )
        )
        db.session.commit()
    return app


def add_message(
    body="2 tiffins", sender=SENDER, message_id=None, kind="original", stamp=None, **extra
):
    message_id = message_id or str(uuid4())
    payload = {
        "group_jid": GROUP,
        "message_id": message_id,
        "sender": sender,
        "timestamp": (stamp or datetime.now(timezone.utc)).timestamp(),
        "body": body,
        "kind": kind,
        "push_name": "Same name",
        **extra,
    }
    item = WAInbox(event_key=hashlib.sha256(str(uuid4()).encode()).hexdigest(), payload=payload)
    db.session.add(item)
    db.session.flush()
    result = ingest(item)
    db.session.commit()
    return result


@pytest.mark.parametrize(
    "text,category,quantity",
    [
        ("1 tiffin for lunch", "order", 1),
        ("two tiffins", "order", 2),
        ("mere liye do tiffin", "order", 2),
        ("मुझे दो टिफिन चाहिए", "order", 2),
        ("२ टिफिन", "order", 2),
        ("3 tifins pls", "order", 3),
        ("Make it 3 instead", "modify", 3),
        ("Cancel my lunch", "cancel", None),
        ("mera tiffin cancel karo", "cancel", None),
        ("Good morning", "conversation", None),
        ("Holiday till Sunday", "announcement", None),
        ("Paid ₹240", "payment", None),
        ("Lunch?", "ambiguous", None),
        ("if I come, 2 tiffins", "ambiguous", None),
        ("Lunch for Amit and me", "ambiguous", None),
        ("0 tiffins", "ambiguous", None),
        ("21 tiffins", "ambiguous", None),
        ("[media: non-text content]", "media", None),
        ("kal 1 tiffin", "ambiguous", None),
    ],
)
def test_conservative_classification(text, category, quantity):
    result = classify(text)
    assert (result.category, result.quantity) == (category, quantity)


def test_guest_participants_without_login_accounts_and_group_counts(wa):
    with wa.app_context():
        first = add_message("2 tiffins")
        second = add_message("1 lunch", sender="919000000002@s.whatsapp.net")
        assert first.state == second.state == "applied"
        assert first.participant_id != second.participant_id
        assert first.participant.member_id is None
        assert User.query.count() == 3  # Guests do not create login users.
        rows = period_totals(local_today(), local_today(), "whatsapp")
        assert sum(row["quantity"] for row in rows) == 3
        assert sum(row["orders"] for row in rows) == 2
        start, end = week_bounds(local_today())
        assert sum(row["amount"] for row in weekly_totals(start, end)) == 240


def test_source_replay_does_not_duplicate(wa):
    with wa.app_context():
        add_message(message_id="fixed")
        duplicate = add_message(message_id="fixed")
        assert duplicate.state == "ignored"
        assert Order.query.count() == 1


def test_modification_cancellation_and_history(wa):
    with wa.app_context():
        first = add_message(message_id="source")
        changed = add_message("Make it 3 instead")
        assert changed.state == "applied" and first.order.quantity == 3
        cancelled = add_message("Cancel my lunch")
        assert cancelled.state == "applied" and first.order.status == "cancelled"
        assert period_totals(local_today(), local_today()) == []
        assert WAAudit.query.filter_by(order_id=first.order_id).count() >= 6


def test_two_orders_same_day_and_explicit_reply_target(wa):
    with wa.app_context():
        first = add_message(message_id="one")
        second = add_message(message_id="two")
        ambiguous = add_message("Make it 3 instead")
        assert ambiguous.state == "review"
        explicit = add_message("Make it 3 instead", target_message_id="one")
        assert explicit.state == "applied"
        assert first.order.quantity == 3 and second.order.quantity == 2


def test_edits_and_deletions(wa):
    with wa.app_context():
        source = add_message(message_id="edited")
        changed = add_message("4 tiffins", message_id="edited", kind="edit")
        assert changed.state == "applied" and source.order.quantity == 4
        deleted = add_message("", message_id="edited", kind="delete")
        assert deleted.state == "applied" and source.order.status == "cancelled"


def test_other_sender_cannot_modify_or_delete(wa):
    with wa.app_context():
        source = add_message(message_id="private")
        attack = add_message(
            "9 tiffins", sender="919000000002@s.whatsapp.net", message_id="private", kind="edit"
        )
        assert attack.state == "review" and source.order.quantity == 2


def test_announcements_questions_payments_and_unrecognized_identity(wa):
    with wa.app_context():
        for body in ["Good morning", "Holiday till Sunday"]:
            assert add_message(body).state == "ignored"
        for body in ["Lunch?", "Paid ₹240", "2 for Amit and 3 for me"]:
            assert add_message(body).state == "review"
        assert add_message("1 tiffin", sender="unknown").state == "review"
        assert Order.query.count() == 0


def test_aliases_are_not_display_name_matching(wa):
    with wa.app_context():
        one = add_message(aliases=["123456789012345@lid"])
        two = add_message(sender="123456789012345@lid", aliases=[SENDER])
        assert one.participant_id == two.participant_id
        assert WAIdentity.query.count() == 2
        assert WAParticipant.query.count() == 1


def test_unselected_group_and_old_history_are_not_billed(wa):
    with wa.app_context():
        assert add_message(group_jid="other@g.us").state == "ignored"
        assert add_message(stamp=datetime.now(timezone.utc) - timedelta(days=30)).state == "ignored"
        assert Order.query.count() == 0


def test_future_order_is_review_only(wa):
    with wa.app_context():
        assert add_message("1 tiffin tomorrow").state == "review"
        assert Order.query.count() == 0


def test_finalized_week_blocks_automatic_changes(wa):
    with wa.app_context():
        start, end = previous_week(local_today())
        stamp = datetime.combine(start, datetime.min.time(), tzinfo=timezone.utc) + timedelta(
            hours=7
        )
        first = add_message(stamp=stamp, message_id="paid")
        member_id = User.query.filter_by(email="a@example.com").one().id
        first.participant.member_id = member_id
        first.order.member_id = member_id
        db.session.add(
            Payment(member_id=member_id, week_start=start, week_end=end, amount=160, status="paid")
        )
        db.session.commit()
        assert add_message(stamp=stamp + timedelta(seconds=1)).state == "review"
        assert (
            add_message(
                "3 tiffins", stamp=stamp + timedelta(seconds=2), message_id="paid", kind="edit"
            ).state
            == "review"
        )
        assert first.order.quantity == 2


def test_durable_inbox_error_and_reprocessing(wa):
    with wa.app_context():
        item = WAInbox(event_key="invalid", payload={"kind": "bad"})
        db.session.add(item)
        db.session.commit()
        assert process_next()
        assert db.session.get(WAInbox, item.id).state == "error"
        assert WAEvent.query.count() == 0


def test_admin_review_export_and_authorization(wa, client):
    with wa.app_context():
        event = add_message("Lunch?")
        event_id = event.id
    login(client)
    assert client.get("/admin/whatsapp").status_code == 403
    assert client.get("/admin/whatsapp/qr.png").status_code == 403
    assert client.get("/admin/whatsapp/export.xlsx").status_code == 403
    assert client.get("/my/counts").status_code == 200
    client.post("/auth/logout")
    login(client, "admin@example.com")
    assert client.get("/admin/whatsapp").status_code == 200
    response = client.post(
        f"/admin/whatsapp/messages/{event_id}/review", data={"action": "order", "quantity": "2"}
    )
    assert response.status_code == 302
    with wa.app_context():
        assert Order.query.count() == 1
    assert (
        client.post(
            f"/admin/whatsapp/messages/{event_id}/review", data={"action": "order", "quantity": "2"}
        ).status_code
        == 409
    )
    book = load_workbook(BytesIO(client.get("/admin/whatsapp/export.xlsx").data))
    assert book["Daily"].cell(2, 4).value == 2


def test_member_mapping_preserves_billing_and_history(wa, client):
    with wa.app_context():
        event = add_message()
        participant_id = event.participant_id
        target = User.query.filter_by(email="a@example.com").one().id
    login(client, "admin@example.com")
    assert (
        client.post(
            f"/admin/whatsapp/participants/{participant_id}/map", data={"member_id": target}
        ).status_code
        == 302
    )
    with wa.app_context():
        assert Order.query.one().member_id == target
        assert WAParticipant.query.one().member_id == target


def test_multi_person_split_is_explicit_and_atomic(wa, client):
    with wa.app_context():
        event = add_message("2 tiffins for Amit and 3 for Sanket")
        event_id = event.id
        assert event.state == "review" and Order.query.count() == 0
        a = User.query.filter_by(email="a@example.com").one().id
        b = User.query.filter_by(email="b@example.com").one().id
    login(client, "admin@example.com")
    response = client.post(
        f"/admin/whatsapp/messages/{event_id}/review",
        data={"action": "split", "split_member": [str(a), str(b)], "split_quantity": ["2", "3"]},
    )
    assert response.status_code == 302
    with wa.app_context():
        assert Order.query.count() == 2
        assert sum(order.quantity for order in Order.query.all()) == 5
        assert db.session.get(WAEvent, event_id).state == "applied"


def test_late_edit_cannot_overwrite_a_later_change(wa):
    with wa.app_context():
        now = datetime.now(timezone.utc)
        source = add_message(stamp=now - timedelta(seconds=3), message_id="late-edit")
        assert (
            add_message("4 tiffins", kind="edit", message_id="late-edit", stamp=now).state
            == "applied"
        )
        late = add_message(
            "3 tiffins", kind="edit", message_id="late-edit", stamp=now - timedelta(seconds=1)
        )
        assert late.state == "review" and source.order.quantity == 4


def test_business_timezone_date_and_historical_price(wa):
    from zoneinfo import ZoneInfo

    with wa.app_context():
        day = local_today() - timedelta(days=1)
        local = datetime.combine(
            day, datetime.min.time(), tzinfo=ZoneInfo("Asia/Kolkata")
        ) + timedelta(minutes=1)
        source = add_message(stamp=local.astimezone(timezone.utc))
        assert source.order_date == day and source.order.order_date == day
        wa.config["TIFFIN_PRICE"] = "99.25"
        newer = add_message()
        assert str(source.order.unit_price) == "80.00"
        assert str(newer.order.unit_price) == "99.25"


def test_ignored_duplicate_cannot_be_approved_or_recreated(wa, client):
    from app.application.order_decisions import (
        apply_event,
        approve_allocations,
        approve_participant_allocations,
    )

    with wa.app_context():
        original = add_message("2 tiffins", message_id="stable-source")
        duplicate = add_message("2 tiffins", message_id="stable-source")
        identity = duplicate.id
        participant_id = original.participant_id
        assert duplicate.state == "ignored"
        assert Order.query.count() == 1
    login(client, "admin@example.com")
    assert (
        client.post(
            f"/admin/whatsapp/messages/{identity}/review", data={"action": "order", "quantity": 2}
        ).status_code
        == 409
    )
    with wa.app_context():
        duplicate = db.session.get(WAEvent, identity)
        duplicate.state = "review"  # Even a stale/manual reset must not replay a source.
        duplicate.participant_id = participant_id
        original_order = Order.query.one()
        original_order.status = "cancelled"
        db.session.commit()
        with pytest.raises(ValueError, match="already created"):
            apply_event(duplicate, action="order", quantity=2, actor_id=1)
        with pytest.raises(ValueError, match="already created"):
            approve_participant_allocations(
                duplicate, [(participant_id, 1), (participant_id + 1, 2)], 1
            )
        with pytest.raises(ValueError, match="already created"):
            approve_allocations(duplicate, [(2, 1), (3, 2)], 1)
        assert Order.query.count() == 1
        assert User.query.count() == 3
        add_message("1 lunch", message_id="distinct-source")
        assert Order.query.count() == 2
