"""Portion pricing and real source-message lifecycle; never contacts WhatsApp."""

import hashlib
import os
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from urllib.parse import urlsplit
from uuid import uuid4

import pytest

from app import create_app
from app.application.ingestion import ingest
from app.application.reporting import period_report
from app.domain.meal_items import grouped_items, portion_price
from app.domain.parsing import classify
from app.extensions import db
from app.models import Order
from app.services import local_today
from app.whatsapp_models import WAAudit, WAConnection, WAEvent, WAInbox


@pytest.mark.parametrize(
    ("body", "quantity", "total"),
    [
        ("1 full tiffin", 1, 80),
        ("1 half tiffin", 1, 60),
        ("2 full tiffin\n1 rajma\n1 methi aloo", 2, 160),
        ("1 full tiffin option A\n1 full tiffin option B", 2, 160),
        ("1 full tiffin option A 1 full tiffin option B", 2, 160),
        ("1 full tiffin option A + 1 full tiffin option B", 2, 160),
        ("1 full tiffin + 1 half tiffin", 2, 140),
        ("2 full tiffins and 3 half tiffins", 5, 340),
        ("one full tiffin, two half tiffins", 3, 200),
        ("२ full टिफिन\n१ rajma\n१ methi aloo", 2, 160),
        ("1 full tiffin for lunch; 1 half tiffin for dinner", 2, 140),
        ("1 full tiffin today", 1, 80),
        ("2 full tiffin\n1 full tiffin option A\n1 full tiffin option B", 2, 160),
        ("2 tiffins\n1 full tiffin\n1 half tiffin", 2, 140),
    ],
)
def test_explicit_lines_have_exact_quantity_and_amount(body, quantity, total):
    parsed = classify(body)
    assert parsed.category == "order" and parsed.confidence == "high"
    assert parsed.quantity == quantity
    groups = grouped_items(parsed.items)
    assert (
        sum(
            group["quantity"] * portion_price(key[0], {"TIFFIN_PRICE": "80"})
            for key, group in groups.items()
        )
        == total
    )


@pytest.mark.parametrize(
    "body",
    [
        "1 rajma\n1 methi aloo",
        "2 full tiffin\n2 rajma\n2 methi aloo",
        "1 full tiffin\n1 rajma\n1 methi aloo",
        "2 full tiffin\n3 rajma",
        "if I come, 1 full tiffin",
        "1 half tiffin?",
        "0 half tiffin",
        "21 full tiffin",
        "20 full tiffin + 1 half tiffin",
        "1 full tiffin for Amit and 1 for me",
        "1 full tiffin\n1 mystery dish",
        "1 full tiffin option C",
        "1 full tiffin\npaid 80",
        "1 full tiffin\nnot coming",
        "1 full tiffin please cancel",
        "1 full tiffin option A extra 2",
    ],
)
def test_uncertain_or_conflicting_lines_never_automatically_order(body):
    parsed = classify(body)
    assert not (parsed.category == "order" and parsed.confidence == "high")


def test_legacy_quantity_and_configurable_price_are_preserved():
    parsed = classify("2 tiffins")
    assert parsed.quantity == 2 and not parsed.items
    assert portion_price("standard", {"TIFFIN_PRICE": "99.25"}) == Decimal("99.25")


@pytest.fixture(params=["sqlite", "postgres"])
def meal_app(request):
    uri = "sqlite://"
    if request.param == "postgres":
        uri = os.getenv("TEST_PORTION_POSTGRES_URL")
        if not uri:
            pytest.skip("Optional isolated portion PostgreSQL database not configured")
        if not urlsplit(uri).path.endswith("_test"):
            pytest.fail("Refusing a database not explicitly named *_test")
    app = create_app(
        {
            "TESTING": True,
            "SECRET_KEY": "t" * 48,
            "SQLALCHEMY_DATABASE_URI": uri,
            "SQLALCHEMY_ENGINE_OPTIONS": {},
            "WTF_CSRF_ENABLED": False,
            "SESSION_COOKIE_SECURE": False,
        }
    )
    with app.app_context():
        db.create_all()
        db.session.add(
            WAConnection(
                id=1,
                group_jid="120363000000001@g.us",
                group_name="Fixture",
                status="connected",
                enabled_since=datetime.now(timezone.utc) - timedelta(days=20),
                heartbeat=datetime.now(timezone.utc),
                worker_heartbeat=datetime.now(timezone.utc),
            )
        )
        db.session.commit()
        yield app
        db.session.remove()
        db.drop_all()


def message(
    body, identity="source", kind="original", sender="919000000001@s.whatsapp.net", **extra
):
    inbox = WAInbox(
        event_key=hashlib.sha256(uuid4().bytes).hexdigest(),
        payload={
            "group_jid": "120363000000001@g.us",
            "message_id": identity,
            "kind": kind,
            "sender": sender,
            "timestamp": datetime.now(timezone.utc).timestamp(),
            "body": body,
            "push_name": "Fixture",
            **extra,
        },
    )
    db.session.add(inbox)
    db.session.flush()
    event = ingest(inbox)
    db.session.commit()
    return event


def totals():
    report = period_report(local_today(), local_today())
    assert report["reconciled"] and not report["discrepancies"]
    return report["totals"]


def test_two_options_and_screenshot_dishes_count_two(meal_app):
    for body in (
        "1 full tiffin option A\n1 full tiffin option B",
        "2 full tiffin\n1 rajma\n1 methi aloo",
    ):
        event = message(body, str(uuid4()))
        assert event.state == "applied" and event.quantity == 2
    assert totals() == {"records": 2, "quantity": 4, "amount": Decimal("320.00")}


def test_mixed_portions_store_historical_prices_and_reconcile(meal_app):
    event = message("1 full tiffin + 1 half tiffin")
    assert event.state == "applied"
    assert sorted((o.quantity, o.unit_price) for o in Order.query.all()) == [
        (1, Decimal("60")),
        (1, Decimal("80")),
    ]
    assert totals() == {"records": 2, "quantity": 2, "amount": Decimal("140.00")}
    meal_app.config.update(FULL_TIFFIN_PRICE="100", HALF_TIFFIN_PRICE="75")
    assert totals()["amount"] == 140


def test_replayed_mixed_source_is_not_billed_twice(meal_app):
    source = message("1 full tiffin + 1 half tiffin")
    replay = message("1 full tiffin + 1 half tiffin")
    assert source.state == "applied" and replay.state == "ignored"
    assert totals()["quantity"] == 2 and totals()["amount"] == 140


def test_same_source_edit_reprices_only_changed_portions(meal_app):
    message("1 full tiffin")
    edit = message("1 full tiffin + 1 half tiffin", kind="edit")
    assert edit.state == "applied" and totals()["amount"] == 140
    again = message("2 half tiffin", kind="edit")
    assert again.state == "applied" and totals()["amount"] == 120
    assert totals()["quantity"] == 2
    assert Order.query.filter_by(status="cancelled").count() == 1
    assert WAAudit.query.filter_by(action="created").count() == 2


def test_delete_mixed_source_cancels_every_price_group(meal_app):
    message("1 full tiffin + 1 half tiffin")
    deleted = message("", kind="delete")
    assert deleted.state == "applied"
    assert not Order.query.filter_by(status="active").count()
    assert totals()["quantity"] == 0 and totals()["amount"] == 0


def test_reply_cancels_only_selected_message(meal_app):
    message("1 full tiffin + 1 half tiffin")
    message("1 full tiffin", "other")
    cancel = message("cancel my order", "cancel", target_message_id="source")
    assert cancel.state == "applied" and totals()["amount"] == 80


def test_other_participant_cannot_edit_mixed_source(meal_app):
    message("1 full tiffin + 1 half tiffin")
    attack = message("9 full tiffin", kind="edit", sender="919000000002@s.whatsapp.net")
    assert attack.state == "review" and totals()["amount"] == 140


def test_plain_quantity_change_does_not_guess_mixed_price_allocation(meal_app):
    message("1 full tiffin + 1 half tiffin")
    change = message("make it 3 instead", "change")
    assert change.state == "review" and totals()["amount"] == 140


def test_plain_quantity_change_preserves_half_price(meal_app):
    message("1 half tiffin")
    change = message("make it 3 instead", "change")
    assert change.state == "applied" and totals()["amount"] == 180


def test_two_separate_sources_do_not_become_one_cancel_target(meal_app):
    message("1 full tiffin")
    message("1 half tiffin", "second")
    change = message("cancel my order", "change")
    assert change.state == "review" and totals()["amount"] == 140


def test_old_message_is_not_automatically_repriced(meal_app):
    stamp = datetime(2026, 11, 15, 7, tzinfo=timezone.utc).timestamp()
    event = message("1 half tiffin", timestamp=stamp)
    assert event.state == "review" and Order.query.count() == 0


def pending_legacy_message(body, stamp=None):
    event = message("if I come, 2 tiffins", str(uuid4()))
    event.body = body
    inbox = db.session.get(WAInbox, event.inbox_id)
    inbox.payload = {**inbox.payload, "body": body}
    if stamp is not None:
        event.timestamp = stamp
        inbox.payload = {**inbox.payload, "timestamp": stamp.timestamp()}
    db.session.commit()
    return event


def test_today_recheck_is_dry_run_then_idempotent_apply(meal_app):
    from app.application.reprocess_meals import reprocess_today

    event = pending_legacy_message("1 full tiffin + 1 half tiffin")
    initial_events = WAEvent.query.count()
    preview = reprocess_today()
    assert preview["eligible"] == [event.id] and Order.query.count() == 0
    result = reprocess_today(dry_run=False)
    assert result["applied"] == [event.id]
    assert totals()["quantity"] == 2 and totals()["amount"] == 140
    assert reprocess_today(dry_run=False)["applied"] == []
    assert WAEvent.query.count() == initial_events


def test_today_recheck_leaves_prior_days_and_ambiguity_unchanged(meal_app):
    from app.application.reprocess_meals import reprocess_today

    old = pending_legacy_message("1 half tiffin", datetime.now(timezone.utc) - timedelta(days=1))
    unclear = pending_legacy_message("1 full tiffin\n2 rajma")
    result = reprocess_today(dry_run=False)
    assert not result["applied"] and Order.query.count() == 0
    assert old.state == unclear.state == "review"
    assert old.id not in result["eligible"]


def test_same_portion_edit_keeps_stored_price(meal_app):
    message("1 half tiffin")
    meal_app.config["HALF_TIFFIN_PRICE"] = "75"
    edit = message("2 half tiffin", kind="edit")
    assert edit.state == "applied"
    assert totals()["amount"] == 120


def test_failed_mixed_application_rolls_back_all_groups(meal_app):
    meal_app.config["HALF_TIFFIN_PRICE"] = "0"
    event = message("1 full tiffin + 1 half tiffin")
    assert event.state == "review" and Order.query.count() == 0
    assert WAAudit.query.filter_by(action="created").count() == 0


def test_mixed_excel_export_uses_actual_amounts(meal_app):
    from openpyxl import load_workbook

    from app.application.exports import counts_workbook

    message("1 full tiffin + 1 half tiffin")
    book = load_workbook(counts_workbook(local_today(), "all"))
    for sheet in book:
        assert sheet.cell(2, 4).value == 2
        assert sheet.cell(2, 5).value == 140


def test_duplicate_typed_edit_changes_quantity_once(meal_app):
    message("1 full tiffin")
    first = message("2 full tiffin", kind="edit")
    again = message("2 full tiffin", kind="edit")
    assert first.state == again.state == "applied"
    assert totals()["quantity"] == 2 and totals()["amount"] == 160
    assert Order.query.count() == 1


def test_manual_half_price_and_idempotent_submission(meal_app):
    from app.application.orders import guest_manual
    from app.whatsapp_models import WAParticipant

    participant = WAParticipant(display_name="Manual fixture")
    db.session.add(participant)
    db.session.commit()
    key = str(uuid4())
    guest_manual(participant.id, local_today(), 2, "lunch", key, portion="half")
    guest_manual(participant.id, local_today(), 2, "lunch", key, portion="half")
    assert Order.query.count() == 1 and Order.query.one().amount == 120
    with pytest.raises(ValueError, match="portion price"):
        guest_manual(participant.id, local_today(), 2, "lunch", key, portion="full")


def test_review_can_explicitly_confirm_a_half_portion(meal_app):
    from app.application.order_decisions import apply_event
    from app.models import User

    admin = User(name="Admin", email="admin@test.invalid", role="admin")
    admin.set_password("test-password-long")
    db.session.add(admin)
    db.session.commit()
    event = message("if I come, 1 half tiffin")
    with pytest.raises(ValueError, match="Select full or half"):
        apply_event(event, "order", 1, admin.id)
    apply_event(event, "order", 1, admin.id, portion="half")
    db.session.commit()
    assert event.state == "applied" and Order.query.one().amount == 60
