from datetime import datetime, timedelta, timezone
from decimal import Decimal
from uuid import uuid4
from zoneinfo import ZoneInfo

import pytest
from conftest import login

from app.application.publications import prepare_weekly, retry_publication
from app.application.reporting import period_report
from app.domain.orders import BusinessError, publication_period, reporting_week
from app.domain.summaries import render_summary
from app.extensions import db
from app.models import Order, User
from app.services import local_today
from app.whatsapp_models import (
    ReconciliationRun,
    SummaryPart,
    SummaryPublication,
    WAAudit,
    WAConnection,
    WAEvent,
    WAInbox,
    WAParticipant,
)


def fixture_orders():
    today = local_today()
    friday = today + timedelta(days=(4 - today.weekday()) % 7)
    start, end = reporting_week(friday)
    db.session.add(
        WAConnection(
            id=1,
            group_jid="120363111111111@g.us",
            group_name="Fixture",
            enabled_since=datetime.combine(
                start - timedelta(days=1), datetime.min.time(), ZoneInfo("Asia/Kolkata")
            ),
            status="connected",
            heartbeat=datetime.now(timezone.utc),
            worker_heartbeat=datetime.now(timezone.utc),
        )
    )
    one = WAParticipant(display_name="Same name", automatically_created=True)
    two = WAParticipant(display_name="Same name", automatically_created=True)
    db.session.add_all([one, two])
    db.session.flush()
    db.session.add_all(
        [
            Order(
                participant_id=one.id,
                order_date=today,
                quantity=2,
                unit_price="80.25",
                status="active",
                source="whatsapp",
            ),
            Order(
                participant_id=one.id,
                order_date=today,
                quantity=1,
                unit_price="90.10",
                status="active",
                source="manual",
            ),
            Order(
                participant_id=two.id,
                order_date=today,
                quantity=3,
                unit_price="70.20",
                status="active",
                source="whatsapp",
            ),
            Order(
                participant_id=two.id,
                order_date=today,
                quantity=20,
                unit_price="90.10",
                status="cancelled",
                source="manual",
            ),
        ]
    )
    db.session.flush()
    for order in Order.query.filter_by(source="whatsapp").all():
        inbox = WAInbox(event_key=str(uuid4()), payload={}, state="processed")
        db.session.add(inbox)
        db.session.flush()
        event = WAEvent(
            inbox_id=inbox.id,
            group_jid="120363111111111@g.us",
            message_id=str(uuid4()),
            kind="original",
            sender="919000000001@s.whatsapp.net",
            participant_id=order.participant_id,
            timestamp=datetime.now(ZoneInfo("Asia/Kolkata")),
            body="Fixture order",
            category="order",
            confidence="high",
            state="applied",
            reason="Golden fixture",
            order_date=order.order_date,
            quantity=order.quantity,
            order_id=order.id,
        )
        db.session.add(event)
        db.session.flush()
        db.session.add(WAAudit(event_id=event.id, order_id=order.id, action="created", detail={}))
    db.session.commit()
    return (
        one.id,
        two.id,
        start,
        end,
        datetime.combine(friday, datetime.min.time(), ZoneInfo("Asia/Kolkata"))
        + timedelta(hours=14),
    )


def test_weekly_only_and_saturday_friday_boundaries():
    from datetime import date

    assert publication_period("weekly", date(2026, 11, 13)) == (
        date(2026, 11, 7),
        date(2026, 11, 13),
    )
    assert reporting_week(date(2026, 11, 7)) == (date(2026, 11, 7), date(2026, 11, 13))
    for kind, day in [("daily", date(2026, 11, 13)), ("weekly", date(2026, 11, 12))]:
        with pytest.raises(BusinessError):
            publication_period(kind, day)


def test_golden_totals_sources_identity_cancelled_and_price(app):
    with app.app_context():
        one, two, start, end, _ = fixture_orders()
        result = period_report(start, end)
        assert result["reconciled"] and not result["provisional"]
        assert result["totals"] == {"records": 3, "quantity": 6, "amount": Decimal("461.20")}
        assert len(result["participants"]) == 2
        assert result["cancelled"] == 1
        assert result["sources"]["manual"]["amount"] == Decimal("90.10")
        assert period_report(start, end, participant_id=one)["totals"]["quantity"] == 3
        assert User.query.count() == 3


def test_publication_snapshot_is_immutable_and_idempotent(app):
    with app.app_context():
        one, _, start, end, now = fixture_orders()
        identity = prepare_weekly(now)
        publication = db.session.get(SummaryPublication, identity)
        assert publication.status == "pending"
        assert publication.snapshot["totals"]["amount"] == "461.20"
        assert SummaryPart.query.one().body.count("Same name") == 2
        db.session.add(
            Order(
                participant_id=one,
                order_date=local_today(),
                quantity=1,
                unit_price=100,
                status="active",
                source="manual",
            )
        )
        db.session.commit()
        assert prepare_weekly(now) == identity
        assert SummaryPublication.query.count() == 1 and ReconciliationRun.query.count() == 1
        assert publication.snapshot["totals"]["quantity"] == 6
        assert period_report(start, end)["totals"]["quantity"] == 7


def test_cutoff_unknown_retry_and_provisional(app):
    with app.app_context():
        _, _, _, _, now = fixture_orders()
        with pytest.raises(BusinessError):
            prepare_weekly(now - timedelta(minutes=1))
        db.session.rollback()
        identity = prepare_weekly(now + timedelta(minutes=3))
        publication = db.session.get(SummaryPublication, identity)
        assert publication.snapshot["provisional"] and publication.snapshot["late_capture"]
        part = SummaryPart.query.one()
        part.status = "unknown"
        publication.status = "unknown"
        db.session.commit()
        with pytest.raises(BusinessError):
            retry_publication(identity, 1, "not_delivered", "")
        retry_publication(
            identity, 1, "not_delivered", "Checked linked device; message was not submitted"
        )
        assert part.status == "pending" and publication.status == "pending"


def test_independent_mismatch_blocks_publication(app, monkeypatch):
    from app.infrastructure.reporting import ReportingRepository

    original = ReportingRepository.totals

    def wrong(self, *args, **kwargs):
        rows = original(self, *args, **kwargs)
        rows[0]["quantity"] += 1
        return rows

    with app.app_context():
        _, _, _, _, now = fixture_orders()
        monkeypatch.setattr(ReportingRepository, "totals", wrong)
        with pytest.raises(BusinessError, match="mismatch"):
            prepare_weekly(now)
        db.session.rollback()
        assert SummaryPublication.query.count() == 0


def test_report_routes_admin_isolation_and_guest_manual(client, app):
    with app.app_context():
        identity, _, _, _, _ = fixture_orders()
        today = local_today()
    login(client)
    assert client.get("/admin/orders").status_code == 403
    assert client.get(f"/admin/participants/{identity}").status_code == 403
    client.post("/auth/logout")
    login(client, "admin@example.com")
    response = client.get("/admin/orders")
    assert response.status_code == 200 and b"461.20" in response.data
    assert client.get(f"/admin/participants/{identity}").status_code == 200
    assert (
        client.post(
            "/admin/participant-orders",
            data={
                "participant_id": identity,
                "date": str(today),
                "quantity": 1,
                "submission_key": str(uuid4()),
            },
        ).status_code
        == 302
    )
    with app.app_context():
        assert User.query.count() == 3
        assert Order.query.filter_by(participant_id=identity, status="active").count() == 3


def test_long_summary_splits_without_losing_participants():
    snapshot = {
        "start": "2026-11-07",
        "end": "2026-11-13",
        "cutoff": "2026-11-13T14:00:00+05:30",
        "provisional": True,
        "participants": [
            {
                "participant_id": i,
                "name": f"Person{i}",
                "records": 1,
                "quantity": 2,
                "amount": "160.00",
            }
            for i in range(1, 181)
        ],
        "totals": {"records": 180, "quantity": 360, "amount": "28800.00"},
        "cancelled": 2,
        "review": 1,
        "unresolved": 0,
        "coverage_incomplete": False,
    }
    parts = render_summary(1, "weekly", snapshot)
    assert len(parts) > 1 and all(len(part) <= 3000 for part in parts)
    joined = "\n".join(parts)
    assert "1 order records — 2 tiffins — INR 160.00" in joined
    assert "Provisional reasons: 1 messages awaiting review" in joined
    for i in range(1, 181):
        assert joined.count(f"Person{i} (#") == 1


def test_guest_manual_audit_and_conflicting_replay(app):
    from app.application.orders import guest_manual
    from app.models import OrderAudit

    with app.app_context():
        identity, _, _, _, _ = fixture_orders()
        key = str(uuid4())
        day = local_today()
        guest_manual(identity, day, 1, "lunch", key, 1)
        audit = OrderAudit.query.one()
        assert audit.actor_id == 1 and audit.action == "created"
        guest_manual(identity, day, 1, "lunch", key, 1)
        assert OrderAudit.query.count() == 1
        with pytest.raises(BusinessError, match="conflicts"):
            guest_manual(identity, day, 2, "lunch", key, 1)
        db.session.rollback()
        assert OrderAudit.query.count() == 1


def test_guest_multi_person_review_without_login_accounts(app):
    from app.application.order_decisions import approve_participant_allocations

    with app.app_context():
        one, two, start, end, _ = fixture_orders()
        item = WAInbox(event_key=str(uuid4()), payload={}, state="processed")
        db.session.add(item)
        db.session.flush()
        event = WAEvent(
            inbox_id=item.id,
            group_jid="120363111111111@g.us",
            message_id=str(uuid4()),
            kind="original",
            sender="919000000001@s.whatsapp.net",
            participant_id=one,
            timestamp=datetime.now(ZoneInfo("Asia/Kolkata")),
            body="Explicit split fixture",
            category="ambiguous",
            confidence="medium",
            state="review",
            reason="Fixture",
            order_date=local_today(),
        )
        db.session.add(event)
        db.session.flush()
        approve_participant_allocations(event, [(one, 2), (two, 3)], 1)
        db.session.commit()
        assert User.query.count() == 3
        assert period_report(start, end)["totals"]["quantity"] == 11
        assert period_report(start, end)["reconciled"]


def test_dashboard_pagination_never_changes_group_totals(app):
    from app.application.reporting import dashboard_data

    with app.app_context():
        fixture_orders()
        for index in range(60):
            p = WAParticipant(display_name=f"Guest {index}", automatically_created=False)
            db.session.add(p)
            db.session.flush()
            db.session.add(
                Order(
                    participant_id=p.id,
                    order_date=local_today(),
                    quantity=1,
                    unit_price=2,
                    status="active",
                    source="manual",
                )
            )
        db.session.commit()
        first = dashboard_data(local_today(), page=1)
        second = dashboard_data(local_today(), page=2)
        assert first["reports"]["daily"]["totals"] == second["reports"]["daily"]["totals"]
        assert first["reports"]["daily"]["totals"]["amount"] == Decimal("581.20")
        assert len(first["reports"]["daily"]["participants"]) == 50
        assert len(second["reports"]["daily"]["participants"]) == 12
        assert first["report_pages"] == 2


def test_stale_tracking_is_provisional_even_when_financial_sums_match(app):
    with app.app_context():
        _, _, start, end, _ = fixture_orders()
        config = db.session.get(WAConnection, 1)
        config.heartbeat = datetime.now(timezone.utc) - timedelta(minutes=2)
        db.session.commit()
        report = period_report(start, end)
        assert report["reconciled"] and report["provisional"] and report["coverage_incomplete"]


def test_dashboard_excel_and_publication_share_record_quantity_amount(app):
    from openpyxl import load_workbook

    from app.application.exports import counts_workbook
    from app.application.reporting import dashboard_data

    with app.app_context():
        _, _, start, end, now = fixture_orders()
        report = period_report(start, end)
        dashboard = dashboard_data(local_today())
        book = load_workbook(counts_workbook(local_today(), "all"), data_only=True)
        for period in ("daily", "weekly", "monthly"):
            rows = list(book[period.title()].values)[1:]
            totals = dashboard["reports"][period]["totals"]
            assert sum(row[2] for row in rows) == totals["records"] == 3
            assert sum(row[3] for row in rows) == totals["quantity"] == 6
            assert (
                sum(Decimal(str(row[4])) for row in rows) == totals["amount"] == Decimal("461.20")
            )
        identity = prepare_weekly(now)
        publication = db.session.get(SummaryPublication, identity)
        assert publication.snapshot["totals"] == {"records": 3, "quantity": 6, "amount": "461.20"}
        assert report["reconciled"]
        body = SummaryPart.query.one().body
        assert "Order records: 3" in body and "Tiffins: 6" in body and "Amount: INR 461.20" in body
        assert "2 order records" in body and "1 order records" in body
