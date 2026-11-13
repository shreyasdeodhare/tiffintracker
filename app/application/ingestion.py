from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from flask import current_app

from app.application.order_decisions import apply_event, audit
from app.application.order_decisions import approve_allocations as approve_allocations
from app.application.order_decisions import target_orders as target_orders
from app.domain.parsing import classify
from app.extensions import db
from app.infrastructure.inbox import InboxRepository
from app.infrastructure.participants import ParticipantRepository
from app.models import utcnow
from app.services import FinalizedPeriod
from app.whatsapp_models import WAConnection, WAEvent, WAInbox, WAParticipant


def connection_lock() -> WAConnection:
    return InboxRepository(db.session).connection_lock()


def participant_for(payload: dict) -> WAParticipant:

    return ParticipantRepository(db.session).resolve(payload)


def ingest(inbox: WAInbox) -> WAEvent:
    config = connection_lock()  # Same lock order for imports, reviews and group selection.
    payload = inbox.payload
    kind = payload.get("kind")
    if kind not in {"original", "edit", "delete"}:
        raise ValueError("Unsupported event kind")
    for key, maximum in [("group_jid", 100), ("message_id", 150), ("sender", 100), ("body", 10000)]:
        if not isinstance(payload.get(key), str) or len(payload[key]) > maximum:
            raise ValueError("Invalid message envelope")
    stamp = datetime.fromtimestamp(float(payload["timestamp"]), timezone.utc)
    interpretation = classify(payload["body"])
    day = stamp.astimezone(ZoneInfo(current_app.config["APP_TIMEZONE"])).date() + timedelta(
        days=interpretation.day_offset
    )
    event = WAEvent(
        inbox_id=inbox.id,
        group_jid=payload["group_jid"],
        message_id=payload["message_id"],
        sender=payload["sender"],
        timestamp=stamp,
        body=payload["body"],
        kind=kind,
        category=interpretation.category,
        confidence=interpretation.confidence,
        quantity=interpretation.quantity,
        order_date=day,
        state="review",
        reason=interpretation.reason,
        target_message_id=payload.get("target_message_id"),
    )
    db.session.add(event)
    db.session.flush()
    since = config.enabled_since
    if since and since.tzinfo is None:
        since = since.replace(tzinfo=timezone.utc)
    if payload["group_jid"] != config.group_jid or not since or stamp < since:
        event.state, event.reason = "ignored", "Outside selected group or before tracking start"
    elif payload.get("truncated"):
        event.state, event.reason = "review", "Oversized message; not automatically counted"
    elif kind == "original" and InboxRepository(db.session).original(event, other=True):
        event.state, event.reason = "ignored", "Repeated source message; no second order"
    elif kind == "original" and interpretation.category in {"conversation", "announcement"}:
        event.state = "ignored"
    else:
        try:
            # Keep valid identities even when a business decision requires review.
            with db.session.begin_nested():
                event.participant = participant_for(payload)
                db.session.flush()
            with db.session.begin_nested():
                if kind in {"edit", "delete"}:
                    original = InboxRepository(db.session).original(event)
                    if original:
                        event.order_date = original.order_date
                    if kind == "edit" and interpretation.category != "order":
                        raise ValueError("Edited content is not a clear replacement quantity")
                    if kind == "delete":
                        event.category, event.confidence = "cancel", "high"
                if event.confidence == "high" and event.category in {"order", "modify", "cancel"}:
                    apply_event(event)
                else:
                    event.reason = interpretation.reason + "; not automatically counted"
        except (ValueError, FinalizedPeriod) as exc:
            event.state, event.reason = "review", str(exc)[:500]
    audit(
        event,
        "decision",
        {"state": event.state, "category": event.category, "reason": event.reason},
    )
    inbox.state, inbox.error = "processed", None
    inbox.processed_at = utcnow()
    return event


def process_next() -> bool:
    # One worker preserves arrival ordering, PostgreSQL locks also defend accidental concurrency.
    config = connection_lock()
    config.worker_heartbeat = utcnow()
    inbox = InboxRepository(db.session).next_pending()
    if inbox is None:
        db.session.commit()
        return False
    inbox_id = inbox.id
    try:
        ingest(inbox)
        db.session.commit()
    except Exception as exc:
        db.session.rollback()
        failed = InboxRepository(db.session).inbox(inbox_id)
        failed.attempts += 1
        failed.state = "error"  # Explicit retry avoids an endless poison-message loop.
        failed.error = type(exc).__name__
        db.session.commit()
        current_app.logger.error(
            "WhatsApp event processing failed inbox_id=%s type=%s", inbox_id, type(exc).__name__
        )
    return True


def period_totals(start, end, source="all", member_id=None, session=None) -> list[dict]:
    from app.infrastructure.reporting import ReportingRepository

    session = session if session is not None else db.session
    participant_id = None
    if member_id is not None:
        participant = InboxRepository(session).member_participant(member_id)
        if participant is None:
            return []
        participant_id = participant.id
    return ReportingRepository(session).totals(start, end, source, participant_id)
