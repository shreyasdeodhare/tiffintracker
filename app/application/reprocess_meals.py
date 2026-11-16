"""Re-check today's unresolved messages in source order; never recreate an event."""

from datetime import datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

from flask import current_app

from app.application.ingestion import connection_lock, participant_for
from app.application.order_decisions import apply_event, audit
from app.domain.parsing import classify
from app.extensions import db
from app.infrastructure.inbox import InboxRepository
from app.infrastructure.meal_orders import MealOrderRepository
from app.services import local_today


def reprocess_today(dry_run=True):
    config = connection_lock()
    zone = ZoneInfo(current_app.config["APP_TIMEZONE"])
    start = datetime.combine(local_today(), time.min, zone).astimezone(timezone.utc)
    end = start + timedelta(days=1)
    events = MealOrderRepository().unresolved_between(config.group_jid, start, end)
    result = {
        "dry_run": dry_run,
        "date": str(local_today()),
        "considered": len(events),
        "eligible": [],
        "applied": [],
        "still_review": [],
        "skipped": [],
    }
    for event in events:
        parsed = classify(event.body)
        eligible = event.kind == "delete" or (
            parsed.confidence == "high" and parsed.category in {"order", "modify", "cancel"}
        )
        if not eligible:
            result["skipped"].append(event.id)
            continue
        result["eligible"].append(event.id)
        if dry_run:
            continue
        try:
            with db.session.begin_nested():
                event.category, event.confidence, event.quantity = (
                    parsed.category,
                    parsed.confidence,
                    parsed.quantity,
                )
                event.reason = parsed.reason
                if event.participant is None:
                    inbox = InboxRepository(db.session).inbox(event.inbox_id)
                    event.participant = participant_for(inbox.payload)
                    db.session.flush()
                if event.kind in {"edit", "delete"}:
                    origin = InboxRepository(db.session).original(event)
                    if origin is None:
                        raise ValueError("Original source is unavailable")
                    event.order_date = origin.order_date
                    if event.kind == "edit" and parsed.category != "order":
                        raise ValueError("Edited content is not a clear replacement")
                    if event.kind == "delete":
                        event.category, event.confidence = "cancel", "high"
                apply_event(event)
                audit(event, "portion_rollout_recheck", {"policy_date": str(local_today())})
            result["applied"].append(event.id)
        except ValueError:
            result["still_review"].append(event.id)
    if dry_run:
        db.session.rollback()
    else:
        db.session.commit()
    return result
