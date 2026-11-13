"""Group selection, review and identity-mapping transactions."""

import json
from datetime import datetime, timedelta, timezone
from urllib.error import URLError

from app.domain.orders import BusinessError
from app.extensions import db
from app.infrastructure.bridge_client import bridge
from app.infrastructure.participants import ParticipantRepository
from app.infrastructure.whatsapp_admin import WhatsAppAdminRepository
from app.models import utcnow
from app.services import ensure_mutable, lock_member
from app.whatsapp_service import (
    apply_event,
    approve_allocations,
    audit,
    connection_lock,
    participant_for,
)


def status_data():
    try:
        result = json.loads(bridge("/status")[0])
    except (OSError, URLError, ValueError):
        result = {"connection": "unavailable", "groups": [], "qr": False}
    repo = WhatsAppAdminRepository(db.session)
    config = repo.config()
    heartbeat = config.worker_heartbeat if config else None
    if heartbeat and heartbeat.tzinfo is None:
        heartbeat = heartbeat.replace(tzinfo=timezone.utc)
    result["worker"] = (
        "running"
        if heartbeat and datetime.now(timezone.utc) - heartbeat < timedelta(seconds=30)
        else "not_running"
    )
    states = repo.inbox_states()
    result["pending"] = states.get("pending", 0)
    result["errors"] = states.get("error", 0)
    return result


def dashboard_data(state, page):
    return WhatsAppAdminRepository(db.session).dashboard(state, page)


def detail_data(identity):
    repo = WhatsAppAdminRepository(db.session)
    event = repo.event(identity)
    if event is None:
        raise BusinessError("Event not found", 404)
    return repo.detail(event)


def select_group(identity, actor_id):
    chosen = next(
        (item for item in status_data().get("groups", []) if item["id"] == identity), None
    )
    if chosen is None:
        raise BusinessError("Connect and select an available group")
    config = connection_lock()
    if config.group_jid and config.group_jid != identity:
        raise BusinessError("Changing the tracked group requires reviewed migration", 409)
    if not config.group_jid:
        config.group_jid, config.group_name = identity, chosen["name"][:200]
        config.enabled_since, config.selected_by = utcnow(), actor_id
    db.session.commit()
    try:
        bridge("/refresh", "POST")
    except (OSError, ValueError):
        pass


def review(
    identity,
    action,
    quantity,
    member_id,
    target_id,
    day,
    allocations,
    actor_id,
    participant_id=None,
    participant_allocations=None,
):
    connection_lock()
    repo = WhatsAppAdminRepository(db.session)
    event = repo.event(identity)
    if event is None:
        raise BusinessError("Event not found", 404)
    if event.state != "review":
        raise BusinessError("Decision is already final", 409)
    if action == "reject":
        event.state, event.reason = "rejected", "Administrator rejected"
        audit(event, "rejected", {}, actor_id)
    else:
        if event.participant is None:
            event.participant = (
                repo.participant(participant_id)
                if participant_id
                else participant_for(repo.inbox(event.inbox_id).payload)
            )
            if event.participant is None:
                raise BusinessError("Choose a verified participant")
            db.session.flush()
        if day is not None:
            event.order_date = day
        if action == "split" and participant_allocations:
            from app.application.order_decisions import approve_participant_allocations

            approve_participant_allocations(event, participant_allocations, actor_id)
        elif action == "split":
            approve_allocations(event, allocations, actor_id)
        else:
            apply_event(event, action, quantity, actor_id, member_id, target_id, participant_id)
    db.session.commit()


def retry(identity):
    connection_lock()
    inbox = WhatsAppAdminRepository(db.session).inbox(identity)
    if inbox is None:
        raise BusinessError("Inbox not found", 404)
    if inbox.state != "error":
        raise BusinessError("Only failed processing can be retried", 409)
    inbox.state, inbox.error = "pending", None
    db.session.commit()


def map_member(participant_id, member_id, actor_id):
    connection_lock()
    repo = WhatsAppAdminRepository(db.session)
    participant = repo.participant(participant_id)
    if participant is None:
        raise BusinessError("Participant not found", 404)
    for identity in sorted(
        {value for value in (member_id, participant.member_id) if value is not None}
    ):
        lock_member(identity)
    participant = ParticipantRepository(db.session).lock(participant_id)
    member = repo.user(member_id)
    if not member or member.role != "member" or not member.is_active:
        raise BusinessError("Choose an active member")
    if repo.mapping_conflict(participant_id, member_id):
        raise BusinessError("Conflicting canonical identity; no automatic merge")
    orders = repo.participant_orders(participant_id)
    previous = participant.member_id
    for order in orders:
        ensure_mutable(order.member_id, order.order_date)
        ensure_mutable(member_id, order.order_date)
        order.member_id = member_id
        event = repo.order_event(order.id)
        if event:
            audit(
                event,
                "member_mapping",
                {"previous": previous, "member_id": member_id, "participant_id": participant_id},
                actor_id,
                order,
            )
    from app.whatsapp_models import ParticipantAudit

    db.session.add(
        ParticipantAudit(
            participant_id=participant_id,
            actor_id=actor_id,
            action="member_mapping",
            detail={"previous": previous, "member_id": member_id},
        )
    )
    participant.member_id, participant.automatically_created = member_id, False
    old = repo.user(previous)
    if old and previous != member_id and old.email.endswith("@members.tiffintrack.invalid"):
        old.is_active = False
    db.session.commit()
