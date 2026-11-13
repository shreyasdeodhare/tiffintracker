from datetime import timedelta
from decimal import Decimal

from flask import current_app

from app.extensions import db
from app.infrastructure.inbox import InboxRepository
from app.infrastructure.participants import ParticipantRepository
from app.models import Order, utcnow
from app.services import ensure_mutable, local_today, lock_member
from app.whatsapp_models import WAAudit, WAEvent


def audit(event: WAEvent, action: str, detail: dict, actor_id=None, order=None) -> None:
    db.session.add(
        WAAudit(
            event_id=event.id,
            order_id=order.id if order else event.order_id,
            actor_id=actor_id,
            action=action,
            detail=detail,
        )
    )


def validate_order_day(day) -> None:
    today = local_today()
    if not today - timedelta(days=14) <= day <= today:
        raise ValueError("Order date must be within the same 14-day window as manual orders")


def ensure_new_source(event):
    if InboxRepository(db.session).source_has_created_orders(event):
        raise ValueError("This source message already created an order")


def approve_allocations(event: WAEvent, allocations: list[tuple[int, int]], actor_id: int) -> None:
    """Resolve an explicit multi-person request atomically, without name guessing."""
    if event.state != "review" or event.kind != "original":
        raise ValueError("Only unresolved original messages can be split")
    if not 2 <= len(allocations) <= 10 or len({member for member, _ in allocations}) != len(
        allocations
    ):
        raise ValueError("Choose 2–10 different members for a split")
    ensure_new_source(event)
    validate_order_day(event.order_date)
    for member_id, quantity in sorted(allocations):
        member = lock_member(member_id)
        if not member or member.role != "member" or not member.is_active or not 1 <= quantity <= 20:
            raise ValueError("Each allocation needs an active member and quantity 1–20")
        ensure_mutable(member_id, event.order_date)
    for member_id, quantity in allocations:
        order = Order(
            member_id=member_id,
            order_date=event.order_date,
            quantity=quantity,
            unit_price=Decimal(str(current_app.config["TIFFIN_PRICE"])),
            source="whatsapp",
            status="active",
        )
        db.session.add(order)
        db.session.flush()
        if event.order_id is None:
            event.order_id = order.id
        audit(
            event,
            "created",
            {"member_id": member_id, "quantity": quantity, "split": True},
            actor_id,
            order,
        )
    event.state, event.reason = "applied", "Administrator confirmed multi-person allocation"


def target_orders(event: WAEvent) -> list[Order]:
    return InboxRepository(db.session).target_orders(event)


def apply_event(
    event: WAEvent,
    action=None,
    quantity=None,
    actor_id=None,
    member_id=None,
    target_id=None,
    participant_id=None,
) -> None:
    if event.state != "review":
        raise ValueError("This message decision is already final")
    participant = event.participant
    if participant is None:
        raise ValueError("Participant identity must be resolved first")

    participants = ParticipantRepository(db.session)
    if participant_id is not None:
        if actor_id is None:
            raise ValueError("Only an administrator can override participant selection")
        participant = participants.get(participant_id)
        if participant is None:
            raise ValueError("Participant not found")
    if member_id is not None and member_id != participant.member_id:
        participant = participants.for_member(member_id)
    chosen_member = participant.member_id
    if chosen_member is not None:
        member = lock_member(chosen_member)
        if not member or member.role != "member" or not member.is_active:
            raise ValueError("Only active mapped members can receive orders")
    participant = participants.lock(participant.id)
    if not participant or not participant.is_active:
        raise ValueError("Only active participants can receive orders")
    action = action or (
        "modify" if event.kind == "edit" else "cancel" if event.kind == "delete" else event.category
    )
    quantity = event.quantity if quantity is None else quantity
    validate_order_day(event.order_date)
    ensure_mutable(chosen_member, event.order_date)
    if action == "order":
        ensure_new_source(event)
        if event.kind != "original":
            raise ValueError("Edits cannot create a second order")
        if quantity is None or not 1 <= quantity <= 20:
            raise ValueError("Quantity must be 1–20")
        order = Order(
            member_id=chosen_member,
            participant_id=participant.id,
            category=(
                "dinner"
                if "dinner" in event.body.casefold()
                else "lunch"
                if "lunch" in event.body.casefold()
                else "unspecified"
            ),
            order_date=event.order_date,
            quantity=quantity,
            unit_price=Decimal(str(current_app.config["TIFFIN_PRICE"])),
            source="whatsapp",
            status="active",
        )
        db.session.add(order)
        db.session.flush()
        event.order_id = order.id
        audit(
            event,
            "created",
            {"quantity": quantity, "member_id": chosen_member, "date": str(event.order_date)},
            actor_id,
            order,
        )
    elif action in {"modify", "cancel"}:
        candidates = target_orders(event)
        if target_id is not None:
            candidates = [order for order in candidates if order.id == target_id]
        if len(candidates) != 1:
            raise ValueError("A unique active source order is required; select an exact target")
        order = candidates[0]
        if order.participant_id != participant.id:
            raise ValueError("Source order belongs to a different member")
        validate_order_day(order.order_date)
        ensure_mutable(order.member_id, order.order_date)
        # A late revision must not overwrite a newer applied revision.
        later = InboxRepository(db.session).later_change(event, order.id)
        if later and actor_id is None:
            raise ValueError("Out-of-order change requires review")
        before = {"quantity": order.quantity, "status": order.status}
        if action == "modify":
            if quantity is None or not 1 <= quantity <= 20:
                raise ValueError("Quantity must be 1–20")
            order.quantity = quantity
        else:
            order.status = "cancelled"
            order.cancelled_at = utcnow()
        event.order_id = order.id
        audit(
            event,
            action,
            {"before": before, "after": {"quantity": order.quantity, "status": order.status}},
            actor_id,
            order,
        )
    else:
        raise ValueError("Choose order, modify or cancel explicitly")
    event.state = "applied"
    event.reason = (
        "Administrator confirmed"
        if actor_id
        else "Automatically applied from stable sender identity"
    )


def approve_participant_allocations(event, allocations, actor_id):
    if event.state != "review" or event.kind != "original":
        raise ValueError("Only unresolved original messages can be split")
    if not 2 <= len(allocations) <= 10 or len({identity for identity, _ in allocations}) != len(
        allocations
    ):
        raise ValueError("Choose 2–10 different participants")
    ensure_new_source(event)
    validate_order_day(event.order_date)
    repo = ParticipantRepository(db.session)
    participants = {identity: repo.get(identity) for identity, _ in allocations}
    if any(p is None for p in participants.values()):
        raise ValueError("Participant not found")
    for member_id in sorted(
        {p.member_id for p in participants.values() if p.member_id is not None}
    ):
        member = lock_member(member_id)
        if not member or not member.is_active:
            raise ValueError("Mapped member inactive")
    for identity in sorted(participants):
        participants[identity] = repo.lock(identity)
        if not participants[identity].is_active:
            raise ValueError("Participant inactive")
        ensure_mutable(participants[identity].member_id, event.order_date)
    from app.domain.orders import validate_quantity

    for _, quantity in allocations:
        validate_quantity(quantity)
    for identity, quantity in allocations:
        participant = participants[identity]
        order = Order(
            member_id=participant.member_id,
            participant_id=identity,
            order_date=event.order_date,
            quantity=quantity,
            unit_price=Decimal(str(current_app.config["TIFFIN_PRICE"])),
            source="whatsapp",
            status="active",
        )
        db.session.add(order)
        db.session.flush()
        if event.order_id is None:
            event.order_id = order.id
        audit(
            event,
            "created",
            {"participant_id": identity, "quantity": quantity, "split": True},
            actor_id,
            order,
        )
    event.state, event.reason = "applied", "Administrator confirmed participant allocations"
