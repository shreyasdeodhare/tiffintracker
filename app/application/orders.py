"""Manual order/payment use cases: rules and transactions outside HTTP handlers."""

from flask import current_app

from app.domain.meal_items import portion_price
from app.domain.orders import BusinessError, validate_category, validate_day, validate_quantity
from app.extensions import db
from app.infrastructure.core import CoreRepository
from app.infrastructure.participants import ParticipantRepository
from app.models import Order, Payment, utcnow
from app.services import completed_week, ensure_mutable, local_today, lock_member, weekly_totals


def create_manual(
    member_id,
    day,
    quantity,
    submission_key=None,
    category="unspecified",
    actor_id=None,
    portion="standard",
):
    validate_day(day, local_today())
    validate_quantity(quantity)
    validate_category(category)
    price = portion_price(portion, current_app.config)
    if portion != "standard":
        from app.application.meal_orders import require_current_pricing

        require_current_pricing(day)
    repo = CoreRepository(db.session)
    member = lock_member(member_id)
    if not member or not member.is_active or member.role != "member":
        raise BusinessError("Choose an active member")
    previous = repo.submission(submission_key) if submission_key else None
    if previous:
        if (previous.member_id, previous.order_date, previous.quantity, previous.category) != (
            member_id,
            day,
            quantity,
            category,
        ):
            raise BusinessError("Submission key was already used for different order details", 409)
        if portion != "standard" and previous.unit_price != price:
            raise BusinessError("Submission key conflicts with another portion price", 409)
        db.session.rollback()
        return False
    participant = ParticipantRepository(db.session).for_member(member_id)
    ParticipantRepository(db.session).lock(participant.id)
    ensure_mutable(member_id, day)
    order = repo.add(
        Order(
            member_id=member_id,
            participant_id=participant.id,
            order_date=day,
            quantity=quantity,
            category=category,
            unit_price=price,
            status="active",
            submission_key=submission_key,
        )
    )
    repo.audit(
        order,
        "created",
        {
            "quantity": quantity,
            "price": str(order.unit_price),
            "date": str(day),
            "portion": portion,
        },
        actor_id,
    )
    db.session.commit()
    return True


def cancel(identity, actor_id, is_admin):
    repo = CoreRepository(db.session)
    order = repo.order(identity)
    if order is None:
        raise BusinessError("Order not found", 404)
    if not is_admin and order.member_id != actor_id:
        raise BusinessError("Forbidden", 403)
    validate_day(order.order_date, local_today())
    former = (order.member_id, order.participant_id)
    if former[0] is not None:
        lock_member(former[0])
    ParticipantRepository(db.session).lock(former[1])
    repo.refresh(order)
    if not is_admin and order.member_id != actor_id:
        raise BusinessError("Forbidden", 403)
    if former != (order.member_id, order.participant_id):
        raise BusinessError("Order ownership changed; reload", 409)
    ensure_mutable(order.member_id, order.order_date)
    if order.status != "active":
        db.session.rollback()
        return False
    order.status = "cancelled"
    order.cancelled_at = utcnow()
    origin = repo.origin(order.id)
    if origin:
        from app.whatsapp_service import audit

        audit(origin, "dashboard_cancel", {"status": "cancelled"}, actor_id, order)
    repo.audit(order, "cancelled", {"before": "active", "after": "cancelled"}, actor_id)
    db.session.commit()
    return True


def record_payment(member_id, start, end):
    member = lock_member(member_id)
    if not member or member.role != "member":
        raise BusinessError("Member not found", 404)
    if not completed_week(start, end):
        raise BusinessError("Only completed Monday–Sunday payment periods are supported")
    rows = weekly_totals(start, end, member_id)
    if not rows:
        raise BusinessError("No payable orders")
    repo = CoreRepository(db.session)
    payment = repo.payment(member_id, start, end)
    if payment is None:
        payment = repo.add(
            Payment(
                member_id=member_id,
                week_start=start,
                week_end=end,
                amount=rows[0]["amount"],
                status="paid",
            )
        )
    else:
        payment.amount, payment.status = rows[0]["amount"], "paid"
    db.session.commit()


def guest_manual(
    participant_id, day, quantity, category, submission_key, actor_id=None, portion="standard"
):
    validate_day(day, local_today())
    validate_quantity(quantity)
    validate_category(category)
    price = portion_price(portion, current_app.config)
    if portion != "standard":
        from app.application.meal_orders import require_current_pricing

        require_current_pricing(day)
    participants = ParticipantRepository(db.session)
    participant = participants.get(participant_id)
    if participant is None:
        raise BusinessError("Participant not found", 404)
    former_member = participant.member_id
    if participant.member_id is not None:
        member = lock_member(participant.member_id)
        if not member or not member.is_active:
            raise BusinessError("Mapped member inactive")
    participant = participants.lock(participant_id)
    if participant.member_id != former_member:
        raise BusinessError("Participant mapping changed; reload before creating an order", 409)
    if not participant.is_active:
        raise BusinessError("Participant inactive")
    previous = CoreRepository(db.session).submission(submission_key)
    if previous:
        if (previous.participant_id, previous.order_date, previous.quantity, previous.category) != (
            participant_id,
            day,
            quantity,
            category,
        ):
            raise BusinessError("Submission key conflicts with another order", 409)
        if portion != "standard" and previous.unit_price != price:
            raise BusinessError("Submission key conflicts with another portion price", 409)
        db.session.rollback()
        return
    ensure_mutable(participant.member_id, day)
    order = CoreRepository(db.session).add(
        Order(
            participant_id=participant.id,
            member_id=participant.member_id,
            order_date=day,
            quantity=quantity,
            category=category,
            unit_price=price,
            status="active",
            source="manual",
            submission_key=submission_key,
        )
    )
    CoreRepository(db.session).audit(
        order,
        "created",
        {
            "quantity": quantity,
            "price": str(order.unit_price),
            "date": str(day),
            "portion": portion,
        },
        actor_id,
    )
    db.session.commit()
