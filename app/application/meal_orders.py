"""Price-grouped source orders, stored using existing rows and JSON audit metadata."""

from dataclasses import asdict
from datetime import date

from flask import current_app

from app.domain.meal_items import grouped_items, portion_price
from app.extensions import db
from app.infrastructure.inbox import InboxRepository
from app.infrastructure.meal_orders import MealOrderRepository
from app.models import Order, utcnow

EFFECTIVE_DAY = date(2026, 11, 16)


def require_current_pricing(day):
    if day < EFFECTIVE_DAY:
        raise ValueError(
            "Full/half and itemized pricing starts on 2026-11-16; earlier dates require historical pricing review"
        )


def create_groups(event, request, participant, actor_id, audit, origin=None):
    require_current_pricing(event.order_date)
    anchor = origin or event
    orders = []
    for (portion, meal), group in grouped_items(request.items).items():
        order = Order(
            member_id=participant.member_id,
            participant_id=participant.id,
            category=meal,
            order_date=event.order_date,
            quantity=group["quantity"],
            unit_price=portion_price(portion, current_app.config),
            source="whatsapp",
            status="active",
        )
        db.session.add(order)
        db.session.flush()
        audit(
            anchor,
            "created",
            {
                "quantity": order.quantity,
                "portion": portion,
                "unit_price": str(order.unit_price),
                "options": group["options"],
                "descriptions": list(request.descriptions),
                "items": [asdict(item) for item in request.items],
                "replacement_event_id": event.id if origin else None,
            },
            actor_id,
            order,
        )
        if origin:
            audit(
                event,
                "modify_added",
                {
                    "portion": portion,
                    "quantity": order.quantity,
                    "unit_price": str(order.unit_price),
                },
                actor_id,
                order,
            )
        orders.append(order)
    event.order_id = orders[0].id
    return orders


def source_identity(order):
    entry = MealOrderRepository().creation(order.id)
    return entry.event_id, entry.detail.get("portion", "standard"), order.category


def replace_groups(event, request, candidates, participant, actor_id, audit):
    """An edit replaces exactly one source message; never another request from that day."""
    require_current_pricing(event.order_date)
    origin = InboxRepository(db.session).original(event)
    if origin is None or origin.participant_id != participant.id:
        raise ValueError("Original source message must belong to this participant")
    if not candidates:
        raise ValueError("No active source order to replace")
    groups = grouped_items(request.items)
    existing = {}
    for order in candidates:
        identity, portion, meal = source_identity(order)
        if identity != origin.id:
            raise ValueError("Replacement spans multiple source messages")
        key = (portion, meal)
        if key in existing:
            raise ValueError("Multiple active rows for one portion/meal require review")
        existing[key] = order
    added = []
    retained = []
    for key, group in groups.items():
        order = existing.pop(key, None)
        if order is None:
            added.extend(item for item in request.items if (item.portion, item.meal) == key)
            continue
        before = {
            "quantity": order.quantity,
            "unit_price": str(order.unit_price),
            "status": order.status,
        }
        order.quantity = group["quantity"]
        # Existing price is historical: a same-portion edit must not reprice it.
        audit(
            event,
            "modify",
            {
                "before": before,
                "after": {
                    "quantity": order.quantity,
                    "unit_price": str(order.unit_price),
                    "status": order.status,
                },
                "options": group["options"],
                "descriptions": list(request.descriptions),
            },
            actor_id,
            order,
        )
        retained.append(order)
    for order in existing.values():
        before = {
            "quantity": order.quantity,
            "unit_price": str(order.unit_price),
            "status": order.status,
        }
        order.status, order.cancelled_at = "cancelled", utcnow()
        audit(
            event,
            "modify_removed",
            {"before": before, "after": {"status": "cancelled"}},
            actor_id,
            order,
        )
    if added:
        from app.domain.meal_items import MealRequest

        retained.extend(
            create_groups(
                event,
                MealRequest(tuple(added), request.descriptions),
                participant,
                actor_id,
                audit,
                origin,
            )
        )
    event.order_id = retained[0].id


def cancel_groups(event, candidates, actor_id, audit):
    if len({source_identity(order)[0] for order in candidates}) != 1:
        raise ValueError("Multiple source messages require an exact cancellation target")
    for order in candidates:
        before = {
            "quantity": order.quantity,
            "unit_price": str(order.unit_price),
            "status": order.status,
        }
        order.status, order.cancelled_at = "cancelled", utcnow()
        audit(
            event,
            "cancel",
            {
                "before": before,
                "after": {
                    "quantity": order.quantity,
                    "unit_price": str(order.unit_price),
                    "status": order.status,
                },
            },
            actor_id,
            order,
        )
    event.order_id = candidates[0].id
