"""Compatibility dashboard for optional member login and historical payments."""

from datetime import timedelta
from decimal import Decimal
from uuid import uuid4

from app.extensions import db
from app.infrastructure.core import CoreRepository
from app.services import local_today, week_bounds, weekly_totals


def data(user, day, page):
    start, end = week_bounds(day)
    repo = CoreRepository(db.session)
    members = repo.members() if user.role == "admin" else []
    totals = weekly_totals(start, end, None if user.role == "admin" else user.id)
    ids = {row["member_id"]: row for row in totals}
    summary = [
        {
            "member": m,
            **ids.get(m.id, {"orders": 0, "amount": Decimal("0"), "payment_status": "pending"}),
        }
        for m in (members if user.role == "admin" else [user])
    ]
    orders = repo.orders_page(start, end, page, None if user.role == "admin" else user.id)
    return dict(
        members=members,
        summary=summary,
        orders=orders.items,
        pagination=orders,
        submission_key=str(uuid4()),
        week_start=start,
        week_end=end,
        today=local_today(),
        previous_week_start=start - timedelta(days=7),
        next_week_start=start + timedelta(days=7),
        current_week=start == week_bounds(local_today())[0],
    )
