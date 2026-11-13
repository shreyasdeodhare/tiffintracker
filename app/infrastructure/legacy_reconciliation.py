"""Read-only financial checks; suspicious repeats are never auto-deleted."""

from collections import defaultdict
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from sqlalchemy import func

from app.models import EmailLog, Order, Payment

from .finance import report_snapshot, weekly_totals


def reconcile_week(start, end):
    with report_snapshot() as session:
        return _reconcile_week(session, start, end)


def _reconcile_week(session, start, end):
    independent = defaultdict(lambda: {"quantity": 0, "amount": Decimal("0.00"), "records": 0})
    orders = (
        session.query(Order.member_id, Order.quantity, Order.unit_price)
        .filter(Order.order_date.between(start, end), Order.status == "active")
        .yield_per(200)
    )
    for member_id, quantity, price in orders:
        row = independent[member_id]
        row["records"] += 1
        row["quantity"] += quantity
        row["amount"] += Decimal(price) * quantity
    aggregate = {row["member_id"]: row for row in weekly_totals(start, end, session=session)}
    mismatches = []
    for member_id in sorted(set(independent) | set(aggregate)):
        actual = independent[member_id]
        reported = aggregate.get(member_id)
        if not reported or (actual["quantity"], actual["amount"]) != (
            reported["orders"],
            reported["amount"],
        ):
            mismatches.append({"member_id": member_id, "type": "aggregation"})
    for payment in session.query(Payment).filter_by(week_start=start, week_end=end):
        if payment.status == "paid" and payment.amount != independent[payment.member_id]["amount"]:
            mismatches.append({"member_id": payment.member_id, "type": "payment_amount"})
    # Same member/day/quantity/price can also represent intentional second orders.
    # Limit displayed candidates while returning the total number of groups.
    repeated = (
        session.query(Order.member_id, Order.order_date, Order.quantity, func.count(Order.id))
        .filter(Order.order_date.between(start, end), Order.status == "active")
        .group_by(Order.member_id, Order.order_date, Order.quantity, Order.unit_price)
        .having(func.count(Order.id) > 1)
    )
    candidate_count = repeated.count()
    candidates = [
        {"member_id": mid, "date": day.isoformat(), "quantity": qty, "records": count}
        for mid, day, qty, count in repeated.order_by(Order.member_id, Order.order_date).limit(100)
    ]
    cutoff = datetime.now(timezone.utc) - timedelta(hours=1)
    uncertain = session.query(EmailLog).filter(
        EmailLog.week_start == start,
        EmailLog.week_end == end,
        (EmailLog.status == "unknown")
        | ((EmailLog.status == "sending") & (EmailLog.created_at < cutoff)),
    )
    return {
        "week_start": start.isoformat(),
        "week_end": end.isoformat(),
        "active_records": sum(row["records"] for row in independent.values()),
        "tiffin_quantity": sum(row["quantity"] for row in independent.values()),
        "amount": str(sum((row["amount"] for row in independent.values()), Decimal("0.00"))),
        "financially_consistent": not mismatches,
        "mismatches": mismatches,
        "potential_duplicate_groups": candidate_count,
        "potential_duplicates": candidates,
        "email_reconciliation_count": uncertain.count(),
        "email_reconciliation_ids": [row.id for row in uncertain.order_by(EmailLog.id).limit(100)],
        "note": "Repeat candidates may be legitimate. No records were modified. Email ages use record creation, not a delivery guarantee.",
    }
