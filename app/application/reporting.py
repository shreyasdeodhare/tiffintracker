"""Snapshot reporting and independent verification orchestration."""

from datetime import datetime, time, timedelta, timezone
from decimal import Decimal
from zoneinfo import ZoneInfo

from flask import current_app

from app.domain.orders import month_bounds, reporting_week
from app.infrastructure.reporting import ReportingRepository
from app.models import Order
from app.services import report_snapshot


def period_report(start, end, source="all", participant_id=None, session=None):
    if session is None:
        with report_snapshot() as snapshot:
            return period_report(start, end, source, participant_id, snapshot)
    if source not in {"all", "manual", "whatsapp"}:
        raise ValueError("Invalid order source")
    repo = ReportingRepository(session)
    rows = repo.totals(start, end, source, participant_id)
    sql_sources, calculated = repo.independent_totals(start, end, source, participant_id)
    projected = {row["participant_id"]: row for row in rows}
    discrepancies = repo.reference_issues(start, end, source, participant_id)
    for pid in sorted(set(projected) | set(calculated)):
        a, b = projected.get(pid), calculated.get(pid)
        if not a or not b or any(a[key] != b[key] for key in ["records", "quantity", "amount"]):
            discrepancies.append({"participant_id": pid, "type": "participant_totals"})
    sources = {}
    for pid, origin, records, quantity, value in sql_sources:
        row = sources.setdefault(origin, {"records": 0, "quantity": 0, "amount": Decimal("0.00")})
        row["records"] += records
        row["quantity"] += int(quantity)
        row["amount"] += value
    totals = {
        key: sum((row[key] for row in rows), Decimal("0.00") if key == "amount" else 0)
        for key in ["records", "quantity", "amount"]
    }
    for key in totals:
        if totals[key] != sum((row[key] for row in sources.values()), 0):
            discrepancies.append({"type": "source_totals", "field": key})
    config, review, unresolved, enabled = repo.exceptions(start, end, source, participant_id)
    if enabled and enabled.tzinfo is None:
        enabled = enabled.replace(tzinfo=timezone.utc)
    start_stamp = datetime.combine(
        start, time.min, tzinfo=ZoneInfo(current_app.config["APP_TIMEZONE"])
    )
    coverage = source != "manual" and (not enabled or enabled > start_stamp)

    def stale(value):
        if value is None:
            return True
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return datetime.now(timezone.utc) - value > timedelta(seconds=30)

    if source != "manual" and (
        not config
        or config.status != "connected"
        or stale(config.heartbeat)
        or stale(config.worker_heartbeat)
    ):
        coverage = True
    cancelled = repo.orders(start, end, source, participant_id, "cancelled").count()
    return {
        "start": str(start),
        "end": str(end),
        "source": source,
        "participants": rows,
        "participant_count": len(rows),
        "totals": totals,
        "sources": sources,
        "cancelled": cancelled,
        "review": review,
        "unresolved": unresolved,
        "coverage_incomplete": coverage,
        "discrepancies": discrepancies,
        "reconciled": not discrepancies,
        "provisional": bool(review or unresolved or coverage),
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }


def serializable(value):
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, dict):
        return {key: serializable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [serializable(item) for item in value]
    return value


def dashboard_data(day, source="all", participant_id=None, status="active", page=1):
    with report_snapshot() as session:
        periods = {"daily": (day, day), "weekly": reporting_week(day), "monthly": month_bounds(day)}
        reports = {
            key: period_report(*bounds, source, participant_id, session)
            for key, bounds in periods.items()
        }
        repo = ReportingRepository(session)
        offset = (page - 1) * 50
        detail_count = repo.orders(day, day, source, participant_id, status).count()
        orders = (
            repo.orders(day, day, source, participant_id, status)
            .order_by(Order.id)
            .offset(offset)
            .limit(50)
            .all()
        )
        # Materialize presentation fields before closing the snapshot session.
        details = [
            {
                "id": o.id,
                "participant_id": o.participant_id,
                "name": o.participant.display_name,
                "date": o.order_date,
                "category": o.category,
                "quantity": o.quantity,
                "amount": o.amount,
                "source": o.source,
                "status": o.status,
            }
            for o in orders
        ]
        trend = repo.daily_trend(*periods["monthly"], source, participant_id)
        pages = max(
            1,
            (max(detail_count, *(report["participant_count"] for report in reports.values())) + 49)
            // 50,
        )
        # All reconciliation and totals above use full projections. Slice only
        # presentation rows, never the data used to verify financial totals.
        reports = {
            label: {**report, "participants": report["participants"][offset : offset + 50]}
            for label, report in reports.items()
        }
        return {
            "reports": reports,
            "details": details,
            "trend": trend,
            "daily_average": sum(row[1] for row in trend if row[0] <= day) / day.day,
            "participant_count": repo.participant_count(),
            "report_page": page,
            "report_pages": pages,
            "detail_count": detail_count,
        }


def participant_detail(identity):
    with report_snapshot() as session:
        repo = ReportingRepository(session)
        participant = repo.participant(identity)
        if participant is None:
            return None
        return {
            "participant": participant,
            "aliases": repo.aliases(identity),
            "orders": repo.history(identity),
            "events": repo.events(identity),
            "audits": repo.audits(identity),
            "identity_audits": repo.identity_audits(identity),
            "manual_audits": repo.manual_audits(identity),
        }
