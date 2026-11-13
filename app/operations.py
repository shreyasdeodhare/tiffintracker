"""Administrator-only diagnostics, separate from public database liveness."""

from datetime import date, timedelta

from flask import Blueprint, abort, render_template, request
from flask_login import login_required

from .application.monitoring import tracking_status as tracking_status
from .reconciliation import reconcile_week
from .routes import admin_required
from .services import local_today, week_bounds

operations_bp = Blueprint("operations", __name__)


@operations_bp.get("/admin/operations")
@login_required
def dashboard():
    admin_required()
    today = local_today()
    try:
        day = date.fromisoformat(request.args.get("week", today.isoformat()))
    except ValueError:
        abort(400)
    if not today - timedelta(days=365) <= day <= today:
        abort(400)
    start, end = week_bounds(day)
    return render_template(
        "operations.html", tracking=tracking_status(), reconciliation=reconcile_week(start, end)
    )
