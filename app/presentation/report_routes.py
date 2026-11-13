"""HTTP validation and rendering only; reporting is an application use case."""

from datetime import date, timedelta
from uuid import UUID, uuid4

from flask import Blueprint, abort, flash, redirect, render_template, request, url_for
from flask_login import current_user, login_required

from app.application.monitoring import tracking_status
from app.application.orders import guest_manual
from app.application.participants import create_guest
from app.application.publications import publication_overview, retry_publication
from app.application.reporting import dashboard_data, participant_detail
from app.domain.orders import BusinessError
from app.presentation.security import admin_required
from app.services import local_today

report_bp = Blueprint("reporting", __name__)


def report_filters():
    try:
        day = date.fromisoformat(request.args.get("day", local_today().isoformat()))
        raw = request.args.get("participant")
        participant_id = int(raw) if raw else None
    except ValueError:
        abort(400)
    source = request.args.get("source", "all")
    status = request.args.get("status", "active")
    if (
        not local_today() - timedelta(days=365) <= day <= local_today()
        or source not in {"all", "manual", "whatsapp"}
        or status not in {"active", "cancelled", "all"}
    ):
        abort(400)
    return day, source, participant_id, status


@report_bp.get("/admin/orders")
@login_required
def dashboard():
    admin_required()
    day, source, participant_id, status = report_filters()
    page = request.args.get("page", 1, type=int)
    if page < 1:
        abort(400)
    return render_template(
        "order_reports.html",
        day=day,
        source=source,
        status=status,
        participant_id=participant_id,
        publications=publication_overview(),
        tracking=tracking_status(),
        submission_key=str(uuid4()),
        **dashboard_data(day, source, participant_id, status, page),
    )


@report_bp.get("/admin/participants/<int:participant_id>")
@login_required
def participant(participant_id):
    admin_required()
    data = participant_detail(participant_id)
    if data is None:
        abort(404)
    return render_template("participant.html", **data)


@report_bp.post("/admin/publications/<int:publication_id>/reconcile")
@login_required
def reconcile_publication(publication_id):
    admin_required()
    try:
        retry_publication(
            publication_id,
            current_user.id,
            request.form.get("outcome"),
            request.form.get("reason", ""),
        )
        flash(
            "Publication outcome recorded. Only confirmed unsent parts may be retried.", "success"
        )
    except BusinessError as exc:
        abort(exc.status, str(exc))
    return redirect(url_for("reporting.dashboard"))


@report_bp.post("/admin/participants")
@login_required
def new_participant():
    admin_required()
    try:
        create_guest(request.form.get("name", "").strip())
    except BusinessError as exc:
        abort(exc.status, str(exc))
    return redirect(url_for("reporting.dashboard"))


@report_bp.post("/admin/participant-orders")
@login_required
def new_order():
    admin_required()
    try:
        identity = int(request.form["participant_id"])
        day = date.fromisoformat(request.form["date"])
        quantity = int(request.form["quantity"])
        key = str(UUID(request.form["submission_key"]))
        guest_manual(
            identity,
            day,
            quantity,
            request.form.get("category", "unspecified"),
            key,
            current_user.id,
        )
    except (ValueError, KeyError) as exc:
        abort(getattr(exc, "status", 400), str(exc))
    return redirect(url_for("reporting.dashboard"))
