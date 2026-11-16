"""WhatsApp HTTP adapters: validate, authorize and delegate to use cases."""

from datetime import date, timedelta
from io import BytesIO

from flask import Blueprint, abort, flash, redirect, render_template, request, send_file, url_for
from flask_login import current_user, login_required

from app.application import whatsapp_admin as usecases
from app.application.exports import counts_workbook, summaries
from app.domain.orders import BusinessError
from app.extensions import db
from app.infrastructure.bridge_client import bridge
from app.presentation.security import admin_required
from app.services import local_today

whatsapp_bp = Blueprint("whatsapp", __name__)
status_data = usecases.status_data


def selected_day():
    try:
        day = date.fromisoformat(request.args.get("day", local_today().isoformat()))
    except ValueError:
        abort(400)
    if not local_today() - timedelta(days=365) <= day <= local_today():
        abort(400)
    return day


def selected_source():
    source = request.args.get("source", "all")
    if source not in {"all", "manual", "whatsapp"}:
        abort(400)
    return source


def number(name):
    raw = request.form.get(name)
    return int(raw) if raw else None


@whatsapp_bp.get("/admin/whatsapp")
@login_required
def dashboard():
    admin_required()
    day = selected_day()
    source = selected_source()
    state = request.args.get("state", "review")
    page = request.args.get("page", 1, type=int)
    if state not in {"all", "review", "applied", "ignored", "rejected"} or page < 1:
        abort(400)
    return render_template(
        "whatsapp.html",
        day=day,
        source=source,
        state=state,
        summaries=summaries(day, source),
        bridge=status_data(),
        **usecases.dashboard_data(state, page),
    )


@whatsapp_bp.get("/admin/whatsapp/status.json")
@login_required
def status():
    admin_required()
    return status_data()


@whatsapp_bp.get("/admin/whatsapp/qr.png")
@login_required
def qr():
    admin_required()
    try:
        content, mime = bridge("/qr")
        if mime != "image/png":
            abort(404)
        return send_file(BytesIO(content), mimetype="image/png", max_age=0)
    except (OSError, ValueError):
        abort(404)


@whatsapp_bp.post("/admin/whatsapp/relink")
@login_required
def relink():
    admin_required()
    try:
        bridge("/relink", "POST")
        flash("Link your phone using the new QR code.", "success")
    except (OSError, ValueError):
        flash("Connector unavailable.", "error")
    return redirect(url_for("whatsapp.dashboard"))


@whatsapp_bp.post("/admin/whatsapp/group")
@login_required
def select_group():
    admin_required()
    try:
        usecases.select_group(request.form.get("group_jid"), current_user.id)
    except BusinessError as exc:
        db.session.rollback()
        abort(exc.status, str(exc))
    return redirect(url_for("whatsapp.dashboard"))


@whatsapp_bp.get("/admin/whatsapp/messages/<int:event_id>")
@login_required
def detail(event_id):
    admin_required()
    try:
        data = usecases.detail_data(event_id)
    except BusinessError as exc:
        abort(exc.status, str(exc))
    return render_template("whatsapp_detail.html", **data)


@whatsapp_bp.post("/admin/whatsapp/messages/<int:event_id>/review")
@login_required
def review(event_id):
    admin_required()
    try:
        raw = request.form.get("order_date")
        allocations = [
            (int(member), int(quantity))
            for member, quantity in zip(
                request.form.getlist("split_member"),
                request.form.getlist("split_quantity"),
                strict=True,
            )
            if member
        ]
        participants = [
            (int(identity), int(quantity))
            for identity, quantity in zip(
                request.form.getlist("split_participant"),
                request.form.getlist("split_participant_quantity"),
                strict=True,
            )
            if identity
        ]
        usecases.review(
            event_id,
            request.form.get("action"),
            number("quantity"),
            number("member_id"),
            number("target_id"),
            date.fromisoformat(raw) if raw else None,
            allocations,
            current_user.id,
            number("participant_id"),
            participants,
            request.form.get("portion") or None,
        )
        flash("Message decision saved.", "success")
    except BusinessError as exc:
        db.session.rollback()
        abort(exc.status, str(exc))
    except (ValueError, KeyError) as exc:
        db.session.rollback()
        flash(str(exc), "error")
    return redirect(url_for("whatsapp.detail", event_id=event_id))


@whatsapp_bp.post("/admin/whatsapp/errors/<int:inbox_id>/retry")
@login_required
def retry(inbox_id):
    admin_required()
    try:
        usecases.retry(inbox_id)
    except BusinessError as exc:
        db.session.rollback()
        abort(exc.status, str(exc))
    return redirect(url_for("whatsapp.dashboard"))


@whatsapp_bp.post("/admin/whatsapp/participants/<int:participant_id>/map")
@login_required
def map_member(participant_id):
    admin_required()
    try:
        usecases.map_member(participant_id, int(request.form["member_id"]), current_user.id)
    except (ValueError, KeyError) as exc:
        db.session.rollback()
        flash(str(exc), "error")
    return redirect(url_for("whatsapp.dashboard"))


@whatsapp_bp.get("/my/counts")
@login_required
def my_counts():
    day = selected_day()
    return render_template(
        "whatsapp_counts.html", summaries=summaries(day, member_id=current_user.id), day=day
    )


@whatsapp_bp.get("/admin/whatsapp/export.xlsx")
@login_required
def export():
    admin_required()
    day = selected_day()
    return send_file(
        counts_workbook(day, selected_source()),
        as_attachment=True,
        download_name=f"tiffintrack-counts-{day}.xlsx",
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
