"""Legacy HTTP surface; business operations live in application services."""

from datetime import date, timedelta
from uuid import UUID

from flask import Blueprint, abort, flash, redirect, render_template, request, send_file, url_for
from flask_login import current_user, login_required
from sqlalchemy.exc import IntegrityError

from app.application import legacy_dashboard, orders
from app.domain.orders import BusinessError
from app.extensions import db
from app.presentation.security import admin_required as admin_required
from app.reports import weekly_workbook
from app.services import FinalizedPeriod, local_today, week_bounds

main_bp = Blueprint("main", __name__)


def selected_week():
    try:
        day = date.fromisoformat(request.args.get("week", local_today().isoformat()))
    except ValueError:
        abort(400)
    if not local_today() - timedelta(days=365) <= day <= local_today():
        abort(400)
    return day


@main_bp.get("/legacy-dashboard")
@main_bp.get("/")
@login_required
def dashboard():
    if current_user.role == "admin" and request.path == "/":
        return redirect(url_for("reporting.dashboard"))
    day = selected_week()
    page = request.args.get("page", 1, type=int)
    if page < 1:
        abort(400)
    return render_template("dashboard.html", **legacy_dashboard.data(current_user, day, page))


@main_bp.post("/admin/members")
@login_required
def create_member():
    admin_required()
    abort(410, description="Email account creation is retired. Use participant order tracking.")


@main_bp.post("/orders")
@login_required
def add_order():
    key = request.form.get("submission_key")
    if key:
        try:
            key = str(UUID(key))
        except ValueError:
            abort(400)
    try:
        day = date.fromisoformat(request.form.get("order_date", ""))
        quantity = int(request.form.get("quantity", "1"))
        member_id = current_user.id
        if current_user.role == "admin":
            if not request.form.get("member_id"):
                abort(400)
            member_id = int(request.form["member_id"])
        orders.create_manual(
            member_id,
            day,
            quantity,
            key,
            request.form.get("category", "unspecified"),
            current_user.id,
        )
        flash("Order recorded.", "success")
    except FinalizedPeriod as exc:
        db.session.rollback()
        abort(409, str(exc))
    except BusinessError as exc:
        db.session.rollback()
        if exc.status != 400:
            abort(exc.status, str(exc))
        flash(str(exc), "error")
    except (ValueError, TypeError, IntegrityError):
        db.session.rollback()
        flash("Invalid order or duplicate submission. Check date, quantity and member.", "error")
    return redirect(url_for("main.dashboard"))


@main_bp.post("/orders/<int:order_id>/cancel")
@login_required
def cancel_order(order_id):
    try:
        changed = orders.cancel(order_id, current_user.id, current_user.role == "admin")
        flash("Order cancelled." if changed else "Order is already cancelled.", "success")
    except FinalizedPeriod as exc:
        db.session.rollback()
        abort(409, str(exc))
    except BusinessError as exc:
        db.session.rollback()
        abort(exc.status, str(exc))
    return redirect(url_for("main.dashboard"))


@main_bp.post("/admin/payments/<int:member_id>/paid")
@login_required
def mark_paid(member_id):
    admin_required()
    try:
        orders.record_payment(
            member_id,
            date.fromisoformat(request.form["week_start"]),
            date.fromisoformat(request.form["week_end"]),
        )
        flash("Payment marked paid.", "success")
    except (ValueError, KeyError, IntegrityError):
        db.session.rollback()
        flash("Unable to update payment.", "error")
    return redirect(url_for("main.dashboard"))


@main_bp.get("/admin/export.xlsx")
@login_required
def export_xlsx():
    admin_required()
    start, end = week_bounds(selected_week())
    return send_file(
        weekly_workbook(start, end),
        as_attachment=True,
        download_name=f"tiffintrack-{start}-to-{end}.xlsx",
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


@main_bp.get("/healthz")
def healthz():
    return {"status": "ok"}
