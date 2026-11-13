from urllib.parse import urlsplit

from flask import Blueprint, flash, redirect, render_template, request, session, url_for
from flask_login import current_user, login_required, login_user, logout_user

from .application import accounts
from .domain.orders import BusinessError
from .extensions import login_manager

auth_bp = Blueprint("auth", __name__, url_prefix="/auth")


def safe_next_url(target):
    if not target or not target.startswith("/") or target.startswith("//") or "\\" in target:
        return False
    parsed = urlsplit(target)
    return not parsed.scheme and not parsed.netloc and not any(ord(c) < 32 for c in target)


@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("main.dashboard"))
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        try:
            user = accounts.authenticate(email, password, request.remote_addr or "unknown")
        except BusinessError as exc:
            flash(str(exc), "error")
            return render_template("login.html"), exc.status
        if user:
            session.clear()
            login_user(user, remember=False, fresh=True)
            session.permanent = True
            if user.must_change_password:
                return redirect(url_for("auth.change_password"))
            target = request.args.get("next")
            return redirect(target if safe_next_url(target) else url_for("main.dashboard"))
        flash("Invalid email or password.", "error")
    return render_template("login.html")


@auth_bp.post("/logout")
@login_required
def logout():
    logout_user()
    session.clear()
    return redirect(url_for("auth.login"))


@auth_bp.route("/change-password", methods=["GET", "POST"])
@login_required
def change_password():
    if request.method == "POST":
        current = request.form.get("current_password", "")
        new = request.form.get("new_password", "")
        confirm = request.form.get("confirm_password", "")
        try:
            accounts.change_password(current_user, current, new, confirm)
        except BusinessError as exc:
            flash(str(exc), "error")
        else:
            user = current_user._get_current_object()
            session.clear()
            login_user(user, fresh=True)
            session.permanent = True
            flash("Password updated.", "success")
            return redirect(url_for("main.dashboard"))
    return render_template("change_password.html")


@login_manager.unauthorized_handler
def unauthorized():
    return redirect(url_for("auth.login", next=request.path))
