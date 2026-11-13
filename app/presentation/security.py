from flask import abort
from flask_login import current_user


def admin_required():
    if not current_user.is_authenticated or current_user.role != "admin":
        abort(403)
