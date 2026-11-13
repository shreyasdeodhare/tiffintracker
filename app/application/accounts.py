from app.domain.orders import BusinessError
from app.extensions import db
from app.infrastructure.accounts import AccountRepository
from app.models import User


def authenticate(email, password, ip):
    repo = AccountRepository(db.session)
    account = "account:" + email[:254]
    network = "ip:" + ip[:45]
    if repo.limited(network) or repo.limited(account):
        raise BusinessError("Too many login attempts. Please try again later.", 429)
    user = repo.find(email)
    if user and user.check_password(password):
        repo.clear(account)
        db.session.commit()
        return user
    repo.failure(network)
    repo.failure(account)
    db.session.commit()
    return None


def change_password(user, current, new, confirm):
    if not user.check_password(current):
        raise BusinessError("Current password is incorrect.")
    if not 12 <= len(new) <= 128 or new != confirm or new == current:
        raise BusinessError("Use a different matching password of 12–128 characters.")
    user.set_password(new)
    user.must_change_password = False
    db.session.commit()


def create_member(name, email, password):
    if (
        not name
        or len(name) > 100
        or "@" not in email
        or len(email) > 254
        or not 12 <= len(password) <= 128
    ):
        raise BusinessError(
            "Provide a valid name, email and temporary password of 12–128 characters."
        )
    user = User(name=name, email=email, role="member", is_active=True, must_change_password=True)
    user.set_password(password)
    AccountRepository(db.session).add(user)
    db.session.commit()


def create_admin(name, email, password):
    repo = AccountRepository(db.session)
    if repo.any_user(email):
        raise BusinessError("A user with this email already exists")
    if (
        not name
        or len(name) > 100
        or "@" not in email
        or len(email) > 254
        or not 12 <= len(password) <= 128
    ):
        raise BusinessError("Provide valid details and a password of 12–128 characters")
    user = User(name=name, email=email, role="admin", is_active=True)
    user.set_password(password)
    repo.add(user)
    db.session.commit()


def activate_member(email, active):
    user = AccountRepository(db.session).member(email)
    if not user:
        raise BusinessError("Member not found")
    user.is_active = active
    db.session.commit()


def reset_member_password(email, password):
    user = AccountRepository(db.session).member(email)
    if not user:
        raise BusinessError("Member not found")
    if not 12 <= len(password) <= 128:
        raise BusinessError("Password must have 12–128 characters")
    user.set_password(password)
    user.must_change_password = True
    db.session.commit()


def prune_login_attempts():
    from datetime import datetime, timedelta, timezone

    count = AccountRepository(db.session).prune(datetime.now(timezone.utc) - timedelta(days=30))
    db.session.commit()
    return count
