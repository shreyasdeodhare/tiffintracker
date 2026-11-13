from datetime import datetime, timedelta, timezone

from sqlalchemy import func

from app.models import LoginAttempt, User


class AccountRepository:
    def __init__(self, session):
        self.session = session

    def limited(self, key):
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=15)
        return (
            self.session.query(func.count(LoginAttempt.id))
            .filter(LoginAttempt.key == key, LoginAttempt.created_at >= cutoff)
            .scalar()
            >= 8
        )

    def find(self, email):
        return self.session.query(User).filter_by(email=email, is_active=True).first()

    def failure(self, key):
        self.session.add(LoginAttempt(key=key))

    def clear(self, key):
        self.session.query(LoginAttempt).filter_by(key=key).delete(synchronize_session=False)

    def add(self, user):
        self.session.add(user)

    def any_user(self, email):
        return self.session.query(User).filter_by(email=email).first()

    def member(self, email):
        return self.session.query(User).filter_by(email=email, role="member").first()

    def prune(self, cutoff):
        return (
            self.session.query(LoginAttempt)
            .filter(LoginAttempt.created_at < cutoff)
            .delete(synchronize_session=False)
        )
