import pytest

from app import create_app
from app.extensions import db
from app.models import User


@pytest.fixture()
def app():
    app = create_app(
        {
            "TESTING": True,
            "SECRET_KEY": "t" * 48,
            "SQLALCHEMY_DATABASE_URI": "sqlite://",
            "SQLALCHEMY_ENGINE_OPTIONS": {},
            "WTF_CSRF_ENABLED": False,
            "SESSION_COOKIE_SECURE": False,
        }
    )
    with app.app_context():
        db.create_all()
        for role, email in [
            ("admin", "admin@example.com"),
            ("member", "a@example.com"),
            ("member", "b@example.com"),
        ]:
            user = User(name=email.split("@")[0], email=email, role=role)
            user.set_password("correct-password-123")
            db.session.add(user)
        db.session.commit()
    yield app
    with app.app_context():
        db.session.remove()
        db.drop_all()


@pytest.fixture()
def client(app):
    return app.test_client()


def login(client, email="a@example.com", password="correct-password-123"):
    return client.post(
        "/auth/login", data={"email": email, "password": password}, follow_redirects=False
    )


def cleanup_member_participants(ids):
    """Only used by disposable PostgreSQL fixtures after their orders are removed."""
    from app.whatsapp_models import ParticipantAudit, WAParticipant

    participants = list(
        db.session.scalars(db.select(WAParticipant.id).where(WAParticipant.member_id.in_(ids)))
    )
    ParticipantAudit.query.filter(ParticipantAudit.participant_id.in_(participants)).delete(
        synchronize_session=False
    )
    WAParticipant.query.filter(WAParticipant.id.in_(participants)).delete(synchronize_session=False)


def cleanup_order_audits(ids):
    from app.models import Order, OrderAudit

    orders = db.select(Order.id).where(Order.member_id.in_(ids))
    OrderAudit.query.filter(OrderAudit.order_id.in_(orders)).delete(synchronize_session=False)
