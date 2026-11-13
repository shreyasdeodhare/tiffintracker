from datetime import datetime, timezone
from hashlib import sha256
from hmac import compare_digest

from flask_login import UserMixin
from werkzeug.security import check_password_hash, generate_password_hash

from app.extensions import db, login_manager


def utcnow():
    return datetime.now(timezone.utc)


class User(UserMixin, db.Model):
    __tablename__ = "users"
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(254), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(20), nullable=False, default="member")
    is_active = db.Column(db.Boolean, nullable=False, default=True)
    must_change_password = db.Column(db.Boolean, nullable=False, default=False)
    created_at = db.Column(db.DateTime(timezone=True), nullable=False, default=utcnow)
    billing_email_enabled = db.Column(
        db.Boolean, nullable=False, default=True, server_default=db.true()
    )
    orders = db.relationship("Order", backref="member", lazy=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def get_id(self):
        return f"{self.id}:{sha256(self.password_hash.encode()).hexdigest()}"


@login_manager.user_loader
def load_user(user_id):
    try:
        identity, _token = user_id.split(":", 1)
        user = db.session.get(User, int(identity))
        return user if user and user.is_active and compare_digest(user.get_id(), user_id) else None
    except (TypeError, ValueError):
        return None


class Order(db.Model):
    __tablename__ = "orders"
    id = db.Column(db.Integer, primary_key=True)
    member_id = db.Column(
        db.Integer, db.ForeignKey("users.id", ondelete="RESTRICT"), nullable=True, index=True
    )
    participant_id = db.Column(
        db.Integer,
        db.ForeignKey("wa_participants.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    participant = db.relationship("WAParticipant", foreign_keys=[participant_id])
    category = db.Column(
        db.String(20), nullable=False, default="unspecified", server_default="unspecified"
    )
    updated_at = db.Column(
        db.DateTime(timezone=True),
        nullable=False,
        default=utcnow,
        onupdate=utcnow,
        server_default=db.func.now(),
    )
    order_date = db.Column(db.Date, nullable=False, index=True)
    quantity = db.Column(db.Integer, nullable=False)
    unit_price = db.Column(db.Numeric(10, 2), nullable=False)
    status = db.Column(db.String(20), nullable=False, default="active")
    created_at = db.Column(db.DateTime(timezone=True), nullable=False, default=utcnow)
    cancelled_at = db.Column(db.DateTime(timezone=True))
    submission_key = db.Column(db.String(36), unique=True)
    source = db.Column(db.String(12), nullable=False, default="manual", server_default="manual")
    __table_args__ = (
        db.Index("ix_orders_participant_date", "participant_id", "order_date"),
        db.CheckConstraint(
            "category IN ('lunch','dinner','unspecified')", name="ck_order_category"
        ),
    )

    @property
    def amount(self):
        from app.domain.orders import amount

        return amount(self.quantity, self.unit_price)


class Payment(db.Model):
    __tablename__ = "payments"
    id = db.Column(db.Integer, primary_key=True)
    member_id = db.Column(
        db.Integer, db.ForeignKey("users.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    week_start = db.Column(db.Date, nullable=False)
    week_end = db.Column(db.Date, nullable=False)
    amount = db.Column(db.Numeric(10, 2), nullable=False)
    status = db.Column(db.String(20), nullable=False, default="paid")
    updated_at = db.Column(
        db.DateTime(timezone=True), nullable=False, default=utcnow, onupdate=utcnow
    )
    __table_args__ = (
        db.UniqueConstraint("member_id", "week_start", "week_end", name="uq_payment_member_week"),
    )


class EmailLog(db.Model):
    __tablename__ = "email_logs"
    id = db.Column(db.Integer, primary_key=True)
    member_id = db.Column(
        db.Integer, db.ForeignKey("users.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    week_start = db.Column(db.Date, nullable=False)
    week_end = db.Column(db.Date, nullable=False)
    recipient = db.Column(db.String(254), nullable=False)
    status = db.Column(db.String(20), nullable=False)
    sent_at = db.Column(db.DateTime(timezone=True))
    error = db.Column(db.Text)
    created_at = db.Column(db.DateTime(timezone=True), nullable=False, default=utcnow)


class LoginAttempt(db.Model):
    __tablename__ = "login_attempts"
    id = db.Column(db.Integer, primary_key=True)
    key = db.Column(db.String(300), nullable=False, index=True)
    created_at = db.Column(db.DateTime(timezone=True), nullable=False, default=utcnow, index=True)


class OrderAudit(db.Model):
    __tablename__ = "order_audits"
    id = db.Column(db.Integer, primary_key=True)
    order_id = db.Column(
        db.Integer, db.ForeignKey("orders.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    order = db.relationship("Order")
    actor_id = db.Column(db.Integer, db.ForeignKey("users.id", ondelete="RESTRICT"))
    action = db.Column(db.String(30), nullable=False)
    detail = db.Column(db.JSON, nullable=False)
    created_at = db.Column(db.DateTime(timezone=True), nullable=False, default=utcnow)
