"""Durable live-message ingestion, identity, decisions and connection state."""

from app.extensions import db

from .models import utcnow


class WAConnection(db.Model):
    __tablename__ = "wa_connection"
    id = db.Column(db.Integer, primary_key=True)
    group_jid = db.Column(db.String(100), unique=True)
    group_name = db.Column(db.String(200))
    enabled_since = db.Column(db.DateTime(timezone=True))
    selected_by = db.Column(db.Integer, db.ForeignKey("users.id", ondelete="RESTRICT"))
    status = db.Column(db.String(40), nullable=False, default="not_started")
    heartbeat = db.Column(db.DateTime(timezone=True))
    worker_heartbeat = db.Column(db.DateTime(timezone=True))
    last_received = db.Column(db.DateTime(timezone=True))
    error = db.Column(db.String(200))


class WAParticipant(db.Model):
    __tablename__ = "wa_participants"
    id = db.Column(db.Integer, primary_key=True)
    member_id = db.Column(
        db.Integer, db.ForeignKey("users.id", ondelete="RESTRICT"), nullable=True, unique=True
    )
    is_active = db.Column(db.Boolean, nullable=False, default=True, server_default=db.true())
    created_at = db.Column(
        db.DateTime(timezone=True), nullable=False, default=utcnow, server_default=db.func.now()
    )
    updated_at = db.Column(
        db.DateTime(timezone=True),
        nullable=False,
        default=utcnow,
        onupdate=utcnow,
        server_default=db.func.now(),
    )
    verified_contact = db.Column(db.String(100))
    display_name = db.Column(db.String(100), nullable=False)
    automatically_created = db.Column(db.Boolean, nullable=False, default=True)
    member = db.relationship("User", foreign_keys=[member_id])


class WAIdentity(db.Model):
    __tablename__ = "wa_identities"
    jid = db.Column(db.String(100), primary_key=True)
    participant_id = db.Column(
        db.Integer,
        db.ForeignKey("wa_participants.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )


class WAInbox(db.Model):
    __tablename__ = "wa_inbox"
    id = db.Column(db.Integer, primary_key=True)
    event_key = db.Column(db.String(64), nullable=False, unique=True)
    payload = db.Column(db.JSON, nullable=False)
    state = db.Column(
        db.String(12), nullable=False, default="pending", server_default="pending", index=True
    )
    attempts = db.Column(db.Integer, nullable=False, default=0, server_default="0")
    deliveries = db.Column(db.Integer, nullable=False, default=1, server_default="1")
    error = db.Column(db.String(100))
    processed_at = db.Column(db.DateTime(timezone=True))
    created_at = db.Column(
        db.DateTime(timezone=True), nullable=False, default=utcnow, server_default=db.func.now()
    )
    __table_args__ = (
        db.CheckConstraint("state IN ('pending','processed','error')", name="ck_wa_inbox_state"),
    )


class WAEvent(db.Model):
    __tablename__ = "wa_events"
    id = db.Column(db.Integer, primary_key=True)
    inbox_id = db.Column(
        db.Integer, db.ForeignKey("wa_inbox.id", ondelete="RESTRICT"), nullable=False, unique=True
    )
    group_jid = db.Column(db.String(100), nullable=False)
    message_id = db.Column(db.String(150), nullable=False)
    kind = db.Column(db.String(12), nullable=False)
    sender = db.Column(db.String(100), nullable=False)
    participant_id = db.Column(db.Integer, db.ForeignKey("wa_participants.id", ondelete="RESTRICT"))
    timestamp = db.Column(db.DateTime(timezone=True), nullable=False)
    body = db.Column(db.Text, nullable=False)
    category = db.Column(db.String(20), nullable=False)
    confidence = db.Column(db.String(10), nullable=False)
    state = db.Column(db.String(20), nullable=False, index=True)
    reason = db.Column(db.String(500), nullable=False)
    order_date = db.Column(db.Date, nullable=False)
    quantity = db.Column(db.Integer)
    order_id = db.Column(db.Integer, db.ForeignKey("orders.id", ondelete="RESTRICT"), index=True)
    target_message_id = db.Column(db.String(150))
    participant = db.relationship("WAParticipant")
    order = db.relationship("Order")
    __table_args__ = (
        db.Index("ix_wa_source", "group_jid", "message_id", "kind"),
        db.Index("ix_wa_participant_time", "participant_id", "timestamp"),
        db.CheckConstraint(
            "state IN ('review','applied','ignored','rejected')", name="ck_wa_event_state"
        ),
        db.CheckConstraint("confidence IN ('high','medium','low')", name="ck_wa_event_confidence"),
        db.CheckConstraint(
            "quantity IS NULL OR quantity BETWEEN 1 AND 20", name="ck_wa_event_quantity"
        ),
    )


class WAAudit(db.Model):
    __tablename__ = "wa_audits"
    id = db.Column(db.Integer, primary_key=True)
    event_id = db.Column(
        db.Integer, db.ForeignKey("wa_events.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    order_id = db.Column(db.Integer, db.ForeignKey("orders.id", ondelete="RESTRICT"))
    actor_id = db.Column(db.Integer, db.ForeignKey("users.id", ondelete="RESTRICT"))
    action = db.Column(db.String(40), nullable=False)
    detail = db.Column(db.JSON, nullable=False)
    created_at = db.Column(db.DateTime(timezone=True), nullable=False, default=utcnow)


class WAAuth(db.Model):
    __tablename__ = "wa_auth"
    key = db.Column(db.String(300), primary_key=True)
    encrypted_value = db.Column(db.Text, nullable=False)


class ReconciliationRun(db.Model):
    __tablename__ = "reconciliation_runs"
    id = db.Column(db.Integer, primary_key=True)
    period_start = db.Column(db.Date, nullable=False)
    period_end = db.Column(db.Date, nullable=False)
    status = db.Column(db.String(20), nullable=False)
    result = db.Column(db.JSON, nullable=False)
    created_at = db.Column(db.DateTime(timezone=True), nullable=False, default=utcnow)


class SummaryPublication(db.Model):
    __tablename__ = "summary_publications"
    id = db.Column(db.Integer, primary_key=True)
    group_jid = db.Column(db.String(100), nullable=False)
    kind = db.Column(db.String(10), nullable=False)
    period_start = db.Column(db.Date, nullable=False)
    period_end = db.Column(db.Date, nullable=False)
    source = db.Column(db.String(12), nullable=False, default="all")
    status = db.Column(db.String(12), nullable=False, default="pending")
    snapshot = db.Column(db.JSON, nullable=False)
    snapshot_hash = db.Column(db.String(64), nullable=False)
    reconciliation_id = db.Column(
        db.Integer, db.ForeignKey("reconciliation_runs.id", ondelete="RESTRICT"), nullable=False
    )
    created_at = db.Column(db.DateTime(timezone=True), nullable=False, default=utcnow)
    sent_at = db.Column(db.DateTime(timezone=True))
    error = db.Column(db.String(100))
    attempts = db.Column(db.Integer, nullable=False, default=0)
    __table_args__ = (
        db.UniqueConstraint(
            "group_jid", "kind", "period_start", "period_end", "source", name="uq_summary_period"
        ),
        db.CheckConstraint(
            "status IN ('pending','sending','sent','failed','unknown')", name="ck_summary_state"
        ),
        db.CheckConstraint("kind='weekly'", name="ck_summary_weekly_only"),
        db.CheckConstraint("source IN ('all','manual','whatsapp')", name="ck_summary_source"),
    )


class SummaryPart(db.Model):
    __tablename__ = "summary_parts"
    id = db.Column(db.Integer, primary_key=True)
    publication_id = db.Column(
        db.Integer,
        db.ForeignKey("summary_publications.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    position = db.Column(db.Integer, nullable=False)
    body = db.Column(db.Text, nullable=False)
    message_id = db.Column(db.String(100), unique=True, nullable=False)
    status = db.Column(db.String(12), nullable=False, default="pending")
    sent_at = db.Column(db.DateTime(timezone=True))
    error = db.Column(db.String(100))
    __table_args__ = (
        db.UniqueConstraint("publication_id", "position", name="uq_summary_part"),
        db.CheckConstraint(
            "status IN ('pending','sending','sent','failed','unknown')",
            name="ck_summary_part_state",
        ),
    )


class PublicationAudit(db.Model):
    __tablename__ = "publication_audits"
    id = db.Column(db.Integer, primary_key=True)
    publication_id = db.Column(
        db.Integer,
        db.ForeignKey("summary_publications.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    actor_id = db.Column(db.Integer, db.ForeignKey("users.id", ondelete="RESTRICT"))
    action = db.Column(db.String(30), nullable=False)
    detail = db.Column(db.JSON, nullable=False)
    created_at = db.Column(db.DateTime(timezone=True), nullable=False, default=utcnow)


class ParticipantAudit(db.Model):
    __tablename__ = "participant_audits"
    id = db.Column(db.Integer, primary_key=True)
    participant_id = db.Column(
        db.Integer,
        db.ForeignKey("wa_participants.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    participant = db.relationship("WAParticipant")
    actor_id = db.Column(db.Integer, db.ForeignKey("users.id", ondelete="RESTRICT"))
    action = db.Column(db.String(40), nullable=False)
    detail = db.Column(db.JSON, nullable=False)
    created_at = db.Column(db.DateTime(timezone=True), nullable=False, default=utcnow)
