"""Queries for connection administration, reviews and operational visibility."""

from sqlalchemy import func
from sqlalchemy.orm import joinedload

from app.models import EmailLog, Order, User
from app.whatsapp_models import (
    ReconciliationRun,
    SummaryPublication,
    WAAudit,
    WAConnection,
    WAEvent,
    WAIdentity,
    WAInbox,
    WAParticipant,
)


class WhatsAppAdminRepository:
    def __init__(self, session):
        self.session = session

    def config(self):
        return self.session.get(WAConnection, 1)

    def event(self, identity):
        return self.session.get(WAEvent, identity)

    def inbox(self, identity):
        return self.session.get(WAInbox, identity)

    def participant(self, identity):
        return self.session.get(WAParticipant, identity)

    def user(self, identity):
        return self.session.get(User, identity) if identity is not None else None

    def members(self):
        return (
            self.session.query(User)
            .filter_by(role="member", is_active=True)
            .order_by(User.name)
            .all()
        )

    def participants(self):
        return (
            self.session.query(WAParticipant)
            .filter_by(is_active=True)
            .order_by(WAParticipant.display_name, WAParticipant.id)
            .all()
        )

    def aliases(self):
        result = {}
        for identity in self.session.query(WAIdentity):
            result.setdefault(identity.participant_id, []).append(identity.jid)
        return result

    def inbox_states(self):
        return dict(self.session.query(WAInbox.state, func.count()).group_by(WAInbox.state).all())

    def oldest_pending(self):
        return self.session.query(func.min(WAInbox.created_at)).filter_by(state="pending").scalar()

    def last_processed(self):
        return (
            self.session.query(func.max(WAInbox.processed_at))
            .filter(WAInbox.state == "processed")
            .scalar()
        )

    def review_count(self):
        return self.session.query(WAEvent).filter_by(state="review").count()

    def email_states(self):
        return dict(
            self.session.query(EmailLog.status, func.count()).group_by(EmailLog.status).all()
        )

    def last_check(self):
        return self.session.query(ReconciliationRun).order_by(ReconciliationRun.id.desc()).first()

    def latest_publications(self):
        return (
            self.session.query(SummaryPublication)
            .order_by(SummaryPublication.id.desc())
            .limit(20)
            .all()
        )

    def dashboard(self, state, page):
        query = WAEvent.query
        if state != "all":
            query = query.filter_by(state=state)
        return dict(
            connection=self.config(),
            events=query.order_by(WAEvent.id.desc()).paginate(
                page=page, per_page=30, error_out=False
            ),
            counts=dict(
                self.session.query(WAEvent.state, func.count()).group_by(WAEvent.state).all()
            ),
            duplicate_count=self.session.query(
                func.coalesce(func.sum(WAInbox.deliveries - 1), 0)
            ).scalar(),
            participants=self.participants(),
            aliases=self.aliases(),
            members=self.members(),
            errors=self.session.query(WAInbox)
            .filter_by(state="error")
            .order_by(WAInbox.id.desc())
            .limit(50)
            .all(),
        )

    def detail(self, event):
        return dict(
            event=event,
            audits=self.session.query(WAAudit)
            .filter_by(event_id=event.id)
            .order_by(WAAudit.id)
            .all(),
            members=self.members(),
            participants=self.participants(),
            orders=self.session.query(Order)
            .options(joinedload(Order.participant))
            .filter_by(source="whatsapp", status="active")
            .filter(Order.order_date == event.order_date)
            .order_by(Order.id)
            .all(),
        )

    def mapping_conflict(self, participant_id, member_id):
        return (
            self.session.query(WAParticipant)
            .filter_by(member_id=member_id)
            .filter(WAParticipant.id != participant_id)
            .first()
        )

    def participant_orders(self, identity):
        return self.session.query(Order).filter_by(participant_id=identity).all()

    def order_event(self, identity):
        return self.session.query(WAEvent).filter_by(order_id=identity).order_by(WAEvent.id).first()
