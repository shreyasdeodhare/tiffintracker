"""Intake and source-order lookups with stable participant ownership."""

from app.extensions import db
from app.models import Order
from app.whatsapp_models import WAAudit, WAConnection, WAEvent, WAInbox, WAParticipant


class InboxRepository:
    def __init__(self, session):
        self.session = session

    def connection_lock(self):
        return self.session.execute(
            db.select(WAConnection)
            .where(WAConnection.id == 1)
            .execution_options(populate_existing=True)
            .with_for_update()
        ).scalar_one()

    def next_pending(self):
        return self.session.execute(
            db.select(WAInbox)
            .where(WAInbox.state == "pending")
            .order_by(WAInbox.id)
            .limit(1)
            .with_for_update(skip_locked=True)
        ).scalar_one_or_none()

    def inbox(self, identity):
        return self.session.get(WAInbox, identity)

    def member_participant(self, identity):
        return self.session.query(WAParticipant).filter_by(member_id=identity).first()

    def original(self, event, other=False):
        query = self.session.query(WAEvent).filter_by(
            group_jid=event.group_jid, message_id=event.message_id, kind="original"
        )
        if other:
            query = query.filter(WAEvent.id != event.id)
        return query.order_by(WAEvent.id).first()

    def source_has_created_orders(self, event):
        return (
            self.session.query(WAAudit.id)
            .join(WAEvent, WAAudit.event_id == WAEvent.id)
            .filter(
                WAEvent.group_jid == event.group_jid,
                WAEvent.message_id == event.message_id,
                WAEvent.kind == "original",
                WAAudit.action == "created",
                WAAudit.order_id.isnot(None),
            )
            .first()
            is not None
        )

    def later_change(self, event, order_id):
        return (
            self.session.query(WAEvent.id)
            .join(WAAudit, WAAudit.event_id == WAEvent.id)
            .filter(
                WAAudit.order_id == order_id,
                WAEvent.timestamp > event.timestamp,
                WAEvent.state == "applied",
            )
            .first()
        )

    def target_orders(self, event):
        target = event.message_id if event.kind in {"edit", "delete"} else event.target_message_id
        origins = self.session.query(WAEvent.id).filter_by(
            group_jid=event.group_jid, participant_id=event.participant_id, kind="original"
        )
        if target:
            origins = origins.filter_by(message_id=target)
        else:
            origins = origins.filter(
                WAEvent.order_date == event.order_date, WAEvent.timestamp <= event.timestamp
            )
        # A join/subquery avoids loading every source message into Python.
        identities = self.session.query(WAAudit.order_id).filter(
            WAAudit.event_id.in_(origins), WAAudit.action == "created"
        )
        return (
            self.session.query(Order)
            .filter(Order.id.in_(identities), Order.status == "active")
            .order_by(Order.id)
            .all()
        )
