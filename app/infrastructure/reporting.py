"""One SQL projection for canonical participant, source and period reporting."""

from sqlalchemy import func, select
from sqlalchemy.orm import joinedload

from app.models import Order, OrderAudit
from app.whatsapp_models import (
    ParticipantAudit,
    WAAudit,
    WAConnection,
    WAEvent,
    WAIdentity,
    WAInbox,
    WAParticipant,
)


class ReportingRepository:
    def __init__(self, session):
        self.session = session

    def orders(self, start, end, source="all", participant_id=None, status="active"):
        query = (
            self.session.query(Order)
            .options(joinedload(Order.participant))
            .filter(Order.order_date.between(start, end))
        )
        if source != "all":
            query = query.filter(Order.source == source)
        if participant_id is not None:
            query = query.filter(Order.participant_id == participant_id)
        if status != "all":
            query = query.filter(Order.status == status)
        return query

    def totals(self, start, end, source="all", participant_id=None, status="active"):
        query = self.orders(start, end, source, participant_id, status).join(
            WAParticipant, WAParticipant.id == Order.participant_id
        )
        return [
            dict(
                participant_id=pid,
                id=pid,
                name=name,
                orders=records,
                records=records,
                quantity=int(quantity),
                amount=value,
            )
            for pid, name, records, quantity, value in query.with_entities(
                WAParticipant.id,
                WAParticipant.display_name,
                func.count(Order.id),
                func.sum(Order.quantity),
                func.sum(Order.quantity * Order.unit_price),
            )
            .group_by(WAParticipant.id, WAParticipant.display_name)
            .order_by(WAParticipant.display_name, WAParticipant.id)
        ]

    def independent_totals(self, start, end, source="all", participant_id=None):
        # Independent SQL uses raw order rows, grouped by participant AND source,
        # without the presentation join. It detects missing/duplicated joined
        # rows and source inconsistencies without materializing every ORM order.
        query = self.orders(start, end, source, participant_id)
        source_rows = (
            query.with_entities(
                Order.participant_id,
                Order.source,
                func.count(Order.id),
                func.sum(Order.quantity),
                func.sum(Order.quantity * Order.unit_price),
            )
            .group_by(Order.participant_id, Order.source)
            .all()
        )
        calculated = {}
        for identity, origin, records, quantity, value in source_rows:
            row = calculated.setdefault(identity, {"records": 0, "quantity": 0, "amount": 0})
            row["records"] += records
            row["quantity"] += int(quantity)
            row["amount"] += value
        return source_rows, calculated

    def reference_issues(self, start, end, source="all", participant_id=None):
        query = self.orders(start, end, source, participant_id, "all")
        wrong = (
            query.join(WAParticipant, WAParticipant.id == Order.participant_id)
            .filter(Order.member_id.is_distinct_from(WAParticipant.member_id))
            .with_entities(Order.id)
            .limit(100)
            .all()
        )
        missing = (
            query.outerjoin(WAAudit, (WAAudit.order_id == Order.id) & (WAAudit.action == "created"))
            .filter(Order.source == "whatsapp")
            .with_entities(Order.id, func.count(WAAudit.id))
            .group_by(Order.id)
            .having(func.count(WAAudit.id) != 1)
            .limit(100)
            .all()
        )
        return [{"order_id": row[0], "type": "member_participant_reference"} for row in wrong] + [
            {"order_id": row[0], "type": "source_audit_reference", "references": row[1]}
            for row in missing
        ]

    def exceptions(self, start, end, source="all", participant_id=None):
        config = self.session.get(WAConnection, 1)
        events = self.session.query(WAEvent).filter(
            WAEvent.order_date.between(start, end), WAEvent.state == "review"
        )
        if config and config.group_jid:
            events = events.filter(WAEvent.group_jid == config.group_jid)
        if participant_id is not None:
            # Unresolved sender events cannot safely be attributed to one guest.
            events = events.filter(
                (WAEvent.participant_id == participant_id) | (WAEvent.participant_id.is_(None))
            )
        review = events.count() if source != "manual" else 0
        # Conservative global barrier: pending/error payload dates may be invalid
        # or unresolved. Never label a final summary complete while they exist.
        unresolved = (
            self.session.query(WAInbox).filter(WAInbox.state.in_(["pending", "error"])).count()
            if source != "manual"
            else 0
        )
        enabled = config.enabled_since if config else None
        return config, review, unresolved, enabled

    def daily_trend(self, start, end, source="all", participant_id=None):
        return (
            self.orders(start, end, source, participant_id)
            .with_entities(
                Order.order_date,
                func.sum(Order.quantity),
                func.sum(Order.quantity * Order.unit_price),
            )
            .group_by(Order.order_date)
            .order_by(Order.order_date)
            .all()
        )

    def participant(self, participant_id):
        return self.session.get(WAParticipant, participant_id)

    def participant_count(self):
        return self.session.query(WAParticipant).count()

    def identity_audits(self, identity):
        return (
            self.session.query(ParticipantAudit)
            .filter_by(participant_id=identity)
            .order_by(ParticipantAudit.id.desc())
            .limit(200)
            .all()
        )

    def manual_audits(self, identity):
        return (
            self.session.query(OrderAudit)
            .join(Order, Order.id == OrderAudit.order_id)
            .filter(Order.participant_id == identity)
            .order_by(OrderAudit.id.desc())
            .limit(200)
            .all()
        )

    def aliases(self, participant_id):
        return self.session.scalars(
            select(WAIdentity.jid).where(WAIdentity.participant_id == participant_id)
        ).all()

    def history(self, participant_id, limit=100):
        return (
            self.session.query(Order)
            .filter_by(participant_id=participant_id)
            .order_by(Order.order_date.desc(), Order.id.desc())
            .limit(limit)
            .all()
        )

    def events(self, participant_id, limit=100):
        return (
            self.session.query(WAEvent)
            .filter_by(participant_id=participant_id)
            .order_by(WAEvent.id.desc())
            .limit(limit)
            .all()
        )

    def audits(self, participant_id, limit=200):
        return (
            self.session.query(WAAudit)
            .join(Order, Order.id == WAAudit.order_id)
            .filter(Order.participant_id == participant_id)
            .order_by(WAAudit.id.desc())
            .limit(limit)
            .all()
        )
