"""Publication persistence and atomic preparation; no transport calls."""

from app.extensions import db
from app.whatsapp_models import (
    PublicationAudit,
    ReconciliationRun,
    SummaryPart,
    SummaryPublication,
    WAConnection,
)


class PublicationRepository:
    def __init__(self, session):
        self.session = session

    def connection(self, lock=False):
        query = db.select(WAConnection).where(WAConnection.id == 1)
        if lock:
            query = query.execution_options(populate_existing=True).with_for_update()
        return self.session.execute(query).scalar_one_or_none()

    def existing(self, group, kind, start, end, source):
        return (
            self.session.query(SummaryPublication)
            .filter_by(
                group_jid=group, kind=kind, period_start=start, period_end=end, source=source
            )
            .first()
        )

    def get(self, identity, lock=False):
        query = db.select(SummaryPublication).where(SummaryPublication.id == identity)
        if lock:
            query = query.with_for_update().execution_options(populate_existing=True)
        return self.session.execute(query).scalar_one_or_none()

    def parts(self, identity):
        return (
            self.session.query(SummaryPart)
            .filter_by(publication_id=identity)
            .order_by(SummaryPart.position)
            .all()
        )

    def latest(self, limit=20):
        return (
            self.session.query(SummaryPublication)
            .order_by(SummaryPublication.id.desc())
            .limit(limit)
            .all()
        )

    def last_reconciliation(self):
        return self.session.query(ReconciliationRun).order_by(ReconciliationRun.id.desc()).first()

    def add(self, value):
        self.session.add(value)
        return value

    def audit(self, identity, action, detail, actor_id=None):
        self.session.add(
            PublicationAudit(
                publication_id=identity, action=action, detail=detail, actor_id=actor_id
            )
        )
