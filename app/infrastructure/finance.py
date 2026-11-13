from contextlib import contextmanager
from decimal import Decimal

from sqlalchemy.orm import Session

from app.domain.orders import FinalizedPeriod, week_bounds
from app.extensions import db
from app.models import EmailLog, Payment, User


def lock_member(member_id):
    """Serialize order, invoice and payment changes in PostgreSQL."""
    return db.session.execute(
        db.select(User)
        .where(User.id == member_id)
        .execution_options(populate_existing=True)
        .with_for_update()
    ).scalar_one_or_none()


def ensure_mutable(member_id, day):
    if member_id is None:
        return
    start, end = week_bounds(day)
    paid = Payment.query.filter_by(
        member_id=member_id, week_start=start, week_end=end, status="paid"
    ).first()
    issued = (
        EmailLog.query.filter_by(member_id=member_id, week_start=start, week_end=end)
        .filter(EmailLog.status.in_(["sending", "sent", "unknown"]))
        .first()
    )
    if paid or issued:
        raise FinalizedPeriod("This week is finalized by a payment or issued invoice.")


@contextmanager
def report_snapshot():
    """Keep every sheet in an export on one committed database snapshot."""
    with db.engine.connect() as connection:
        if connection.dialect.name == "postgresql":
            connection = connection.execution_options(isolation_level="REPEATABLE READ")
        with Session(bind=connection) as session:
            if connection.dialect.name == "postgresql":
                session.execute(db.text("SET TRANSACTION READ ONLY"))
            yield session


def weekly_totals(start, end, member_id=None, session=None):
    """Legacy member projection derived from the canonical participant aggregate."""
    from app.infrastructure.reporting import ReportingRepository
    from app.whatsapp_models import WAParticipant

    session = session if session is not None else db.session
    identities = session.query(WAParticipant.id, WAParticipant.member_id)
    payment_query = session.query(Payment.member_id, Payment.status).filter_by(
        week_start=start, week_end=end
    )
    if member_id is not None:
        identities = identities.filter(WAParticipant.member_id == member_id)
        payment_query = payment_query.filter(Payment.member_id == member_id)
    mapping = dict(identities.all())
    if member_id is not None and not mapping:
        return []
    participant_id = next(iter(mapping)) if member_id is not None else None
    payments = dict(payment_query.all())
    grouped = {}
    for row in ReportingRepository(session).totals(start, end, participant_id=participant_id):
        identity = mapping[row["participant_id"]]
        if member_id is not None and identity != member_id:
            continue
        item = grouped.setdefault(
            identity,
            {
                "member_id": identity,
                "orders": 0,
                "amount": Decimal("0.00"),
                "payment_status": payments.get(identity, "pending"),
            },
        )
        item["orders"] += row["quantity"]
        item["amount"] += row["amount"]
    return list(grouped.values())
