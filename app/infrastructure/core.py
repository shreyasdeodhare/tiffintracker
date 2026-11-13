"""Persistence operations used by account/manual-order application use cases."""

from sqlalchemy.orm import selectinload

from app.models import Order, OrderAudit, Payment, User
from app.whatsapp_models import WAEvent


class CoreRepository:
    def __init__(self, session):
        self.session = session

    def user(self, identity):
        return self.session.get(User, identity)

    def order(self, identity):
        return self.session.get(Order, identity)

    def refresh(self, entity):
        self.session.refresh(entity)

    def members(self):
        return self.session.query(User).filter_by(role="member").order_by(User.name).all()

    def orders_page(self, start, end, page, member_id=None):
        query = Order.query.options(selectinload(Order.member)).filter(
            Order.order_date.between(start, end)
        )
        if member_id is not None:
            query = query.filter_by(member_id=member_id)
        return query.order_by(Order.order_date.desc(), Order.id.desc()).paginate(
            page=page, per_page=50, error_out=False
        )

    def submission(self, key):
        return self.session.query(Order).filter_by(submission_key=key).first()

    def payment(self, member_id, start, end):
        return (
            self.session.query(Payment)
            .filter_by(member_id=member_id, week_start=start, week_end=end)
            .first()
        )

    def origin(self, identity):
        return self.session.query(WAEvent).filter_by(order_id=identity, kind="original").first()

    def add(self, entity):
        self.session.add(entity)
        return entity

    def audit(self, order, action, detail, actor_id=None):
        self.session.add(OrderAudit(order=order, action=action, detail=detail, actor_id=actor_id))
