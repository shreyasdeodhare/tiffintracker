"""Persistence queries for itemized source history and the bounded rollout re-check."""

from app.whatsapp_models import WAAudit, WAEvent


class MealOrderRepository:
    def creation(self, order_id):
        entries = WAAudit.query.filter_by(order_id=order_id, action="created").limit(2).all()
        if len(entries) != 1:
            raise ValueError("Source creation audit missing or nonunique")
        return entries[0]

    def unresolved_between(self, group_jid, start, end):
        return (
            WAEvent.query.filter(
                WAEvent.state == "review",
                WAEvent.group_jid == group_jid,
                WAEvent.timestamp >= start,
                WAEvent.timestamp < end,
            )
            .order_by(WAEvent.timestamp, WAEvent.id)
            .with_for_update()
            .all()
        )
