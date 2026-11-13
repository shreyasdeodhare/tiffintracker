from datetime import datetime, timezone

from app.extensions import db
from app.infrastructure.whatsapp_admin import WhatsAppAdminRepository


def tracking_status():
    now = datetime.now(timezone.utc)
    repo = WhatsAppAdminRepository(db.session)
    config = repo.config()

    def age(value):
        if value is None:
            return None
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return max(0, int((now - value).total_seconds()))

    receiver_age = age(config.heartbeat) if config else None
    worker_age = age(config.worker_heartbeat) if config else None
    inbox = repo.inbox_states()
    oldest = repo.oldest_pending()
    if not config or not config.group_jid:
        state = "not_configured"
    elif receiver_age is None or receiver_age > 30 or config.status != "connected":
        state = "receiver_unavailable"
    elif worker_age is None or worker_age > 30:
        state = "worker_unavailable"
    elif inbox.get("error", 0):
        state = "processing_errors"
    elif oldest and age(oldest) > 60:
        state = "processing_backlog"
    else:
        state = "components_ready_delivery_not_proven"
    last_check = repo.last_check()
    return {
        "state": state,
        "receiver_status": config.status if config else "not_configured",
        "receiver_heartbeat_age_seconds": receiver_age,
        "worker_heartbeat_age_seconds": worker_age,
        "last_received_age_seconds": age(config.last_received) if config else None,
        "oldest_pending_age_seconds": age(oldest),
        "inbox": inbox,
        "review": repo.review_count(),
        "email_states": repo.email_states(),
        "last_processed": str(repo.last_processed() or "not_observed"),
        "last_reconciliation": str(last_check.created_at) if last_check else None,
        "weekly_publications": [
            {
                "id": p.id,
                "status": p.status,
                "start": str(p.period_start),
                "end": str(p.period_end),
                "sent_at": str(p.sent_at) if p.sent_at else None,
            }
            for p in repo.latest_publications()
        ],
    }
