"""Prepare immutable weekly snapshots; transport consumes durable parts separately."""

import hashlib
import json
import secrets
from datetime import datetime, time
from zoneinfo import ZoneInfo

from flask import current_app

from app.application.reporting import period_report, serializable
from app.domain.orders import BusinessError, publication_period
from app.domain.summaries import render_summary
from app.extensions import db
from app.infrastructure.publications import PublicationRepository
from app.whatsapp_models import ReconciliationRun, SummaryPart, SummaryPublication


def prepare_weekly(now=None, source="all"):
    """One snapshot per Saturday–Friday period, only after Friday 14:00 IST.

    Holding the connection row prevents concurrent intake/review while capturing
    the snapshot. REPEATABLE READ is scoped to a separate report connection;
    its first read happens after the serialization lock has been acquired.
    """
    local = ZoneInfo(current_app.config["APP_TIMEZONE"])
    supplied_clock = now is not None
    now = now or datetime.now(local)
    if now.tzinfo is None:
        raise BusinessError("Scheduler timestamp must include its timezone")
    now = now.astimezone(local)
    start, end = publication_period("weekly", now.date())
    cutoff = datetime.combine(end, time(14), tzinfo=local)
    if now < cutoff:
        raise BusinessError("Weekly summary is not due before Friday 14:00 IST")
    repo = PublicationRepository(db.session)
    config = repo.connection(lock=True)
    if not config or not config.group_jid:
        raise BusinessError("Select the authorized group before preparing a summary")
    existing = repo.existing(config.group_jid, "weekly", start, end, source)
    if existing:
        db.session.rollback()
        return existing.id
    report = serializable(period_report(start, end, source))
    if not report["reconciled"]:
        repo.add(
            ReconciliationRun(period_start=start, period_end=end, status="mismatch", result=report)
        )
        db.session.commit()
        raise BusinessError("Financial reconciliation mismatch; publication blocked", 409)
    # A delayed job cannot reproduce a historical cutoff from current mutable
    # rows. Disclose the actual capture time rather than inventing such a snapshot.
    now = now if supplied_clock else datetime.now(local)
    report["cutoff"] = now.isoformat()
    report["scheduled_cutoff"] = cutoff.isoformat()
    report["late_capture"] = now > cutoff.replace(second=59, microsecond=999999)
    report["provisional"] = report["provisional"] or report["late_capture"]
    check = repo.add(
        ReconciliationRun(
            period_start=start,
            period_end=end,
            status="provisional" if report["provisional"] else "reconciled",
            result=report,
        )
    )
    db.session.flush()
    digest = hashlib.sha256(
        json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    publication = repo.add(
        SummaryPublication(
            group_jid=config.group_jid,
            kind="weekly",
            period_start=start,
            period_end=end,
            source=source,
            status="pending",
            snapshot=report,
            snapshot_hash=digest,
            reconciliation_id=check.id,
        )
    )
    db.session.flush()
    for index, body in enumerate(render_summary(publication.id, "weekly", report), 1):
        repo.add(
            SummaryPart(
                publication_id=publication.id,
                position=index,
                body=body,
                message_id="3EB0" + secrets.token_hex(14).upper(),
                status="pending",
            )
        )
    repo.audit(
        publication.id, "prepared", {"snapshot_hash": digest, "provisional": report["provisional"]}
    )
    identity = publication.id
    db.session.commit()
    return identity


def retry_publication(identity, actor_id, outcome, reason):
    """An administrator supplies reconciliation evidence for uncertain sends."""
    if outcome not in {"not_delivered", "delivered"} or not reason or len(reason.strip()) < 10:
        raise BusinessError("Choose provider-observed outcome and supply a reconciliation note")
    repo = PublicationRepository(db.session)
    publication = repo.get(identity, lock=True)
    if not publication or publication.status not in {"failed", "unknown"}:
        raise BusinessError("Only failed or interrupted publications can be reconciled", 409)
    from app.models import utcnow

    for part in repo.parts(identity):
        if part.status in {"failed", "unknown"}:
            part.status = "sent" if outcome == "delivered" else "pending"
            part.sent_at = utcnow() if outcome == "delivered" else None
            part.error = None
    parts = repo.parts(identity)
    publication.status = "sent" if all(part.status == "sent" for part in parts) else "pending"
    publication.sent_at = utcnow() if publication.status == "sent" else None
    publication.error = None
    repo.audit(identity, "reconciled", {"outcome": outcome, "reason": reason[:500]}, actor_id)
    db.session.commit()


def publication_overview():
    repo = PublicationRepository(db.session)
    latest = repo.latest()
    return [
        {
            "id": p.id,
            "kind": p.kind,
            "start": str(p.period_start),
            "end": str(p.period_end),
            "status": p.status,
            "hash": p.snapshot_hash,
            "provisional": p.snapshot["provisional"],
        }
        for p in latest
    ]
