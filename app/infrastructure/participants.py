"""Canonical participant persistence; authentication accounts are optional."""

import re

from sqlalchemy import event
from sqlalchemy.orm import Session

from app.extensions import db
from app.models import Order, User
from app.whatsapp_models import ParticipantAudit, WAIdentity, WAParticipant


class ParticipantRepository:
    def __init__(self, session):
        self.session = session

    def record(self, participant, action, detail):
        self.session.add(ParticipantAudit(participant=participant, action=action, detail=detail))

    def get(self, identity):
        return self.session.get(WAParticipant, identity)

    def create_guest(self, name):
        participant = WAParticipant(display_name=name, automatically_created=False)
        self.session.add(participant)
        self.record(participant, "created", {"source": "manual"})
        return participant

    def for_member(self, member_id):
        participant = self.session.query(WAParticipant).filter_by(member_id=member_id).first()
        if participant is None:
            member = self.session.get(User, member_id)
            if member is None:
                raise ValueError("Member does not exist")
            participant = WAParticipant(
                member_id=member_id, display_name=member.name, automatically_created=False
            )
            self.session.add(participant)
            self.session.flush()
        return participant

    def lock(self, participant_id):
        return self.session.execute(
            db.select(WAParticipant)
            .where(WAParticipant.id == participant_id)
            .execution_options(populate_existing=True)
            .with_for_update()
        ).scalar_one_or_none()

    def resolve(self, payload):
        identities = list(dict.fromkeys([payload["sender"], *payload.get("aliases", [])]))
        if len(identities) > 4 or any(
            not re.fullmatch(r"\d{5,30}@(s\.whatsapp\.net|lid)", jid) for jid in identities
        ):
            raise ValueError("Invalid stable participant identifier")
        found = self.session.scalars(
            db.select(WAIdentity.participant_id).where(WAIdentity.jid.in_(identities))
        ).all()
        if len(set(found)) > 1:
            raise ValueError(
                "Conflicting participant identities require administrator reconciliation"
            )
        if found:
            participant = self.session.get(WAParticipant, found[0])
            if payload.get("push_name"):
                name = str(payload["push_name"])[:100]
                if name != participant.display_name:
                    self.record(
                        participant,
                        "display_name",
                        {"previous": participant.display_name, "current": name},
                    )
                    participant.display_name = name
        else:
            participant = WAParticipant(
                display_name=str(payload.get("push_name") or "WhatsApp participant")[:100],
                automatically_created=True,
            )
            self.session.add(participant)
            self.session.flush()
        for jid in identities:
            if self.session.get(WAIdentity, jid) is None:
                self.session.add(WAIdentity(jid=jid, participant_id=participant.id))
                self.record(participant, "alias_observed", {"jid": jid})
        return participant


@event.listens_for(Session, "before_flush")
def attach_legacy_member_orders(session, _context, _instances):
    """Batch adapter for retained member-based scripts; no guest login creation."""
    orders = [
        o
        for o in session.new
        if isinstance(o, Order) and o.participant_id is None and o.participant is None
    ]
    if not orders:
        return
    if any(o.member_id is None for o in orders):
        raise ValueError("Every order requires a participant")
    ids = {o.member_id for o in orders}
    participants = {
        p.member_id: p for p in session.new if isinstance(p, WAParticipant) and p.member_id in ids
    }
    with session.no_autoflush:
        participants.update(
            {
                p.member_id: p
                for p in session.query(WAParticipant).filter(WAParticipant.member_id.in_(ids)).all()
            }
        )
        missing = ids - set(participants)
        users = (
            {u.id: u for u in session.query(User).filter(User.id.in_(missing)).all()}
            if missing
            else {}
        )
    for identity in missing:
        user = users.get(identity)
        if user is None:
            continue
        participant = WAParticipant(
            member_id=identity, display_name=user.name, automatically_created=False
        )
        session.add(participant)
        session.add(
            ParticipantAudit(
                participant=participant, action="legacy_member_link", detail={"member_id": identity}
            )
        )
        participants[identity] = participant
    for order in orders:
        if order.member_id in participants:
            order.participant = participants[order.member_id]
