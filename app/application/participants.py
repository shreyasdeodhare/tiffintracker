from app.domain.orders import BusinessError
from app.extensions import db
from app.infrastructure.participants import ParticipantRepository


def create_guest(name):
    if not name or len(name) > 100:
        raise BusinessError("Provide a participant name of 1–100 characters")
    participant = ParticipantRepository(db.session).create_guest(name)
    db.session.commit()
    return participant.id
