"""Real queue/concurrency/rollback tests, restricted to a disposable PostgreSQL database."""

import hashlib
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest
from flask_migrate import upgrade
from sqlalchemy import text
from sqlalchemy.engine import make_url

from app import create_app
from app.extensions import db
from app.models import Order, User
from app.whatsapp_models import (
    ParticipantAudit,
    WAAudit,
    WAConnection,
    WAEvent,
    WAIdentity,
    WAInbox,
    WAParticipant,
)
from app.whatsapp_service import process_next


@pytest.mark.skipif(not os.getenv("TEST_POSTGRES_URL"), reason="TEST_POSTGRES_URL not set")
def test_postgres_whatsapp_durable_concurrency_and_rollback():
    import json

    url = os.environ["TEST_POSTGRES_URL"]
    assert make_url(url).database.endswith("_test")
    app = create_app({"TESTING": True, "SECRET_KEY": "w" * 48, "SQLALCHEMY_DATABASE_URI": url})
    group = "120363987654321@g.us"
    with app.app_context():
        assert db.engine.dialect.name == "postgresql"
        upgrade()
        config = db.session.get(WAConnection, 1)
        config.group_jid, config.enabled_since = (
            group,
            datetime.now(timezone.utc) - timedelta(days=1),
        )
        db.session.commit()
        initial = Order.query.count()
        engine = db.engine
    event_key = hashlib.sha256(b"concurrent-whatsapp").hexdigest()
    payload = {
        "kind": "original",
        "group_jid": group,
        "message_id": "concurrent-message",
        "sender": "919123456789@s.whatsapp.net",
        "timestamp": datetime.now(timezone.utc).timestamp(),
        "body": "2 tiffins",
        "push_name": "Anonymized test",
    }

    def deliver(_):
        with engine.begin() as connection:
            connection.execute(
                text(
                    "INSERT INTO wa_inbox (event_key,payload) VALUES (:key,:payload) ON CONFLICT (event_key) DO NOTHING"
                ),
                {"key": event_key, "payload": json.dumps(payload)},
            )

    def process(_):
        with app.app_context():
            return process_next()

    try:
        with ThreadPoolExecutor(max_workers=8) as executor:
            list(executor.map(deliver, range(8)))
        with ThreadPoolExecutor(max_workers=4) as executor:
            list(executor.map(process, range(4)))
        with app.app_context():
            assert Order.query.count() == initial + 1
            assert WAInbox.query.filter_by(event_key=event_key).count() == 1
            assert WAEvent.query.filter_by(message_id="concurrent-message").one().state == "applied"
            # A processing crash rolls back user/order/audit work, retaining a retryable inbox record.
            second = WAInbox(
                event_key="simulated-failure", payload={**payload, "message_id": "crash"}
            )
            db.session.add(second)
            db.session.commit()
            with patch(
                "app.whatsapp_service.apply_event", side_effect=RuntimeError("test failure")
            ):
                assert process_next()
            assert Order.query.count() == initial + 1
            assert WAInbox.query.filter_by(event_key="simulated-failure").one().state == "error"
            assert WAEvent.query.filter_by(message_id="crash").count() == 0
    finally:
        with app.app_context():
            # Remove only synthetic integration identities and their dependent rows.
            inbox_ids = list(
                db.session.scalars(
                    db.select(WAInbox.id).where(
                        WAInbox.event_key.in_([event_key, "simulated-failure"])
                    )
                )
            )
            events = WAEvent.query.filter(WAEvent.inbox_id.in_(inbox_ids)).all()
            event_ids = [event.id for event in events]
            participant_ids = [event.participant_id for event in events if event.participant_id]
            order_ids = [event.order_id for event in events if event.order_id]
            member_ids = list(
                db.session.scalars(
                    db.select(WAParticipant.member_id).where(WAParticipant.id.in_(participant_ids))
                )
            )
            WAAudit.query.filter(WAAudit.event_id.in_(event_ids)).delete(synchronize_session=False)
            WAEvent.query.filter(WAEvent.id.in_(event_ids)).delete(synchronize_session=False)
            Order.query.filter(Order.id.in_(order_ids)).delete(synchronize_session=False)
            WAIdentity.query.filter(WAIdentity.participant_id.in_(participant_ids)).delete(
                synchronize_session=False
            )
            ParticipantAudit.query.filter(
                ParticipantAudit.participant_id.in_(participant_ids)
            ).delete(synchronize_session=False)
            WAParticipant.query.filter(WAParticipant.id.in_(participant_ids)).delete(
                synchronize_session=False
            )
            User.query.filter(User.id.in_(member_ids)).delete(synchronize_session=False)
            WAInbox.query.filter(WAInbox.id.in_(inbox_ids)).delete(synchronize_session=False)
            config = db.session.get(WAConnection, 1)
            config.group_jid, config.enabled_since = None, None
            db.session.commit()
