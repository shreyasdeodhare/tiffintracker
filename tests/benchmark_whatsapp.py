"""Synthetic live-event workload. ONLY a disposable PostgreSQL *_test database."""

import hashlib
import json
import os
import resource
import statistics
import time
from datetime import datetime, timedelta, timezone

from flask_migrate import upgrade
from sqlalchemy.engine import make_url

from app import create_app
from app.extensions import db
from app.models import Order
from app.services import local_today
from app.whatsapp_models import WAConnection, WAEvent, WAInbox
from app.whatsapp_service import period_totals, process_next


def main():
    url = os.environ["TEST_POSTGRES_URL"]
    if not make_url(url).database.endswith("_test"):
        raise SystemExit("Disposable *_test PostgreSQL database required")
    app = create_app({"TESTING": True, "SECRET_KEY": "b" * 48, "SQLALCHEMY_DATABASE_URI": url})
    group = "120363123123123@g.us"
    now = datetime.now(timezone.utc)
    payloads = []
    for member in range(20):
        sender = f"91980000{member:04d}@s.whatsapp.net"
        for number in range(3):
            payloads.append(
                {
                    "kind": "original",
                    "group_jid": group,
                    "message_id": f"bench-{member}-{number}",
                    "sender": sender,
                    "timestamp": now.timestamp(),
                    "body": "2 tiffins",
                    "push_name": f"Synthetic member {member}",
                }
            )
        payloads.append(
            {
                "kind": "edit",
                "group_jid": group,
                "message_id": f"bench-{member}-0",
                "sender": sender,
                "timestamp": now.timestamp() + 1,
                "body": "3 tiffins",
            }
        )
        payloads.append(
            {
                "kind": "original",
                "group_jid": group,
                "message_id": f"bench-cancel-{member}",
                "target_message_id": f"bench-{member}-1",
                "sender": sender,
                "timestamp": now.timestamp() + 2,
                "body": "Cancel my lunch",
            }
        )
    with app.app_context():
        assert db.engine.dialect.name == "postgresql"
        upgrade()
        if WAInbox.query.count() or Order.query.filter_by(source="whatsapp").count():
            raise SystemExit("WhatsApp workload requires an empty WhatsApp inbox/order dataset")
        config = db.session.get(WAConnection, 1)
        config.group_jid, config.enabled_since = group, now - timedelta(days=1)
        db.session.commit()
        for payload in payloads:
            key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
            # Two transport deliveries per source event; only one pending record.
            for _ in range(2):
                db.session.execute(
                    db.text(
                        "INSERT INTO wa_inbox (event_key,payload) VALUES (:key,:payload) ON CONFLICT(event_key) DO UPDATE SET deliveries=wa_inbox.deliveries+1"
                    ),
                    {"key": key, "payload": json.dumps(payload)},
                )
        db.session.commit()
        latencies = []
        started = time.perf_counter()
        while True:
            start = time.perf_counter()
            if not process_next():
                break
            latencies.append((time.perf_counter() - start) * 1000)
        elapsed = time.perf_counter() - started
        rows = period_totals(local_today(), local_today(), "whatsapp")
        assert len(rows) == 20 and all(row["orders"] == 2 and row["quantity"] == 5 for row in rows)
        assert sum(row["quantity"] for row in rows) == 100
        assert WAEvent.query.filter_by(state="applied").count() == 100
        assert Order.query.filter_by(source="whatsapp").count() == 60
        assert WAInbox.query.filter_by(state="error").count() == 0
        sorted_latencies = sorted(latencies)

        def percentile(p):
            return round(
                sorted_latencies[min(len(latencies) - 1, int((len(latencies) - 1) * p))], 2
            )

        result = {
            "scope": "Synthetic PostgreSQL durable-inbox worker on host; excludes actual WhatsApp delivery, TLS and container limits",
            "participants": 20,
            "transport_deliveries": 200,
            "unique_events": 100,
            "original_orders": 60,
            "edits": 20,
            "cancellations": 20,
            "final_active_orders": 40,
            "final_tiffin_quantity": 100,
            "duplicate_deliveries_ignored": 100,
            "errors": 0,
            "duration_seconds": round(elapsed, 3),
            "events_per_second": round(100 / elapsed, 2),
            "p50_ms": round(statistics.median(latencies), 2),
            "p95_ms": percentile(0.95),
            "p99_ms": percentile(0.99),
            "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        }
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
