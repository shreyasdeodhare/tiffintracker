"""Read-only production rendering smoke, no password/order/publication changes."""

import json
import statistics
import time
from datetime import datetime, timezone
from io import BytesIO

from flask_login import login_user
from openpyxl import load_workbook

from app import create_app
from app.application.reporting import period_report
from app.extensions import db
from app.models import Order, User
from app.services import local_today
from app.whatsapp_models import SummaryPublication, WAConnection, WAEvent, WAInbox, WAParticipant

app = create_app()
with app.app_context():
    admin = User.query.filter_by(role="admin", is_active=True).first()
    assert admin is not None
    paths = [
        "/admin/orders",
        "/admin/operations",
        "/admin/whatsapp",
        "/my/counts",
        "/admin/export.xlsx",
        "/admin/whatsapp/export.xlsx",
    ]
    participant = WAParticipant.query.first()
    event = WAEvent.query.first()
    if participant:
        paths.append(f"/admin/participants/{participant.id}")
    if event:
        paths.append(f"/admin/whatsapp/messages/{event.id}")
    if admin.must_change_password:
        print("READONLY_ADMIN_RENDER_NOT_RUN_ACCOUNT_REQUIRES_PASSWORD_CHANGE")
    else:
        for path in paths:
            with app.test_request_context(path, base_url="https://192.168.1.101"):
                # Privileged read-only render test, explicitly not a real browser/password login.
                login_user(admin)
                response = app.full_dispatch_request()
                assert response.status_code == 200, (path, response.status_code)
                if path.endswith("export.xlsx"):
                    response.direct_passthrough = False
                    book = load_workbook(BytesIO(response.get_data()))
                    assert len(book.sheetnames) in {2, 3}
                print("READONLY_ADMIN_RENDER_PASS", path)
    config = db.session.get(WAConnection, 1)
    assert config and config.status == "connected"
    for label, value in [("receiver", config.heartbeat), ("worker", config.worker_heartbeat)]:
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        age = (datetime.now(timezone.utc) - value).total_seconds()
        assert age < 30, (label, age)
        print("FRESH_HEARTBEAT", label, round(age, 2))
    report = period_report(local_today(), local_today())
    assert report["reconciled"]
    print("LIVE_ORDER_COUNT", Order.query.count())
    print("LIVE_REVIEW_COUNT", WAEvent.query.filter_by(state="review").count())
    print(
        "LIVE_PENDING_ERROR_COUNT",
        WAInbox.query.filter(WAInbox.state.in_(["pending", "error"])).count(),
    )
    print("LIVE_PUBLICATION_COUNT", SummaryPublication.query.count())
    print("LIVE_DAILY_TOTALS", report["totals"])
    print("LIVE_PROVISIONAL", report["provisional"])
    assert app.config["SUMMARY_PUBLISHING_ENABLED"]
    print("WEEKLY_PUBLISHING_ENABLED")
    timings = {}
    for path in ["/admin/orders", "/admin/whatsapp/export.xlsx"]:
        values = []
        for _ in range(3):
            with app.test_request_context(path, base_url="https://192.168.1.101"):
                login_user(admin)
                started = time.perf_counter()
                response = app.full_dispatch_request()
                assert response.status_code == 200
                if path.endswith(".xlsx"):
                    response.direct_passthrough = False
                    load_workbook(BytesIO(response.get_data()), data_only=True)
                values.append(round((time.perf_counter() - started) * 1000, 2))
        timings[path] = {
            "samples": 3,
            "min_ms": min(values),
            "median_ms": statistics.median(values),
            "max_ms": max(values),
        }
    print("READ_ONLY_ADMIN_RENDER_AND_WORKBOOK_REOPEN_TIMINGS", json.dumps(timings))
print("PRODUCTION_RENDER_SMOKE_COMPLETE_NO_BUSINESS_OR_PASSWORD_WRITES")
