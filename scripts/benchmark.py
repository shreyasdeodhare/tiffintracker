"""Repeatable application-layer benchmark; disposable PostgreSQL *_test only."""

import json
import os
import resource
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from decimal import Decimal
from uuid import uuid4

from sqlalchemy.engine import make_url
from werkzeug.security import generate_password_hash

from app import create_app
from app.extensions import db
from app.models import Order, User
from app.services import local_today, week_bounds, weekly_totals


def main():
    url = os.environ["TEST_POSTGRES_URL"]
    if not make_url(url).database.endswith("_test"):
        raise SystemExit("Disposable *_test database required")
    app = create_app(
        {
            "TESTING": True,
            "SECRET_KEY": "b" * 48,
            "SQLALCHEMY_DATABASE_URI": url,
            "WTF_CSRF_ENABLED": False,
            "SESSION_COOKIE_SECURE": False,
        }
    )
    with app.app_context():
        if db.engine.dialect.name != "postgresql":
            raise SystemExit("PostgreSQL required")
        password_hash = generate_password_hash("benchmark-password-123")
        users = [
            User(
                name=f"Bench {i}",
                email=f"bench-{i}@test.invalid",
                role="admin" if i == 0 else "member",
                is_active=True,
                password_hash=password_hash,
            )
            for i in range(1000)
        ]
        db.session.add_all(users)
        db.session.flush()
        today = local_today()
        db.session.add_all(
            [
                Order(
                    member_id=user.id,
                    order_date=today - timedelta(days=day),
                    quantity=1,
                    unit_price=Decimal("80.25"),
                    status="active",
                )
                for user in users[1:]
                for day in range(14)
            ]
        )
        db.session.commit()
        connections = db.session.execute(
            db.text("SELECT count(*) FROM pg_stat_activity WHERE datname=current_database()")
        ).scalar()
    clients = []
    for i in range(5):
        client = app.test_client()
        client.post(
            "/auth/login",
            data={"email": f"bench-{i + 1}@test.invalid", "password": "benchmark-password-123"},
        )
        clients.append(client)
    admin = app.test_client()
    admin.post(
        "/auth/login", data={"email": "bench-0@test.invalid", "password": "benchmark-password-123"}
    )
    results = {}

    def measure(name, call, count=30, concurrency=5):
        def timed(index):
            start = time.perf_counter()
            status = call(index)
            return (time.perf_counter() - start) * 1000, status

        start = time.perf_counter()
        # Each thread has its own Flask client; batches avoid sharing cookie jars.
        samples = []
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            for offset in range(0, count, concurrency):
                samples.extend(executor.map(timed, range(offset, min(offset + concurrency, count))))
        elapsed = time.perf_counter() - start
        values = sorted(item[0] for item in samples)

        def percentile(p):
            return round(values[min(len(values) - 1, int((len(values) - 1) * p))], 2)

        results[name] = {
            "requests": count,
            "concurrency": concurrency,
            "duration_seconds": round(elapsed, 3),
            "throughput_per_second": round(count / elapsed, 2),
            "errors": sum(status not in (200, 302) for _, status in samples),
            "p50_ms": round(statistics.median(values), 2),
            "p95_ms": percentile(0.95),
            "p99_ms": percentile(0.99),
        }

    measure(
        "login",
        lambda i: (
            app.test_client()
            .post(
                "/auth/login",
                data={
                    "email": f"bench-{i % 5 + 1}@test.invalid",
                    "password": "benchmark-password-123",
                },
            )
            .status_code
        ),
    )
    measure("member_dashboard", lambda i: clients[i % 5].get("/").status_code)
    report_clients = []
    for _ in range(5):
        report_client = app.test_client()
        report_client.post(
            "/auth/login",
            data={"email": "bench-0@test.invalid", "password": "benchmark-password-123"},
        )
        report_clients.append(report_client)
    measure(
        "participant_dashboard", lambda i: report_clients[i % 5].get("/admin/orders").status_code
    )

    measure(
        "order_creation",
        lambda i: (
            clients[i % 5]
            .post(
                "/orders",
                data={
                    "order_date": today.isoformat(),
                    "quantity": 1,
                    "submission_key": str(uuid4()),
                },
            )
            .status_code
        ),
    )

    def billing(_index):
        with app.app_context():
            start, end = week_bounds(today)
            weekly_totals(start, end)
            return 200

    measure("weekly_aggregation", billing)
    measure(
        "excel_export",
        lambda i: admin.get("/admin/export.xlsx").status_code,
        count=5,
        concurrency=1,
    )
    output = {
        "scope": "Flask test client on host; PostgreSQL real; excludes TLS/Gunicorn/network",
        "dataset": {"registered_users": 1000, "seed_orders": 13986},
        "db_connections_before": connections,
        "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "cpu_user_seconds": resource.getrusage(resource.RUSAGE_SELF).ru_utime,
        "results": results,
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
