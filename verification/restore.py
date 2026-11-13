"""Read-only snapshot export and isolated restore; no live schema or data writes."""

import hashlib
import json
import os
import subprocess
from pathlib import Path

import psycopg
from openpyxl import load_workbook
from psycopg import sql

from app import create_app
from app.application.exports import counts_workbook
from app.application.reporting import period_report
from app.domain.orders import reporting_week
from app.services import local_today

restore = "tiffintrack_final_restore_test"
backup = Path("/var/backups/tiffintrack/final-acceptance-20261112")
backup.mkdir(mode=0o700, exist_ok=True)
os.umask(0o077)
with psycopg.connect(dbname="postgres") as admin:
    assert not admin.execute("SELECT 1 FROM pg_database WHERE datname=%s", (restore,)).fetchone(), (
        "Refusing to overwrite existing restore database"
    )


def table_state(connection):
    names = connection.execute(
        "SELECT tablename FROM pg_tables WHERE schemaname='public' ORDER BY tablename"
    ).fetchall()
    result = {}
    for (name,) in names:
        rows = connection.execute(
            sql.SQL("SELECT row_to_json(t)::text FROM {} t").format(sql.Identifier(name))
        ).fetchall()
        digest = hashlib.sha256("\n".join(sorted(row[0] for row in rows)).encode()).hexdigest()
        result[name] = {"rows": len(rows), "sha256": digest}
    return result


file = backup / "consistent-production.dump"
with psycopg.connect(dbname="tiffintrack") as live:
    live.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
    snapshot = live.execute("SELECT pg_export_snapshot()").fetchone()[0]
    before = table_state(live)
    subprocess.run(
        [
            "pg_dump",
            "-Fc",
            "--no-owner",
            "--no-acl",
            "--snapshot=" + snapshot,
            "--file=" + str(file),
            "tiffintrack",
        ],
        check=True,
    )
file.with_suffix(".dump.sha256").write_text(
    hashlib.sha256(file.read_bytes()).hexdigest() + "  " + str(file) + "\n"
)
subprocess.run(["sha256sum", "-c", str(file.with_suffix(".dump.sha256"))], check=True)
subprocess.run(["createdb", restore], check=True)
subprocess.run(
    ["pg_restore", "--exit-on-error", "--no-owner", "--no-acl", "-d", restore, str(file)],
    check=True,
)
with psycopg.connect(dbname=restore) as restored:
    after = table_state(restored)
assert before == after, "Restored row hashes differ from exported production snapshot"
print(
    "ISOLATED_RESTORE_ALL_TABLE_COUNTS_AND_CONTENT_HASHES_MATCH",
    json.dumps({k: v["rows"] for k, v in after.items()}, sort_keys=True),
)
with create_app(
    {
        "SECRET_KEY": "isolated-recovery-test-" * 3,
        "SQLALCHEMY_DATABASE_URI": "postgresql+psycopg:///" + restore,
    }
).app_context():
    result = period_report(*reporting_week(local_today()))
    assert result["reconciled"]
    book = load_workbook(counts_workbook(local_today(), "all"), data_only=True)
    assert book.sheetnames == ["Daily", "Weekly", "Monthly"]
    print("RESTORED_APPLICATION_RECONCILIATION_AND_EXCEL_PASS", result["totals"])
print("LIVE_DATABASE_NOT_RESTORED_OR_MODIFIED")
