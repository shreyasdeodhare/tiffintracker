# TiffinTrack implementation and deployment verification

Verified 11 November 2026 on the authorized support server. Domain setup is deferred by user request. This report describes observed results and remaining limitations.

## A. Audit findings

| Severity | Finding | Resolution |
|---|---|---|
| Critical | Password-change route passed current_user's proxy back into Flask-Login, causing recursive user loading and a test-process crash | Unwrap the real user before restoring the login session; regression verified locally and through live HTTPS |
| High | New backdated orders could change an already-paid bill | Shared finalized-period checks reject creation and cancellation after payment; PostgreSQL member locks serialize changes |
| High | Issued invoices did not prevent later order changes | Sending/sent/unknown invoice state freezes the same week; billing totals are recomputed under the member lock |
| High | Password resets did not revoke active sessions | Password-derived session identity validated on every subsequent request |
| High | Healthy app existed but no accounts, TiffinTrack proxy or operational cron | Created random-password administrator, IP HTTPS Nginx site and host jobs |
| High | PostgreSQL listened on every interface with broad shared-host HBA/firewall rules | Restricted this database's network HBA and CONNECT grants while preserving unrelated database services |
| Medium | Archive omitted migrations/alembic.ini | Added working Alembic configuration and verified fresh/upgrade migrations |
| Medium | Browser form replay could create duplicate orders | Unique UUID submission key and server-side replay handling; eight concurrent requests create exactly one order |
| Medium | PostgreSQL streaming user cursor crossed delivery commits | Fetch bounded member IDs then process recipients independently; 101-recipient PostgreSQL regression test |
| Medium | Mail retry CLI rejected even confirmed failures | Explicit failed-state retry and status inspection; unknown outcomes remain blocked |
| Medium | Last 50 orders hid older weekly entries | Added pagination with ownership filters intact |
| Medium | Disabled members vanished from administrator billing summaries | Include their historical bills, restrict new order dropdown to active members |
| Medium | Excel logic mixed with HTTP and allocated normal worksheets | Separate write-only workbook service, numeric two-decimal money and formula protection |
| Medium | Build dependencies were unpinned; test tools were runtime dependencies | Pin runtime dependency versions and separate development tools |
| Low | Unbounded container logs; stale environment/documentation examples | Rotate Docker logs, update configuration examples and operating instructions |

## B. Significant changes

- app/auth.py: repair password-change login restoration.
- app/models.py: session identity/revocation and nullable unique submission-key persistence.
- app/services.py: centralized period locking and immutability checks; shared Decimal aggregation retained.
- app/routes.py: enforce paid/invoiced period consistency, lock payment/order operations, paginate history, prevent duplicate browser submissions, retain disabled-member history.
- app/reports.py: focused memory-conscious workbook creation with monetary formatting and spreadsheet safety.
- app/templates/dashboard.html: pagination and unique order form keys.
- scripts/weekly_billing.py: PostgreSQL-safe recipient iteration, locked fresh totals, confirmed-failure retries and status CLI.
- migrations/alembic.ini and 0003_submission_and_email.py: operable migration runner, conflict preflight, normalized-email constraint and submission uniqueness.
- docker-compose.yml: preserve dedicated server network and one app container, tagged image and bounded logs.
- requirements.txt / requirements-dev.txt / pyproject.toml: pinned dependencies and reproducible formatting/lint configuration.
- tests/conftest.py: let request app contexts end between requests so session revocation is tested realistically.
- tests/test_integrity.py, test_postgres_concurrency.py, test_postgres_mail.py: financial, authorization/session, export, pagination, replay/concurrency and batch-email regressions.
- scripts/benchmark.py, README.md, ACCESS.md, deployment examples and benchmark-results.json: repeatable measurement and accurate operations documentation.

## C. Design

Single responsibility separates HTTP, authentication, persistence, accounting, exports and SMTP. DRY keeps one billing aggregate and one finalized-week check. Explicit PostgreSQL member locks provide one understandable concurrency boundary rather than introducing a queue, cache or extra service. Decimal arithmetic, historical prices and database constraints preserve financial integrity. Composition through focused functions applies SOLID pragmatically; no unnecessary class framework was added.

## Dâ€“F. Functional, security and database checks

The full server suite uses portable SQLite fixtures where appropriate plus real PostgreSQL tests in a disposable database. Tests cover login/logout, incorrect passwords/throttling, unsafe redirects, forced password changes, reset session revocation, disabled sessions, role restrictions, cross-member cancellation/assignment, CSRF, validation, historical pricing, cancellations, paid/invoiced-period mutation prevention, duplicate payments, exports reopened with OpenPyXL, formula safety, pagination, delivery idempotency, timeout ambiguity and confirmed retry.

Real PostgreSQL checks executed Alembic migrations, a quantity CHECK violation and transaction rollback, eight concurrent submissions sharing one key, and 101-recipient delivery processing across per-recipient commits with one ambiguous failure and no duplicate rerun.

The original archive's initial test run failed in password-change recursion; it was corrected. New session tests exposed an overlong fixture app context, which was corrected without weakening assertions. The 101-recipient test was initially run against a populated benchmark DB and counted its 999 additional members; rerunning in its required empty disposable DB passed. Final isolated full server suite: **29 passed in 12.70 seconds**, no skips. Compileall, Ruff lint and Ruff format checks also passed. Docker build and Compose validation passed.

Production upgraded from 0002_integrity to 0003_submission after restore validation. No existing users/orders were present before the deployment; a bootstrap administrator was created. All production changes were additive, with source/environment and database snapshots retained on the server.

Live HTTPS checks verified administrator login, forced password change, dashboard, XLSX download/open, CSRF rejection and Secure cookie. The bootstrap account was returned to required-password-change state and its current temporary password saved only in the protected server access file. No live customer order/payment test records were inserted.

## G. Measured performance

Synthetic staging dataset: 1,000 users, 13,986 seed orders; 30 additional benchmark order writes produce 14,016 orders. Each workload ran 30 operations at concurrency five, except five sequential exports. These are short application-layer microbenchmarks through Flask's test client with real host PostgreSQL, not sustained network/TLS/Gunicorn tests and not inside Docker's CPU/memory caps.

| Workload | Requests | Concurrency | P50 ms | P95 ms | P99 ms | Operations/s | Errors |
|---|---:|---:|---:|---:|---:|---:|---:|
| login | 30 | 5 | 104.4 | 130.31 | 131.22 | 41.49 | 0 |
| member_dashboard | 30 | 5 | 33.27 | 144.53 | 145.71 | 86.61 | 0 |
| order_creation | 30 | 5 | 23.8 | 46.81 | 50.75 | 156.86 | 0 |
| weekly_aggregation | 30 | 5 | 21.38 | 38.26 | 38.66 | 164.41 | 0 |
| excel_export | 5 | 1 | 390.19 | 427.14 | 427.14 | 2.55 | 0 |

The benchmark process peak RSS was 280,260 KiB and user CPU time 7.252 seconds. Database connections before the workload: one; peak connection usage was NOT MEASURED. Percentiles are sample estimates from small runs. The running app was observed at 140 MiB / 512 MiB and 0.04% CPU after checks; these are idle observations, not peak-under-load guarantees. Raw JSON accompanies this report.

## H. Deployment and recovery

One application container `tiffintrack`, image `tiffintrack:20261111`; host PostgreSQL, host Nginx, host cron. Docker build and Compose validation passed. Runtime inspection: healthy, non-root user app, read-only root, memory 536870912 bytes, NanoCPUs 1500000000, host binding 127.0.0.1:8087 to container 8000. Two Gunicorn workers and two threads; capabilities dropped, no-new-privileges, PID limit, bounded tmpfs/logs and restart policy enabled.

Nginx syntax passed and reload succeeded. IP HTTPS uses a temporary self-signed RSA certificate valid for 90 days, including LAN and VPN IP SANs. The /health endpoint returned HTTP 200 with database status ok through Nginx. Direct workstation access to private LAN/VPN addresses timed out; server-local HTTPS was verified. ACCESS.md gives a working SSH forwarding method for networks without a route. Public ingress reachability is NOT VERIFIED.

SMTP Office365 STARTTLS and authentication succeeded using credentials held only in .env mode 0600. Actual SMTP message sending and inbox delivery are NOT VERIFIED; no message was sent. Mocked delivery tests and dry runs passed. Host billing cron runs Monday 09:00 Asia/Kolkata; backup 02:00 and maintenance 03:00 daily. Log rotation is installed. Cron entries were inspected but future scheduled firing is NOT VERIFIED.

A pre-upgrade production pg_dump archive restored into an isolated database. A populated recovery drill matched schema revision 0003_submission, 1,000 users and 14,016 orders between source and restored database, then verified health/login/dashboard on both. Checksums and pg_restore --exit-on-error were used. A live native local backup also succeeded. Backup retention is 14 days; pre-release source/database snapshots are separately retained. No credentials or dumps are shipped in the ZIP.

The TiffinTrack DB permits its dedicated 10.250.252.0/24 Docker network/account; network access from other addresses is rejected and PUBLIC CONNECT revoked. The shared PostgreSQL listener remains globally bound for other applications; this release does not certify the security of those other databases or perform a shared-host firewall rewrite.

## I. Remaining limitations

- Off-server backup replication and backup-failure alert destination: NOT CONFIGURED; disaster-recovery readiness remains blocked.
- Domain and publicly trusted certificate: deferred; temporary self-signed TLS requires browser acceptance/trust.
- Public/LAN/VPN reachability from arbitrary clients: NOT VERIFIED; workstation direct private-IP probes timed out.
- End-to-end invoice/inbox delivery: NOT VERIFIED, though SMTP TLS/auth passed.
- Sustained network/container-limited load testing, peak PostgreSQL connections and representative customer traffic: NOT VERIFIED.
- No partial payments, refunds, payment-method receipts, PDF invoices, self-service recovery or financial-adjustment UI.
- Ambiguous SMTP outcomes require manual provider-log reconciliation; exactly-once SMTP delivery is not promised.
- Host OS had pending updates and required a restart; shared infrastructure was not restarted or patched.

This release is running and has the verification above. These limitations prevent an unconditional production-ready claim.

## J. Deliverables

The corrected ZIP contains source, migrations, pinned dependencies, tests, Docker configuration, deployment examples, native backup script, benchmark runner/results, access instructions and documentation. It excludes real credentials, .env, database dumps, private keys, .git, virtualenvs, caches and build artifacts.
