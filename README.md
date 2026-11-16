# TiffinTrack

> 16 November 2026: full/half tiffin pricing and itemized WhatsApp orders are deployed as `tiffintrack:portions-20261116`. Full = INR 80, half = INR 60. See [portion rollout](docs/FULL-HALF-ROLLOUT-2026-11-16.md). No schema migration or resource/port change was made. Older acceptance documents below remain historical records.

[Complete file reference with purposes and full source listings](TiffinTrack%20Complete%20File%20Reference.pdf)

> Final acceptance: the stabilization image remains running; this audit introduced no runtime change or restart. See [FINAL-ACCEPTANCE.md](FINAL-ACCEPTANCE.md) and [ACCEPTANCE-MATRIX.csv](ACCEPTANCE-MATRIX.csv) for evidence and pending genuine business checks. Active guidance: [ACCESS.md](ACCESS.md), [RECOVERY.md](RECOVERY.md), [DOMAIN-READINESS.md](DOMAIN-READINESS.md). Dated reports remain historical evidence.

Tiffin order tracking for a selected existing WhatsApp group, with manual corrections, participant reports and Excel exports. The deployed Compose override selects `tiffintrack:portions-20261116`; the base Compose configuration is preserved. It retains the layered architecture and schema `0005_participants`; no migration was introduced.

## Active workflow

The authorized linked WhatsApp account receives messages from the selected group through the unofficial Baileys connector. Stable sender aliases identify participants independently of display names. Guests need no login or email. Clear requests create orders; edits/cancellations target the source order. Ambiguous requests remain under administrator review. Distinct original requests may create multiple same-day orders; repeated source messages and browser submission keys cannot create additional orders.

Administrators use `/admin/orders` for daily, Saturday–Friday weekly and monthly reports, `/admin/participants/<id>` for individual history, and `/admin/whatsapp` for connection status and message review. Reports and Excel distinguish active order records from tiffin quantity, use stored Decimal prices, and exclude cancellations. Independent SQL reconciliation checks participant and group totals. Pagination changes displayed rows, never totals.

Email collection and automated email delivery are outside the active workflow. The old account-creation HTTP endpoint returns 410. Historical authentication accounts, payments, email history and compatibility code remain intact. Install the maintenance cron template without any legacy billing entry; do not replace protected server secrets.

## Weekly group publication

Only Friday at **14:00 Asia/Kolkata (IST)**: previous Saturday through Friday, captured at the actual snapshot time. No daily publication. Unresolved reviews, pending/error intake, incomplete coverage or a delayed capture produce an explicitly provisional report. A reconciliation mismatch blocks publication. Each participant line includes stable ID, name, order records, quantity and amount; group totals and cancellations are included.

The database stores an immutable snapshot and bounded message parts in a durable outbox. The connector serializes publication, records acknowledgements and avoids replaying acknowledged parts. An uncertain delivery requires review; it is not blindly resent. Orders received after capture do not rewrite the saved report. The unofficial connector can disconnect and cannot guarantee complete WhatsApp history.

## Architecture and runtime

`app/presentation` and root HTTP modules handle requests and authorization; `app/application` owns use cases; `app/domain` owns pure rules; `app/infrastructure` owns persistence, transport and workbooks. Root compatibility modules preserve existing imports. Keep this structure; no new services are required.

One Docker Compose application container supervises Gunicorn, the inbox worker and Node connector. PostgreSQL, Nginx and cron are host services. Resource limits remain 512 MiB memory, 1.5 CPU and 128 PIDs. The container has a read-only root, temporary `/tmp`, dropped capabilities, init and bounded logs. Application port 8000 is bound only to host `127.0.0.1:8087`; internal bridge port 8090 is not published.

Current deployed release is `tiffintrack:portions-20261116`, verified on 16 November 2026. Private access: https://192.168.1.101/admin/orders on the server LAN. Public domain access is deferred. The temporary server certificate requires trust and renewal. Do not assume a localhost SSH tunnel remains running.

## Configuration, maintenance and verification

Use `.env.example` as the non-secret configuration reference. Keep the real `.env`, session key, bridge token and database credentials only in protected server configuration. Preserve the existing encrypted linked session. Never package environment files, database dumps or credentials.

`deploy/tiffintrack.cron.example` schedules backups and login-attempt pruning. `deploy/tiffintrack-summary.cron.example` schedules the weekly summary. The audit verified Asia/Kolkata, publishing enabled and the installed Friday schedule; the retired email job is absent. Future configuration changes require review.

Run `python -m compileall -q app scripts tests`, `pytest -q -p no:cacheprovider`, `ruff check .`, `ruff format --check .` and `npm test` from `bridge`. PostgreSQL integration and migration checks require isolated test databases using `TEST_POSTGRES_URL` and `TEST_MIGRATION_URL`. Never run fixtures against production or send synthetic financial summaries to the live group. The supplied Dockerfile built the running image; the final audit found no demonstrated runtime defect requiring another build or restart. Future deployment changes need explicit approval.

The final stabilization report records passing tests, approved deployment and live verification. A genuine live order, its participant attribution and subsequent totals still need to be observed before full production business acceptance can be claimed. `LAYERED-IMPLEMENTATION.md` and other dated documents are historical evidence; older email, architecture and deployment instructions are superseded by this README and the stabilization report.
