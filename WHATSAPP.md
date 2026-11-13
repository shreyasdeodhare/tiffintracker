> Current deployed layered release: read [LAYERED-IMPLEMENTATION.md](LAYERED-IMPLEMENTATION.md) first. New guests need no login/email. Group publication is weekly only, Friday 14:00 IST (Saturday–Friday). Production migration/enablement completed with approval on 12 November 2026; older sections are historical; LAYERED-IMPLEMENTATION.md is authoritative where behavior differs.

# Automatic WhatsApp order tracking

Release: 12 November 2026. The user explicitly approved Baileys for the existing group, replacing the earlier official-only integration restriction. This is an unofficial linked-device connection, not Meta's supported business Groups API. No WhatsApp Web browser scraping or browser automation is used.

## One-time connection

1. Sign in as administrator and open `/admin/whatsapp` (WhatsApp tracking in the navigation).
2. Use an authorized WhatsApp account that already belongs to “Amanora 507B Office The Meal Habit”. On its phone open WhatsApp → Linked devices → Link a device and scan the protected QR.
3. Refresh the group list, select the exact existing group and click Start automatic tracking. IDs distinguish similarly named groups. Initial group selection is one-time; switching a running installation to a different group requires operator review.
4. New messages after that start time enter the PostgreSQL inbox automatically. There are no routine exports, uploads, manual order-entry steps or separate database containers.

LAN: https://192.168.1.101/admin/whatsapp . For this deployment session a private SSH forward is available at https://127.0.0.1:8443/admin/whatsapp on the deployment computer. To recreate it later, use the SSH command in ACCESS.md. The known self-signed certificate must be accepted/trusted until a domain certificate is installed. Nginx forwards `$http_host`, preserving the local tunnel's port so HTTPS CSRF origin checks continue working.

The account holder must perform phone linking; the server cannot authorize itself into a WhatsApp account. The deployment verifies pairing readiness and protected QR delivery, not an actual group receipt before this action. The temporary local forward lasts only while its helper and Cloudflare connection are running.

## What happens automatically

- The receiver admits durable events only from the configured group. Personal chats and other groups are not stored, classified, displayed or replied to. WhatsApp's own-device protocol traffic remains available for cryptographic synchronization.
- Sender JIDs are the identity, with observed phone/LID aliases. Display names are labels only. Two people with the same name never merge automatically.
- A previously unknown valid sender gets a separate tracked User/member record automatically. Its random password is not exposed; an administrator can later map it to an existing registered member or use assisted password setup. No initial roster entry is required for order counting.
- Auto-created records use synthetic non-deliverable email addresses with billing email disabled. Mapping to a real registered member reuses that member's email configuration and preserves eligible WhatsApp order history. Finalized bills cannot be reassigned.
- Explicit self orders such as `2 tiffins`, `1 tiffin for lunch`, `mere liye do tiffin`, `मुझे दो टिफिन चाहिए` and supported spelling/numeral variants create orders at the current configured Decimal price.
- `Make it 3 instead` or `Cancel my lunch` changes the one uniquely matched active source order for that sender and business date. Quoting the exact source message resolves multiple-order targets. Actual source edits replace the quantity; available deletion/revocation events cancel the matched order when sender identity is trustworthy and the period is mutable.
- Every decision, change and cancellation is linked to the source ID/content/timestamp and audit history. Transport event keys and a unique database constraint prevent duplicate processing. Replayed deliveries increment a duplicate counter without creating another order.
- Existing billing, payment, historical price and cancellation rules apply. Paid or issued/sending/unknown-invoice weeks remain frozen. Late changes are visible for reconciliation rather than silently altering accounting.

The sender's business date is calculated from its message timestamp using APP_TIMEZONE. Explicit tomorrow orders remain unconfirmed while future dates are disallowed by the existing application. Ambiguous `kal` is not guessed. Tracking starts when the group is enabled; this release does not automatically reconstruct old chat history or invent historical monthly totals.

## Ambiguity and supported scope

Automation requires sufficient information. `Lunch?`, conditional requests, unknown wording, similar-name references, conflicting identities, payment notices, oversized bodies, media-only/voice orders and uncertain modification targets go to review. Greetings and clear holiday announcements are ignored. A message containing the word lunch is not automatically an order. Short bare quantities without a clear order context are not guessed.

Text captions can be classified; voice transcription, image OCR and media downloads are not implemented. The source message ID and sender remain available so an administrator can reconcile the original in WhatsApp. Deterministic rules cover tested English/Hindi/Hinglish patterns, not unrestricted natural-language understanding. No accuracy claim is made against this group's actual wording until real messages are verified.

Administrators can approve/correct an uncertain order, reject it, select an exact change/cancellation target, or explicitly allocate a multi-person message across different members. Split requests are committed atomically. A split is never inferred from ambiguous display names.

## Counts, reports and integration

The WhatsApp dashboard reports each member's daily, weekly and monthly order-record count, total tiffin quantity and Decimal amount. Group totals are the sums of those same rows. Select All, WhatsApp or Manual sources for reconciliation. Cancelled orders are excluded. One order with quantity three means one order record and three tiffins.

Members can view only their own `/my/counts`; the original dashboard and billing remain available. Administrators can export daily/weekly/monthly XLSX counts, view original messages, inspect decisions and audit logs, link stable identities to registered members and retry failed processing. The existing order-detail export includes provenance in its Source column.

## Runtime and recovery

One Docker container runs Gunicorn, a Node Baileys receiver and one Python queue worker under a small process supervisor. PostgreSQL, Nginx and cron stay on the host. The 1.5 CPU and 512 MiB limits, non-root user, read-only root, dropped capabilities, no-new-privileges, PID limit and loopback-only `127.0.0.1:8087:8000` binding remain. The connector's authenticated HTTP interface binds only to 127.0.0.1:8090 inside the container and is not published.

The receiver durably commits events before the worker acts. Inbox consumption and changes share transaction boundaries and member/group locks. A failed business interpretation remains a review decision; an unexpected processing failure rolls back partial user/order/audit writes and retains an explicit error/retry record. No blind retry loop creates duplicate orders. The supervisor restarts each failed process independently with backoff, keeping the dashboard available if the WhatsApp connector fails. Optional group-list discovery is isolated from durable message handling and skipped for an already-selected group.

Connection status, processor heartbeat, backlog, duplicate deliveries and processing errors are visible to administrators. Reconnection uses bounded backoff. Logout needs another phone link; the Relink button clears the expired local credential state. Connectivity outages can cause delayed or unavailable delivery. Generic linked-device connectivity cannot promise recovery of every message during long disconnections; verify backlog and source messages after an outage.

`wa_auth` stores session/Signal keys encrypted with AES-256-GCM and random nonces. The separate random WA_SESSION_KEY and local HTTP WA_BRIDGE_TOKEN are in the protected server .env, never the image or archive. QR/session/message contents are not logged. Pairing endpoints require an administrator and return no-store responses. Protect database dumps and the encryption key; keep the key separately in secure recovery storage. Losing/rotating it requires a deliberate relink or controlled re-encryption. This is application-level encryption; the running application still needs the key to operate.

PostgreSQL migrations: 0004_whatsapp adds source/notification flags, connection state, identity aliases, inbox, source events, audits and encrypted session storage. It preserves existing users/orders. A restored production snapshot was upgraded and tested before the live migration. The migration intentionally refuses an automatic destructive downgrade of financial audit history.

Source messages and audit data are retained indefinitely by default for accounting traceability; there is no automatic redaction/purge workflow in this release. Only the selected group's live events are retained. Set an organizational retention policy before collecting long-term personal data. Existing local PostgreSQL-native backups include all new tables; off-server replication and failure alerts still need a configured destination.

## Verification commands

```bash
pip install -r requirements-dev.txt
python -m compileall -q app scripts tests
ruff check .
ruff format --check .
pytest -q
# Fresh disposable PostgreSQL database only:
TEST_POSTGRES_URL=postgresql+psycopg:///tiffintrack_wa_test pytest -q
node --test bridge/test/*.test.mjs
# Synthetic durable-inbox workload; leaves data in the disposable database:
PYTHONPATH=. TEST_POSTGRES_URL=postgresql+psycopg:///tiffintrack_wa_test python tests/benchmark_whatsapp.py
docker compose config --quiet
docker compose build
```

The npm package lock pins Baileys and dependencies; npm audit reported zero known vulnerabilities at verification time. Read WHATSAPP-VERIFICATION.md and whatsapp-benchmark-results.json for actual results and limits. Phone linking, real group-message receipts, real sender alias behavior and post-link resource usage require verification with the account holder. The implementation is deployed and ready for that setup; it is not claimed already connected before linking.

Latest operational correction: see GATEWAY-FIX.md.
