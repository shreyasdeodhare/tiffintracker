> Historical pre-layered verification record. Read LAYERED-IMPLEMENTATION.md for the current candidate, weekly-only publishing, guest identity model and pending production approval.

# WhatsApp automation release verification

12 November 2026 — support server deployment. The user explicitly approved an unofficial linked-device connector for the existing group. The earlier official-only/no-unofficial constraint was replaced for this implementation.

## Implemented and deployed

- Baileys 7.0.0-rc14 receiver, locked npm dependencies, Node runtime and one Python inbox worker inside the existing single application container. No new database container, Redis, Celery or separate connector container.
- Live selected-group message envelopes, stable WhatsApp message IDs, phone/LID identity aliases and automatic tracked-member creation without display-name matching or routine manual member entry.
- Conservative deterministic English/Hindi/Hinglish quantity/order parsing, changes, cancellations, source edits and available revocations. Questions, unclear assignments, conditions, payment notices, media-only orders and uncertain identity/targets do not silently create orders.
- PostgreSQL durable inbox, database delivery deduplication/counters, transactional processing, retry/error records, shared member/group locks, original source content/IDs/timestamps and per-order decision/change audit history.
- Shared Order model, stored historical Decimal price, existing billing/payment/issued-period rules. No alternative WhatsApp billing formula. Manual/WhatsApp provenance is shown in exports and count filters.
- Daily/weekly/monthly per-member order-record counts, tiffin quantities and amounts; group totals sum the same member rows. Cancelled orders are excluded.
- Administrator-only protected QR and connection/group setup, worker/backlog/error/duplicate status, message inspection, review/reject/correct controls, exact change targets, explicit atomic multi-person allocations, optional verified mapping to existing accounts and XLSX reports. Members access only their own counts.
- AES-256-GCM encrypted session/Signal state in PostgreSQL with random nonces and a separate protected environment key. The receiver's authenticated control API is inside-container loopback only. Session/QR/message data are not logged.

## Deployment and data safety

Existing source/environment and PostgreSQL dump snapshots were taken under /var/backups/tiffintrack/releases. A production database snapshot was restored to a disposable database, upgraded and tested before the live migration. Production revision is 0004_whatsapp; existing user/order counts (1 user, 0 orders at migration time) were preserved. No real WhatsApp order or payment was invented to make the deployment appear connected.

The migration adds encrypted session, connection, identity, inbox, message and audit tables plus source/email-eligibility flags. Prior orders remain manual and registered members retain billing-email behavior. Auto-created tracked records have synthetic non-deliverable emails and email billing disabled until mapped appropriately.

Image: tiffintrack:whatsapp-20261112. One container remains bound to 127.0.0.1:8087:8000, non-root app user, read-only root, dropped capabilities, no-new-privileges, PID limit, restart policy, bounded logs/tmpfs, 1.5 CPUs and 512 MiB maximum memory. Receiver, worker and Gunicorn are supervised together. Initial observed combined memory was 268.9 MiB / 512 MiB and CPU 0.44%; a later sample was 274.9 MiB and 0.48%; these are observations while waiting for linking, not post-link peak load guarantees. Health was healthy and restart count zero after deployment.

Docker build, runtime Baileys import, Compose validation, database-backed health over HTTPS and Nginx syntax/reload passed. The first dependency build attempt failed; a subsequent validated build succeeded. A pre-migration count check initially used new-model columns against the old schema; it was corrected to schema-independent SQL before migration, without changing production data.

The workstation's private SSH HTTPS forward was verified at https://127.0.0.1:8443/health with HTTP 200. A browser-reported CSRF host/referrer mismatch exposed Nginx stripping the tunnel port; forwarding was corrected from $host to $http_host. A same-origin login-form submission then returned HTTP 200 while an untrusted-origin submission still returned HTTP 400. CSRF was not disabled.

## Executed verification

- 66 Python tests passed on the server, with PostgreSQL integration enabled: portable route/security/accounting tests plus real migrations, constraints/rollback, concurrent form writes, mail transaction behavior and the WhatsApp durable queue.
- 7 Node tests passed: replay event identity, revision/deletion keys, non-group rejection, Baileys editedMessage update semantics, credential encryption round trip, random nonce behavior, tamper detection and wrong-key rejection.
- Real WhatsApp PostgreSQL concurrency test: eight concurrent deliveries of one source event produced one inbox/source/order result; accidental concurrent worker calls serialized safely. A simulated processing crash rolled back partial source/order/audit work, preserving an error record for explicit retry.
- WhatsApp cases include same-name different identities, automatic accounts, aliases, multiple orders/day, explicit quote targets, edits/deletes, cross-member tampering, ambiguous/holiday/payment/chat handling, paid-week locks, future dates, late revisions, timezone dates, historical prices, reviewed multi-person allocation, authorization and workbook reopening.
- Compileall, Ruff lint and Ruff formatting passed. A legacy Query.get warning in the new split test was replaced with Session.get.
- Live HTTPS verification used temporary diagnostic identities and cleaned them up without altering the real administrator's password: admin connection page, processor heartbeat, status JSON and protected PNG QR passed; a member was forbidden from QR access and could view their own count page.
- Office365 SMTP authentication had already passed in the previous deployment. This release does not send test group messages or billing emails. Auto-created non-email accounts are excluded from billing email.

## Actual synthetic processing benchmark

A host-side PostgreSQL workload delivered 200 transport events representing 100 unique decisions for 20 stable sender identities: 60 original order creations, 20 edits and 20 cancellations. Independently expected final state: 40 active orders, 100 tiffins, exactly 5 tiffins per member, 100 duplicate deliveries ignored and zero errors. Assertions passed.

Measured worker duration: 3.288 seconds; 30.41 events/s. Per-event latency P50 14.51 ms, P95 106.95 ms, P99 115.4 ms. Peak benchmark process RSS: 115904 KiB.

These are synthetic durable-inbox/worker measurements on the host. They exclude real WhatsApp transport, TLS/network request latency and the container's enforced resource caps. They do not establish unrestricted language accuracy, sustained 1,000-user traffic performance or peak memory after linking. Raw data: whatsapp-benchmark-results.json.

## Populated recovery

The populated isolated WhatsApp dataset was dumped and restored to a second isolated database. Counts matched: 21 users, 60 orders, 100 inbox rows, 100 source events, 200 audits, 20 participants and 20 identity aliases. Both copies were at 0004_whatsapp, produced the independently expected 100 active tiffins and passed application/database health. A native backup of the live expanded schema also succeeded.

## Pending account-holder setup and limits

At deployment verification the receiver was awaiting_phone_link and no real group had been selected. The protected pairing QR was available; the processor was running. The account holder must scan the QR using an authorized group-member phone account and select the exact existing group before live order collection starts. A real incoming group message, real sender identity/alias behavior, modification/deletion delivery and persistence across a linked reconnect remain NOT VERIFIED until that happens.

Tracking starts at the enable time. Old group history is not automatically reconstructed; pre-start history is excluded to avoid unintended billing. Unknown language, voice/media-only orders, conflicting identities, multi-person name ambiguity, out-of-order changes and finalized bills require reconciliation rather than invented quantities. No OCR/transcription or general AI language model is configured. Complete historical counts or flawless interpretation are not claimed.

Baileys is unofficial and its API/protocol can change. Linked-device logout/connection interruptions may require relinking and reconciliation; complete recovery of every missed message is not guaranteed. Only selected-group live events are persisted. Source/audit retention is indefinite by default, with no automatic purge/redaction UI. There is no group-message auto-reply or bulk messaging feature.

Domain/trusted TLS remain deferred; the IP service uses a temporary self-signed certificate. Off-server backup destination, replication and alerting remain unconfigured. The existing server's unrelated workloads, global PostgreSQL listener/firewall policy and pending OS updates are not certified by this application release.

## Significant files

app/whatsapp_models.py, whatsapp_parser.py, whatsapp_service.py and whatsapp_routes.py separate persistence, interpretation, business decisions and HTTP controls. app/static/whatsapp.js and WhatsApp templates implement setup/review/count views. bridge/index.mjs is the single linked-device receiver; envelope.mjs isolates event identity; session-crypto.mjs isolates authenticated encryption. scripts/run_container.py and whatsapp_worker.py manage the single-container runtime. Migration 0004_whatsapp.py is additive. Existing auth/billing/order/export modules are reused with source, audit and email-eligibility integration. Tests, pinned package lock, updated Dockerfile, Nginx host-port forwarding, README and WHATSAPP.md accompany the release.

The new source archive excludes real credentials, QR/session material, .env, database dumps, private keys, .git, node_modules, caches, virtualenvs and build artifacts. The archive and report describe a deployed implementation awaiting the required account link, not an already verified live WhatsApp connection.
