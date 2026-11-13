# TiffinTrack stabilization verification — 12 November 2026

Status: deployed with explicit user approval at 22:00 IST on 12 November 2026 and live checks passed at 22:01 IST. No schema migration. The current production image is `tiffintrack:stabilization-20261112` (image SHA256 `7881a32451f8ae01e5eec06620ba156d52be9251c058b0a05dbd292c99205923`).

## P0 — corrected gaps

| Files | Exact correction | Verification | Production action |
|---|---|---|---|
| `app/application/order_decisions.py`, `app/application/whatsapp_admin.py`, `app/infrastructure/inbox.py`, `app/templates/whatsapp_detail.html` | Only review-state messages can be decided. Original source group/message IDs are checked against creation audits before a normal or split creation. An ignored duplicate cannot be approved; even resetting it to review or cancelling its original order cannot recreate the source. Intentional distinct same-day originals remain valid. | New HTTP replay regression; both split paths; source replay after cancellation; existing intake/replay/split tests. | Deploy candidate; no migration. |
| `app/domain/summaries.py` | Individual record count is distinct from quantity. Stable ID/name, monetary amount and group totals remain included. Snapshot time is labelled accurately. Review backlog, pending/error intake, coverage and delayed capture have explicit provisional reasons. | Bounded 180-participant splitting; immutable snapshot/idempotency; provisional/late cutoff tests; dashboard/Excel/summary agreement. | Applies to newly prepared summaries; saved historical parts remain immutable. |
| `app/routes.py`, `app/templates/dashboard.html`, `app/templates/base.html`, `.env.example`, `deploy/tiffintrack.cron.example` | Retired email-account collection form/navigation; administrator creation endpoint returns 410, unauthorized member remains 403. No SMTP setup in example or billing cron in template. Historical account/payment/email data and compatibility modules retained. | Retired endpoint and no-new-account assertions; historical export remains accessible; existing auth tests pass. | Remove only `weekly_billing` from active maintenance cron; retain backup/pruning and separate weekly group cron. |

Order-record count, quantities, historical Decimal prices, cancellation exclusion, stable guest identity and shared daily/weekly/monthly totals were already implemented and remain unchanged. No new architecture, schema, service, AI dependency or infrastructure was added.

## P1 — functional evidence

Full isolated PostgreSQL suite: **96 passed, no skips, 27.28 seconds**. Ruff: all checks passed; 86 Python files already formatted. Node suite: **24 passed, 0 failed, 0 skipped**. Separate PostgreSQL concurrent outbox claim and mocked transport test passed; no WhatsApp connection was used. The supplied full Dockerfile built successfully as non-root `app`, Node 24.22.3, with Baileys import verification.

The suite covers clear intake, stable aliases and identical display names, guest creation without login Users, source duplication, distinct same-day orders, modifications, edits, deletes/cancellations, exact-target ownership, ambiguity, outside-group/history rejection, manual submission idempotency, historical prices, reconciliation mismatch blocking, pagination, admin/member authorization, snapshot immutability, Friday-only cutoff, splitting, acknowledgement, uncertain delivery and no automatic replay of sent/unknown parts.

A new cross-surface test reopens the generated XLSX and compares all Daily/Weekly/Monthly sheets with dashboard totals. The fixture has 3 active records, 6 tiffins, INR 461.20, plus an excluded cancelled record, with different stored prices and two separate stable participants sharing a name. Independent SQL reconciliation and the saved weekly publication agree. This is synthetic evidence, not a claim of real-group financial acceptance.

Evidence: `TiffinTrack-Stabilization-Tests-and-Build.log` alongside the downloadable report.

## P2 — operational cleanup and release scope

`README.md` now documents the active layers, participant workflow, correct report semantics, private access, container limits, weekly-only publishing, test commands and limitations. Older dated reports are historical evidence and their email/deployment instructions are superseded. `docker-compose.yml` names the tested candidate; limits/security settings remain unchanged. `CODE-MANIFEST.sha256` covers source and deployment code; the complete source ZIP excludes credentials, `.env`, databases, session data, caches and installed dependencies.

Read-only production checks: container healthy, zero restarts; schema unchanged at the existing participant release. Host timezone Asia/Kolkata. Separate summary cron is Friday 14:00 IST, previous Saturday through Friday. The only discovered legacy billing cron location is `/etc/cron.d/tiffintrack`; its Monday email entry was removed during the approved deployment. Daily backup and login-attempt pruning must remain. Live database currently has zero orders, one review event and no publications. No fake orders, live financial test messages, email or session reset was performed. The approved single-container recreation completed; no business data or schema was changed.

## Approved deployment completed

1. Preserve current code/Compose and take a protected native PostgreSQL backup.
2. Install the verified source/config template changes without replacing protected `.env` or the linked session; no migration command is required.
3. Remove the retired email billing line only, preserving maintenance and Friday group-summary schedules.
4. Recreate the single application service using the built stabilization image, preserving 512 MiB memory, 1.5 CPU, 128 PIDs and network bindings.
5. Verify health, receiver/worker heartbeats, schema, protected HTTPS access/CSRF and read-only reports/Excel. Do not send synthetic summaries to the real group. The old layered image remains available for code rollback because the schema is unchanged.

Explicit approval was obtained for this stabilization release as required by section 3, item 10 of the supplied Critical Stabilization prompt: “Do not modify production data or restart production without explicit approval.” Previous deployment approval applied to the earlier layered release.

## Remaining acceptance limits

A genuine live order has not yet been observed. Its correct participant attribution, subsequent edit/cancellation and live dashboard totals must be confirmed through normal group use before full production business acceptance. First scheduled real Friday summary delivery has not yet been observed. Ambiguous messages require review, and gaps/disconnections produce provisional reports. The unofficial linked-device integration cannot guarantee upstream availability or historical completeness. Real browser password login was not retested in this release. Public hostname deployment remains deferred; private LAN link is https://192.168.1.101/admin/orders and requires network access and trust of the temporary certificate.

## Final live verification

The deployed container is healthy, restart count 0, memory 536870912 bytes, 1.5 CPU, PID limit 128 and read-only root enabled. PostgreSQL remains at `0005_participants`. A protected code/Compose and native database backup was created at `/var/backups/tiffintrack/stabilization-20261112-220042` before replacement. Friday summary cron was byte-for-byte preserved; the retired billing command is absent from installed system/user cron searches, while daily backup and pruning remain installed.

Receiver heartbeat age 1.80 seconds; worker 0.93 seconds; WhatsApp status connected; weekly publishing flag enabled. Orders 0, reviews 1, pending/error inbox 0, publications 0. Daily reconciliation matches zero records/quantity/amount and is correctly provisional. Privileged read-only administrator contexts rendered order reports, operations, WhatsApp, individual counts, participant/event detail and reopened both Excel endpoints successfully; this does not constitute browser password login verification. Actual HTTPS login returned 200, protected endpoints 302, and an un-tokened POST returned CSRF 400. No live test financial publication was sent.

Production is running the stabilization release. The remaining genuine-order and first scheduled real weekly-delivery acceptance limits above still apply. Deployment and live logs are supplied alongside this report.
