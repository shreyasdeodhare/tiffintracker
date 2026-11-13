# TiffinTrack layered implementation and migration review

## 1. Scope and current status

This release implements the latest principal-engineer prompt and the user's later corrections: **weekly WhatsApp summaries only, Friday at 14:00 IST, covering the previous Saturday through Friday's snapshot cutoff**. There is no daily publication job. Unresolved reviews and incomplete coverage make the weekly report provisional.

This is a refactor and extension of the existing Flask/PostgreSQL/Baileys application. It preserves one application container, the existing host PostgreSQL/Nginx/cron, resource limits, Cloudflare SSH route and authorized linked-device session. No public hostname/network change, new email collection, new email billing feature, payment collection, AI service or unsolicited reply is introduced.

**Production migration and deployment completed on 12 November 2026 after explicit user approval.** The running image is `tiffintrack:layered-20261112`, schema `0005_participants`, healthy with zero container restarts at verification. The baseline before deployment was `tiffintrack:audit-fix-20261112` / `0004_whatsapp`.

The attached implementation prompt, section 15, says: “Phase 2: Design participant identity and migration strategy. Obtain approval before destructive or data-affecting changes.” The tested build and migration review were completed first; the user then explicitly answered “Approve migration and deployment” before any production migration or restart. Source: [attached prompt](<C:/Users/admin/.codex/attachments/47aa007f-5751-49e2-a9b3-8111ad7808f1/Pasted text.txt>).

## 2. Baseline and confirmed design gaps

The original implementation mixed SQL, transaction rules and HTTP handling; automatically created a synthetic-email login User for each WhatsApp sender; aggregated primarily by member account; had no durable outbound group-summary workflow; and exposed mostly a legacy weekly member/payment view. These were implementation/design limitations, not evidence of observed production financial corruption.

At the read-only baseline on 12 November 2026, production was healthy at `0004_whatsapp`; the previously recorded business dataset had two users, one tracked participant/review message and no confirmed orders/payments/invoices. The authorized connection and session records existed. Fresh backup/restore rehearsal is a stronger source than assuming this historical count remains unchanged forever.

Compatibility requirements: preserve every historical account/hash/flag, order date/quantity/price/status/source, payment, email log, participant mapping, WhatsApp source/audit link and encrypted authentication record. Keep import paths used by retained tools stable where practical. Do not assume old application code can write the new schema.

## 3. Layered package structure and dependency rules

```text
Presentation
  app/presentation/report_routes.py, security.py
  app/auth.py, routes.py, whatsapp_routes.py, operations.py (thin legacy HTTP adapters)
  app/templates/, static/
       | invokes use cases; no SQL queries in handlers
Application
  app/application/accounts.py, participants.py, orders.py, order_decisions.py
  ingestion.py, reporting.py, exports.py, publications.py, monitoring.py
  finance.py, legacy_dashboard.py
       | explicit repository dependencies; transaction orchestration
Domain
  app/domain/orders.py, parsing.py, summaries.py
       | validation, date periods, Decimal arithmetic, deterministic parsing/formatting
Infrastructure
  app/infrastructure/models.py, whatsapp_models.py
  participants.py, core.py, accounts.py, inbox.py, reporting.py, finance.py
  publications.py, whatsapp_admin.py, bridge_client.py, workbooks.py
  legacy_reconciliation.py
Background / transport
  scripts/whatsapp_worker.py, weekly_summary.py, run_container.py
  bridge/index.mjs, publications.mjs and existing receiver helpers
Testing
  tests/, bridge/test/
```

Legacy root model/service/report/parser modules are compatibility facades, not second implementations. The existing `WAParticipant` class/table name is retained to avoid unnecessary table renames; it now represents the canonical participant for manual and WhatsApp orders.

Architecture tests enforce a framework/persistence-free domain, no direct query/execute/scalars operations in HTTP handlers or application use cases, and no application/presentation imports from infrastructure. Persistence objects are deliberately reused across application use cases rather than introducing a second ORM or an unnecessary mapping framework. Flask configuration/clock and transaction composition remain application concerns.

## 4. Database and identity migration

Forward migration: `0005_participants`, file `migrations/versions/0005_participants_and_summaries.py`.

- Make participant-to-User mapping optional. Add participant activity, creation/update timestamps and optional verified contact storage.
- Add required `orders.participant_id`; backfill from existing mappings, creating canonical participant rows only for previously unmapped historical order owners.
- Allow `orders.member_id` to be null for guests while retaining historical mappings. Add meal/category and update timestamp. Historical meals remain `unspecified`; do not invent lunch/dinner history.
- Add participant/date indexes and category validation.
- Add inbox processing-completion timestamp. Existing unknown processing-completion times remain unknown, not fabricated.
- Add participant and manual-order audit tables. Backfilled records are explicitly labelled `migration_preserved`, with no invented historical actor.
- Add reconciliation runs, weekly publications, immutable publication parts and publication audits.
- Enforce one publication per group/kind/period/source, unique part position/message ID, valid states and weekly-only kind.
- PostgreSQL triggers prohibit changing saved publication snapshot identity/contents or part text/message identity while permitting delivery-state updates.

No existing account is deleted or silently deactivated by this migration. Historical synthetic users remain for compatibility; **new WhatsApp guests do not create Users, passwords or synthetic emails**. A guest may optionally be mapped to a verified existing member later. Mapping conflicts are rejected rather than merged by name. Mapping checks finalized legacy financial periods, updates eligible order compatibility references and appends audit records. Identity aliases/name changes are audited; display name never becomes an identity key.

The migration is forward-only. Guest orders with null member IDs and mandatory participant IDs make the old writer incompatible. Do not roll back only the image after migration and expect old order writes to work.

## 5. Order-processing behavior and accuracy

Receive -> selected-group filter -> normalized envelope -> unique durable inbox -> stable participant/alias resolution -> conservative deterministic intent -> date/quantity/ownership/finalization checks -> order mutation plus source/audit links -> inbox processed timestamp -> shared reporting.

The existing receiver/session compatibility fixes are preserved. Questions, conditions, uncertain dates/people, unsupported media and unclear multi-order targets go to review. Supported explicit English/Hindi/Hinglish quantity/meal forms remain covered by regression tests. No unrestricted LLM posts financial transactions.

Order records retain participant, optional member, business date, category, quantity, historical unit price, status, source, timestamps and cancellation history. Amount is Decimal quantity × historical price. Auto edits/deletes require eligible unique source targets belonging to the resolved participant. Existing cancelled orders are not silently reactivated. Multiple intentional daily orders remain possible.

Manual guest entry requires no email/login and uses a UUID replay key. Reusing a key with conflicting details is rejected. Legacy member entry remains for compatibility. Administrator review can select verified participants, allocate explicit multi-person requests to guests or members, and preserve all source links. It does not guess identities from names.

Existing date window remains today or the preceding 14 days; future orders require review rather than automatic confirmation. There is no new holiday/cutoff-order-rejection policy. Friday's publication cutoff is a reporting boundary, not an implicit order cancellation rule.

## 6. Shared aggregation and independent reconciliation

The canonical projection is `ReportingRepository.totals`: active orders grouped by stable participant, with separate record count, tiffin quantity and Decimal amount. Legacy member/payment totals derive from this canonical projection. Count exports and participant dashboards use it too.

Independent verification queries raw Orders grouped by participant **and source**, without the presentation join, and compares per-participant records/quantities/amounts, source sums and group sums. It additionally checks member/participant compatibility references and missing/duplicate WhatsApp creation-audit references. This is not calling the same aggregate twice. Deliberately corrupted projection tests fail reconciliation and block publication.

Historical member/payment reconciliation remains available and checks paid amounts. No discrepancy is automatically repaired. A mismatch is persisted as an auditable reconciliation result when publication preparation encounters it; no outbound publication is created.

Reports use a dedicated PostgreSQL **read-only REPEATABLE READ** transaction. All sheets/period projections for a request see one committed snapshot. “Reconciled” means the saved financial arithmetic/reference checks passed, not that upstream WhatsApp delivery is proven complete.

## 7. Dashboards and exports

Administrator landing page: `/admin/orders`. It includes overall participant/queue/tracking indicators; daily, Saturday–Friday weekly and calendar-month totals; individual/group quantities and amounts; cancelled/review/unresolved counts; monthly daily trend and elapsed-day average; filters for date/participant/source/status; manual guest entry; and publication status.

Tables show 50 participant/history rows per page. Totals and reconciliation always cover the full dataset. Tests explicitly verify identical totals across pages. Exports retain all matching rows. Participant detail shows aliases, recent order/source history, identity/mapping audits and manual/WhatsApp audit links. Bounded history limits are labelled; use report filters/exports for broader detail.

Member endpoints remain scoped to the authenticated member, regardless of supplied IDs. All participant/source/publication administration is admin-only and mutations remain CSRF-protected.

The count workbook has Daily/Weekly/Monthly sheets with participant IDs, record counts, quantities and formatted numeric amounts. Its week is Saturday–Friday. The retained member/payment workbook and `/legacy-dashboard` preserve Monday–Sunday historical payment semantics. They are explicitly legacy views, not the Friday group-summary period.

Cancelled history remains visible as history and excluded from active totals. Formula/XML-control protection and Unicode handling are preserved. Money is rendered directly from Decimal rather than using floating-point formatting for HTML.

## 8. Weekly group summary preparation

`python -m scripts.weekly_summary` prepares the selected group's **weekly-only** report. Cron example: `deploy/tiffintrack-summary.cron.example`, Friday `0 14 * * 5` on the verified Asia/Kolkata host, with `flock`.

The period starts the previous Saturday and ends Friday. Financial snapshot capture happens when the job actually runs. The saved text records its actual capture timestamp and scheduled target; a materially late capture is marked provisional rather than pretending to reconstruct an earlier database state. This is best-effort scheduling, not an exact-time delivery guarantee. Friday orders arriving after the stored snapshot remain in the database/dashboard and do not silently rewrite the already posted message.

Preparation acquires the existing connection/configuration serialization lock, captures a consistent snapshot, verifies independent reconciliation, records review/backlog/coverage exceptions, stores a reconciliation result and SHA-256 fingerprint, then commits the publication and all deterministic bounded text parts together. Similar names include participant IDs. Sorting and part numbering prevent omitted/ambiguous entries. Long reports split safely with group totals and part counts.

No daily job exists. No immediate test publication is sent during migration testing. `SUMMARY_PUBLISHING_ENABLED=false` is the safe distributable default; live enablement belongs to the approved server release.

## 9. Durable transport, idempotency and retry

`bridge/publications.mjs` consumes only prepared weekly parts for the currently selected group, using the existing connected socket. It does not calculate bills or receive arbitrary public send requests. Port 8090 stays internal.

Rows are claimed with PostgreSQL locks/skip-locked; sending state and an audit are committed before invoking transport. Stable message IDs and part order are retained. The receiver recognizes its own publication IDs and excludes them from order intake.

States: pending, sending, sent, failed and unknown. No successful parent state is recorded until every part has transport acknowledgement. Sent means acknowledged submission, not recipient read/delivery proof. A 30-second deadline, missing/unexpected transport identity or ambiguous transport exception produces unknown. Restart recovery marks interrupted sending claims unknown. No automatic unknown retry occurs.

An administrator must check external evidence and provide a reconciliation note before allowing a retry or recording already-delivered parts. Actively sending claims cannot be reset through the UI. Already-sent parts are not resent routinely. Partial reports retain part numbering and an incomplete parent state until resolved. The normal architecture has one receiver; do not start another live receiver against the same session.

## 10. Monitoring and auditability

Operations distinguishes receiver/worker unavailability, pending/error backlog, review count and component readiness from proven delivery. New fields include actual last-processed time, persisted reconciliation history and weekly publication states. Report coverage becomes provisional for stale/disconnected processing components as well as pre-tracking periods, unresolved reviews and pending/error inbox rows.

Participant, source-event, manual-order and publication audits are append-only through supported application operations. Summary contents have additional PostgreSQL mutation guards. Privileged database access is not represented as tamper-proof storage. Logs identify IDs/error types/SQLSTATE without printing session keys, QR values or private message contents.

No new external alert destination, offsite disaster-recovery service, public hostname or administrator MFA was configured in this scope.

## 11. Verification and measured performance

Final full isolated run: **94 Python tests passed on real PostgreSQL in 30.24 seconds**, with no skips. **24 Node tests passed**, including an explicit test stage using the pinned image runtime (Node 24.22.3). The separate concurrent PostgreSQL outbox/mock-transport test passed. Local lint/format checks passed for 86 Python files. In-memory and bytecode compilation checks passed.

Real PostgreSQL checks cover forward migration from a populated old schema, preservation of historical prices/statuses/users/auth records, duplicate manual intake, concurrent worker processing, concurrent publication preparation, and snapshot mutation guards. A separate Node/PostgreSQL test uses two publishers with a **mock** socket and verifies one claim/submission per part. It imports no live WhatsApp client.

An initial sender integration failure exposed PostgreSQL parameter inference in a delivery-state expression; explicit casts fixed it, and the real PostgreSQL mock-transport test subsequently passed. A candidate build initially tried to write a root-directory bytecode cache as a non-root user; build validation was changed to in-memory compilation without weakening runtime permissions.

The benchmark used 1,000 registered fixture users and approximately 14,000 orders, five concurrent requests and real PostgreSQL on the host. Initial full administrator-report P95 was 2,906.79 ms. Replacing repeated ORM scans with independent SQL and paginating presentation rows reduced measured P95 to **400.79 ms in the latest repeat** (about 86% lower; an earlier optimized repeat measured 227.26 ms), with no sampled errors. Other optimized P95 samples: login 127.61 ms, member dashboard 126.35 ms, order creation 69.40 ms, aggregation 49.09 ms, Excel 733.00 ms. These are finite host-side samples, excluding public network/Gunicorn and container cap enforcement; they are not a capacity/SLA guarantee.

Query plans are included in verification evidence. The optimized path does not cache stale totals or omit reconciliation to gain speed. The raw benchmark JSON is included. Other operations did not all improve; finite samples on a shared host vary. No global optimality claim is made. A separate synthetic inbox workload processed 100 unique events/200 deliveries from 20 guests in 1.72 seconds (58.12 events/second), ending with 40 active order records and 100 tiffins, 100 duplicate deliveries ignored and zero errors. No live WhatsApp transport was used for this benchmark.

## 12. Restore and migration rehearsal

A native production backup was restored into a separate shadow database. The candidate migration was applied only there. Comparisons verified unchanged Users, historical Order business fields, Payments, EmailLogs, existing participant mappings and encrypted auth values. Matching session-key authenticated decryption succeeded without connecting the restored session to WhatsApp. A separate restored database also completed the migration after switching to the actual application database role, verifying its DDL permissions rather than relying solely on PostgreSQL superuser permissions.

A populated fixture backup/restore with 1,000 users and 14,016 orders matched counts and full financial reconciliation results. This proves the exercised local restore, not a total-server-loss/offsite recovery objective. Protected backups contain secrets/private data and are not included in the downloadable ZIP.

## 13. Reviewed production deployment procedure

1. Confirm user approval, final tests/build success, current image/revision and Asia/Kolkata host timezone. Ensure no conflicting release is in progress.
2. Retain previous source/Compose and protected `.env`/session key. Take a fresh native backup and validate it. Preserve the rehearsed shadow backup too.
3. Plan a brief application maintenance restart. Stop the existing container gracefully so old code cannot insert orders without canonical participant IDs during migration.
4. Apply `0005_participants` once with the reviewed release code. Capture/compare non-secret business counts and financial totals; verify required participant references and existing session material.
5. Install the reviewed source and candidate image selection, preserve all existing environment values and add the approved weekly publishing flag. Do not replace real secrets with `.env.example`.
6. Start exactly one application container. Keep 512 MiB, 1.5 CPUs, loopback 8087, private 8090, read-only root and existing supervision/security controls.
7. Verify DB health, migration head, connected receiver, fresh worker, queue preservation, new reports, exports and role/CSRF behavior. Do not reset the linked session or send a fake group order.
8. Install the weekly-only Friday 14:00 cron file after timezone verification. Preserve existing unrelated/legacy jobs. Confirm there is no daily summary job and no immediate production publication created by testing.
9. Record the deployed image, actual checks, first due publication and any outstanding authenticated/live acceptance.

## 14. Rollback and remaining acceptance

Before migration, discard/rebuild the candidate safely. After `0005`, prefer a schema-compatible forward fix. **Do not select the old image as a standalone rollback**: it cannot write mandatory participant references or handle new guest orders correctly. Restoring a pre-migration DB requires separately reviewed preservation/reconciliation of any newer business activity. The migration intentionally refuses an automatic downgrade.

Outstanding live acceptance remains: a genuine group order attributed correctly with matching member/group/export totals; real supported edit/cancel behavior; the first scheduled weekly WhatsApp report acknowledged and observed in the intended group; current-user authenticated browser verification where credentials are unavailable to the maintainer; and peak simultaneous load under the exact container caps. No 100% accuracy or guaranteed upstream delivery claim is made. Golden fixtures do not establish real-world parser precision/recall.

Friday post-snapshot changes, unsupported messages, upstream outages and unknown transport outcomes remain explicit review/recovery cases. Published snapshots are not silently revised. Offsite backups, alert routing, trusted permanent public access, MFA and governance policy remain separate deferred work.

## 15. Deliverables and continuation instructions

The release contains the actual layered source, forward migration, participant identity transition, order rules, canonical aggregation, independent reconciliation, paginated daily/weekly/monthly dashboard, guest/member review, weekly publication service, cron example, durable retry/audit mechanisms, unit/PostgreSQL/Node tests and this deployment/rollback/acceptance report.

Continue from the source and final evidence, not from an older master prompt that mentions daily publishing or synthetic-email guest creation. Never describe a tested candidate as deployed before the production execution record confirms it. Keep production secrets and dumps outside the package. Obtain the specified approval only after the tested build and migration rehearsal are concrete and reviewable.


### Verified build and deployment record — 12 November 2026

- Candidate image: `tiffintrack:layered-20261112`, SHA-256 `d3f857f80dc4174f4a43424fdc7ebf015304dd7fb84837de3dd77ce4ba43218d`, non-root user `app`.
- The full supplied Dockerfile built successfully. Its separate Node verification stage ran all 24 tests; test files are excluded from the final runtime image.
- Final code manifest verified successfully. Deployment helpers are included with explicit approval guards.
- Production now runs `tiffintrack:layered-20261112` / `0005_participants`, healthy, zero restarts. The user approved the migration and deployment. No relink, fake order, test WhatsApp post or billing email was performed.
- Production migration, controlled restart and Friday-only publishing enablement completed after approval. `/etc/cron.d/tiffintrack-summary` contains only Friday 14:00, the host is Asia/Kolkata, and publishing is enabled. The first scheduled target is Friday 13 November 2026 at 14:00 IST; actual transport completion remains best effort and the first real weekly delivery has not yet been observed.
- The test evidence contains the original benchmark import-path failure. A follow-up harness supplied `PYTHONPATH` correctly and completed the synthetic workload, code-manifest check and full build. The benchmark import-path failure is retained in the test evidence alongside its successful follow-up. A temporary-file render probe was also blocked by the read-only container filesystem; the successful follow-up streamed code through standard input without weakening filesystem protection.


### Final production verification — 12 November 2026, 20:03 IST

- Migration compared and preserved 2 Users, 0 Orders, 0 Payments, 0 EmailLogs, 1 participant, 2 aliases, 1 source event and 864 encrypted auth records before restarting the compatible release. Ordinary subsequent session-key updates are expected.
- Running image and schema match the reviewed release. One application container remains, limited to 512 MiB / 1.5 CPUs; sampled usage was 335.7 MiB and 0.64% CPU, not a peak-load claim.
- Receiver connected, receiver heartbeat age two seconds and worker heartbeat age zero at deployment checks. One historical processed/review message remains; no pending/error backlog. Its historical processing-completion time is unknown rather than fabricated.
- Current canonical total: zero active orders/tiffins/amount, one review; provisional. No genuine order was manufactured to change that result.
- Read-only server-side administrator rendering passed for participant reports, Operations, WhatsApp administration, personal counts, both Excel downloads, participant detail and source-message detail. Workbooks reopened successfully. This used privileged test request contexts, not the user's browser/password login. No business/password writes were made.
- Real HTTPS login page returned 200; protected pages/exports redirected unauthenticated requests with 302; a mutation without a CSRF token returned 400. Certificate verification used the installed temporary certificate.
- No test publication was created. Weekly-only cron and Asia/Kolkata timezone were inspected. The next scheduled run is Friday 13 November 2026 at 14:00 IST.
- Office-network dashboard: `https://192.168.1.101/admin/orders`. Permanent public access remains deferred.

The included `deploy/release-layered.sh` is a guarded one-time transition from the old baseline. It intentionally refuses to rerun against the already migrated release. Subsequent releases need a compatible forward deployment plan.
