> Historical pre-layered verification record. Read LAYERED-IMPLEMENTATION.md for the current candidate, weekly-only publishing, guest identity model and pending production approval.

# TiffinTrack engineering audit and remediation — 12 November 2026

## 1. Scope and executive assessment

This is a scoped engineering remediation of the existing application, not a rewrite or a claim of complete production certification. The user's latest instruction authorizes fixes and permits deferring WhatsApp work that is not doable. The user separately deferred public/domain access. No DNS, ingress, firewall or authentication-provider changes are part of this release. The working WhatsApp connector and linked session are preserved.

The principal fixes address concurrency and reporting accuracy: refresh cached users when acquiring financial locks; recheck cancellation ownership after waiting for a lock; recheck email eligibility under lock and preserve the exact claimed recipient; keep all sheets of an export on one committed database snapshot; sanitize workbook text and format money consistently. A new read-only reconciliation command and administrator Operations page detect discrepancies and distinguish component readiness from proven business delivery.

No migration or dependency upgrade is required. The database remains at `0004_whatsapp`. Existing financial data is not rewritten. Exactly one application container remains, with 512 MiB memory and 1.5 CPU limits. The release image is `tiffintrack:audit-fix-20261112`.

## 2. Actual architecture and data flow

```text
Browser -> host Nginx HTTPS -> host 127.0.0.1:8087
                              -> one app container :8000
                                  Gunicorn/Flask
                                  Node linked-device receiver
                                  Python durable-inbox worker
                                            |
                                     host PostgreSQL

WhatsApp -> receiver -> unique inbox -> event/identity -> order/audit
Orders -> shared financial aggregation -> dashboard/payment/email/Excel
Host cron -> app billing command -> SMTP STARTTLS -> individual member
Host cron -> native pg_dump -> restricted local backups/checksums
```

Receiver control port 8090 remains internal. The database is a host service. There is no Redis, message-broker service, second app container or Excel datastore. An SSH forward is temporary client access, not permanent public hosting.

## 3. Read-only production baseline

At approximately 15:25 IST on 12 November: container `tiffintrack`, image `tiffintrack:gateway-fix-20261112`, Docker health healthy, approximately 297.5 MiB / 512 MiB and 0.53% sampled CPU. PostgreSQL, Nginx and Cloudflared services were active. Migration head was `0004_whatsapp`.

Business counts: 2 users, 0 orders, 0 payments, 0 email logs. One WhatsApp inbox row was processed. The connection was connected, with fresh receiver/worker heartbeats; last reception was 10:29:18 IST. Encrypted session records existed. Do not interpret zero confirmed orders as evidence that real order-to-total acceptance has passed.

Installed host cron includes Monday 09:00 billing, daily 02:00 database backup and daily 03:00 login-attempt cleanup. The TLS file is `/etc/tiffintrack/tls/server.crt`. Final deployment evidence is recorded separately in the deployment section/output logs. These are timed observations, not an uptime guarantee.

## 4. Risk register and implemented fixes

No observed P0 financial corruption or unauthorized production access was established. The following are confirmed code defects, reproduced race conditions, or explicitly identified gaps; theoretical exposure is not described as an observed incident.

| ID | Severity / classification | Evidence, trigger and impact | Remediation and acceptance |
|---|---|---|---|
| TT-01 | P1, confirmed concurrency defect | `app/services.py:lock_member`: a locking SELECT could return an already-loaded ORM User without refreshing attributes. Another transaction's eligibility change could be missed. | `populate_existing=True`; real PostgreSQL regression loads user, changes active flag in another connection, then verifies locked read is fresh. No schema change. |
| TT-02 | P1, reproduced authorization race | `app/routes.py:cancel_order`: ownership checked before waiting on the old member lock. Concurrent reassignment could change owner before refresh/cancellation. | Recheck ownership after refresh; if owner moved, reject/reload rather than lock users in an unsafe new order. Regression simulates reassignment while acquiring lock and expects 403; no cancellation applied. |
| TT-03 | P1, confirmed invoice-audit mismatch | `scripts/weekly_billing.py:run`: retry of failed invoice reused the old logged recipient but sent to current member email. | Retry updates logged recipient and freezes destination/body before committing the sending claim. Regression changes member email and verifies actual send argument equals stored recipient. |
| TT-04 | P1, confirmed stale eligibility risk | Billing selected eligible IDs before member locking, then did not revalidate eligibility. | Refresh locked member and recheck active/member/billing-email flags. Regression changes eligibility during locking and verifies no claim/send. |
| TT-05 | P1, confirmed report snapshot gap | Weekly summary/detail and multi-period Excel reads ran in separate statement snapshots under normal PostgreSQL isolation. A concurrent commit could make sheets disagree. | Dedicated read-only-use report session with PostgreSQL REPEATABLE READ, closed after export data is captured. Real PostgreSQL test commits a second order between summary/detail and verifies both retain the original snapshot. |
| TT-06 | P2, confirmed export robustness gap | `app/reports.py:excel_safe`: XML-invalid control characters could cause workbook generation failure; whitespace-prefixed formula text was not explicitly guarded. | Remove illegal XML controls; escape dangerous leading formula text including whitespace-prefixed cases. Preserve numeric cells and Unicode; regression fixtures cover both. |
| TT-07 | P2, confirmed reporting ambiguity | Weekly label “Orders” meant quantity; count workbook lacked explicit money format. | Label weekly column “Tiffin quantity”; format INR numerically with two decimals in both exports. Workbook reopened in tests to validate values, cancellation exclusion and formats. |
| TT-08 | P2, missing implementation | No read-only independent financial comparison or repeated-order advisory view. | `app/reconciliation.py`, `reconcile-week` CLI and admin Operations page. Independently stream/compute active-order amounts and compare shared aggregation and paid amounts. Flag candidate repeats; never auto-delete. Deliberately mismatched payment detected in regression. |
| TT-09 | P2, missing operational visibility | HTTP/DB health did not distinguish receiver/worker/backlog failures for administrators. | `/admin/operations` shows separate state/HB ages, inbox errors/backlog, review count, last-received age and email states. Admin isolation tested. It expressly does not prove upstream delivery. |
| TT-10 | P1, missing recovery capability | No verified off-server backup destination or agreed recovery objectives. | Local backup and populated isolated restore verified; offsite disaster recovery remains OPEN, requires destination/policy. |
| TT-11 | P1, missing live acceptance evidence | No clear real WhatsApp order or genuine invoice receipt was observed in the baseline. | OPEN, no synthetic/live acceptance conflation. Connector work deferred; real outbound billing requires authorized recipient and completed bill. |
| TT-12 | P2, confirmed deployment limitation | LAN/self-signed TLS/temporary SSH access does not provide permanent global access. | DEFERRED explicitly by user. Preserve existing ingress. |
| TT-13 | P2, governance/verification gaps | No established retention/redaction policy, tested alerts, administrator MFA or full host-loss drill. | Roadmap only. Do not claim implemented, certified or verified. |

TT-01–09 require small existing-module changes; regression risk is concentrated in locking, report transaction lifetimes and SMTP claims. No data migration is required. Roll back to the previous compatible image if needed; do not restore an older business database just to roll back code. Dependencies and connector code are unchanged.

## 5. WhatsApp reliability report — preserved/deferred

The selected-group connection was running at baseline. The current design has durable inbox uniqueness, encrypted credentials, conservative application, event/audit links and independent process supervision. Existing 17 Node tests passed again, including metadata failure containment, envelope deduplication, pairing guards and authenticated session encryption. Existing Python WhatsApp tests were included in the PostgreSQL suite.

No new protocol feature, full-history recovery, hours-long real disconnect experiment, group-JID migration, real edit/delete delivery guarantee or parser expansion was implemented. These remain NOT VERIFIED live. The connector is unofficial and cannot guarantee upstream completeness. No QR reset or group message was sent by this remediation.

## 6. Parser accuracy assessment

The existing parser is deterministic and conservative. Explicit supported quantity/meal phrases can qualify; questions, conditions, uncertain dates, media-only content and unclear targets go to review. Parser changes were deferred in this scope.

The audit's examples are useful fixtures, not a labeled real-world corpus. “Kal usual bhejna”, “same as yesterday”, multi-person requests, “add one more” and negation-heavy revisions must not be accepted by inventing context. Relative dates still face current order-window/future-date rules. There is no separate lunch/dinner field or ordering cutoff/holiday calendar. Real-world precision, recall, attribution and false-positive rates are NOT MEASURED. Do not substitute synthetic fixture pass rate for those metrics.

A later evaluation should label source-safe examples for intent, quantity, date, sender/target and expected review/application outcome, report per-class errors and prevent any uncertain result from directly posting financial entries.

## 7. Identity resolution assessment

Stable WhatsApp IDs/aliases, not display names, identify participants. Conflicts require review. Internal synthetic members have billing disabled until deliberately mapped/configured. The existing controlled mapping route locks affected users and audits eligible order reassignment; finalized periods block changes. The new cancellation recheck closes the post-mapping ownership race. Identical names, aliases and mapping are covered by existing tests; no new real participant merge was performed.

## 8. Database integrity assessment

Existing PostgreSQL migrations enforce quantities/statuses, foreign keys, unique form/inbox/event identifiers, member-period payments and unique active email claims. Member locks coordinate order/payment/invoice operations. New locked reads refresh cached attributes. No table/index/isolation default was globally changed. Report isolation is scoped to its own connection, leaving financial mutation transactions unchanged.

The full suite was run against a newly created disposable `_test` database, including migrations/constraints and concurrency tests. The new report test uses a genuinely separate committing connection. Keep test URLs isolated; never run fixture cleanup against production.

## 9. Order lifecycle and reconciliation

Active -> cancelled preserves historical records. WhatsApp edits/cancellations retain source/audit information. Distinct legitimate orders on the same date are supported; identical quantity/date does not prove duplication. Form UUIDs and transport keys prevent replays through their supported paths. Manual submissions without an idempotency key remain a legacy API limitation; the rendered browser form includes one.

New reconciliation reports candidate active orders grouped by member/date/quantity/price, including cross-source candidates. It does not know whether an upstream order was never delivered and cannot infer missing messages without an independent expected-order source. Administrator review remains necessary. There is no silent deduplication, bulk historical repair or historical repricing.

## 10. Financial reconciliation and email assessment

Decimal historical price × active quantity is the financial basis. Cancelled records are excluded from aggregates; detailed workbook history still shows their status. Independent Python accumulation now checks shared SQL totals and paid amounts for a selected week. Dashboard/billing/payment continue to use shared weekly totals. Export snapshot isolation prevents concurrent changes from mixing sheets.

Billing preserves the previous completed Monday–Sunday business week. `sending` is committed before SMTP. Sent/unknown/sending claims and paid periods freeze financial changes. Unknown or interrupted delivery must be reconciled with provider evidence, not automatically resent. The new page flags unknown and old sending records; age is based on row creation, so a recently retried old row can be flagged conservatively. This is an advisory queue, not a new reliable claim timestamp.

Office 365 SMTP configuration is preserved; no real email was sent during tests or release. Mocks cover success, failures and unknown delivery. Actual inbox delivery is NOT VERIFIED. Excel attachments, partial payments, refunds and adjustment ledgers remain separate unimplemented features.

## 11. Security, privacy and threat model

Assets: member identity, password hashes/sessions, financial records, raw selected-group messages, linked-device secrets, SMTP credentials and backups. Actors: authorized member/admin, unauthenticated internet client, compromised account/device and privileged host operator.

Trust boundaries: browser/Nginx, proxy/application, app/host PostgreSQL, app/internal bridge, receiver/WhatsApp, billing/SMTP, host/backups. Existing controls include signed secure cookies, password-derived session invalidation, CSRF, role/ownership checks, parameterized queries, escaped templates, private bridge token, encrypted linked-device records, bounded login attempts and restricted local secrets/backups.

New tests verify that members cannot access Operations or manipulate another member's cancellation after reassignment. Existing member/export/review authorization and CSRF tests remain passing. Diagnostics avoid member email addresses/private message bodies; reconciliation exposes internal IDs only to admins/host CLI.

Residual risks: administrator compromise, privileged host access, raw-message retention, backup confidentiality/offsite absence, self-signed TLS and the unofficial upstream protocol. Administrator MFA, retention policy, incident escalation and external alerts are not implemented by this release. No vulnerability-free or compliance claim is made.

## 12. Docker/infrastructure assessment

One non-root app container, read-only root, bounded tmpfs, dropped capabilities, no-new-privileges, PID limit, rotated logs and independent component restart logic are retained. PostgreSQL/Nginx/cron remain host services. The release layers reviewed application code over the exact existing dependency image; the complete source also retains its normal pinned Dockerfile for full builds.

No new running container is introduced for tests or deployment. Tests/benchmarks use host tools and disposable databases. Per-process peak memory and simultaneous end-to-end receiver/web/email/export capacity inside the capped container are NOT VERIFIED by the host-side benchmark. Do not remove caps based on those results.

## 13. Performance evidence

Fresh before/after benchmarks used the same host, separate real PostgreSQL databases, 1,000 seeded users and 13,986 seeded orders, plus 30 order submissions. Most operations used 30 samples at concurrency five; Excel used five sequential samples. Scope excludes public network, Nginx/Gunicorn and container CPU/memory enforcement. Zero sampled errors were observed.

Raw JSON outputs are supplied as `TiffinTrack-Audit-Benchmark-Before.json` and `TiffinTrack-Audit-Benchmark-After.json`. This is a regression comparison, not a capacity guarantee. REPEATABLE READ deliberately spends an additional report connection to guarantee consistency; no assertion of universal speed improvement is justified. Do not add indexes without measured query-plan evidence. Long exports hold a snapshot until completion and should be monitored for resource/connection pressure.

## 14. Monitoring and operational status

New authenticated page: `/admin/operations`. Navigation includes Operations for administrators. CLI: `flask --app manage.py reconcile-week --week YYYY-MM-DD`; defaults to the current business week. A financial discrepancy produces a nonzero CLI exit. Candidate repeats and uncertain mail are advisory, not automatically mutated.

States distinguish not configured, receiver unavailable, worker unavailable, processing errors, old backlog and components-ready-with-delivery-unproven. A quiet group cannot be distinguished from upstream message loss solely by time since the last message. Heartbeats cannot prove a genuine order was understood. Review events are visible separately.

Host monitoring for disk, backup age, TLS expiry, Docker restarts/resource usage and SMTP provider outcomes still requires host checks. No external alert destination is configured. A 200 `/health` remains DB/web health, not business-delivery health.

## 15. Backup and recovery assessment

Daily native local backups/checksums/14-day retention are preserved. An isolated populated restore with 1,000 users and 14,016 orders was completed during this remediation; restored counts and the entire selected-week independent reconciliation result matched the source fixture. A schema-only/empty test restore was also completed, but it is not presented as the populated recovery proof.

Before production replacement, take a fresh native backup and restricted copies of environment/session key and prior source/Compose. Never print those files. Local snapshot restoration does not prove total-server-loss recovery: offsite copy, encryption-key escrow, replacement-host provisioning and agreed RPO/RTO remain open. Do not delete the production database during cleanup or rollback.

## 16. Global-access recommendation — deferred by user

Existing access is LAN/VPN or a temporary local SSH forward. Permanent worldwide access is intentionally not configured. No domain is invented.

| Option | Suitability | Cost/maintenance/security trade-off |
|---|---|---|
| Owned domain + outbound Cloudflare Tunnel | Good fit for a NAT-hosted server already using Cloudflared | Requires account/DNS ownership and tunnel policy; retain app auth and correctly forwarded Host/scheme; evaluate separate admin Access policy without blocking members |
| Public HTTPS reverse proxy | Suitable with controlled public ingress | Requires public routing, trusted TLS renewal, firewall/WAF/rate limits and inbound maintenance |
| Managed VPN/private network | Appropriate for a closed group of authorized devices | Requires enrolled clients and device revocation; users must connect before opening the private site |

Reassess current provider terms/costs when this phase is authorized. Never expose 5432, 8090 or 8087 publicly. External-network testing and trusted permanent TLS remain NOT VERIFIED/deferred.

## 17. Failure-scenario matrix

“Regression” means isolated automated coverage of the relevant behavior, not a live physical failure experiment. “Runbook only” means the actual fault was not injected. Connector-specific expansion was deferred.

| # | Scenario | Evidence/result and residual recovery requirement |
|---|---|---|
| 1 | WhatsApp disconnect during order | Existing receiver/inbox tests; real network delivery boundary NOT VERIFIED; inspect inbox/HBs and reconnect |
| 2 | Hours-long WhatsApp outage | NOT VERIFIED; upstream completeness cannot be promised; reconcile missed business activity |
| 3 | Receiver crash after insert | Durable inbox/duplicate regression coverage; independent supervision retained |
| 4 | Worker crash during application | Existing PostgreSQL rollback/error-inbox regression; explicit corrected retry |
| 5 | PostgreSQL unavailable | Health/transaction design assessed; physical outage not injected; restore DB connectivity before resuming |
| 6 | Two workers on same event | Existing real PostgreSQL concurrency regression passed |
| 7 | Repeated event deliveries | Existing real PostgreSQL dedup regression passed |
| 8 | Same participant names | Existing identity fixture passed; names are not keys |
| 9 | Participant identity changes | Alias/conflict/mapping tests passed; real protocol transition NOT VERIFIED |
| 10 | Edit after cancellation | Existing lifecycle fixtures passed; real edit transport NOT VERIFIED |
| 11 | Cancel after finalization | Existing financial freeze regression passed |
| 12 | Concurrent payment/order | Shared refreshed member locks and PostgreSQL tests; broader physical stress not performed |
| 13 | SMTP accepted then timeout | Unknown-delivery mock regression passed; provider reconciliation required, no automatic retry |
| 14 | Corrupt backup | Archive/checksum procedures assessed; this release tested valid populated restore, not a new corruption-injection drill |
| 15 | Wrong/missing session key | Existing encryption tamper/wrong-key Node tests passed; restore matching key or authorized relink |
| 16 | Disk full | Runbook only; prevent via host disk monitoring, do not delete financial/session data |
| 17 | Restart during migration | No migration in this release; interrupted-DDL deployment drill NOT VERIFIED |
| 18 | Invalid SDK patch | Existing pinned/idempotence/refusal Node tests passed |
| 19 | Export during order commit | New real PostgreSQL concurrent-commit snapshot regression passed; heavy capped-container ingestion/export NOT VERIFIED |
| 20 | Nginx unavailable | Runbook only; upstream health is separate; validate config before reload |
| 21 | Expired certificate | Current validity inspected; forced expiry not injected; replace certificate without bypassing CSRF/TLS permanently |
| 22 | Exhausted DB connections | Pool boundaries assessed; saturation drill NOT VERIFIED; diagnose leaks/long reports before increasing capacity |
| 23 | Server restarts during billing | Existing sending/unknown no-resend tests; actual reboot not injected; reconcile provider outcome |
| 24 | Message burst | Prior synthetic benchmark plus regressions; real transport/capped-container peak NOT VERIFIED |
| 25 | Cross-member URL | Existing authorization suite plus new Operations/ownership-race regressions passed |
| 26 | Multiple quantities | Existing conservative parser fixtures passed; no unsafe new heuristic |
| 27 | Conditional order | Existing parser review fixtures passed |
| 28 | Intentional second daily order | Existing multiple-order behavior retained; new duplicate advisory never auto-deletes |
| 29 | Public tunnel disconnect | Public access deferred; no new failure injection; temporary SSH access is not permanent hosting |
| 30 | Complete server unavailable | Local populated recovery verified; offsite/key/provisioning full-loss drill OPEN |

For unverified physical scenarios, the remaining business risk is delayed access/tracking or recovery beyond an undefined objective. No corresponding data-loss guarantee is made. Detect through component/host monitoring, preserve durable records and keys, recover dependencies first, then reconcile before replaying financial/outbound actions.

## 18. Regression evidence and maintainability

Baseline local run: 63 passed, 4 PostgreSQL tests skipped. After remediation: 71 local passed, 5 skipped. Isolated server PostgreSQL run: **76 passed** in approximately 26 seconds; one harmless pytest cache-permission warning. **17 Node tests passed**. Ruff lint and formatting checks passed for all 44 Python files. The cache warning did not hide a test failure.

New tests: `tests/test_audit_remediation.py` and `tests/test_postgres_reports.py`. They cover financial mismatches, non-destructive repeated-order advice, recipient changes on retry, eligibility changes, admin isolation, numeric workbook values/formatting, ownership changes, uncertain-mail no-resend, unsafe text and genuine concurrent PostgreSQL commits/identity-map refresh.

Changes remain in existing service/report/route modules with two small focused modules for reconciliation and operational visibility. No framework, background service, AI dependency or global isolation setting was added. Limits and failure behavior are documented rather than hidden behind optimistic status labels.

## 19. Deployment and rollback

Release process: fresh native production backup; protected prior source/environment/Compose; capture non-secret business baseline; build reviewed application overlay; validate Compose; recreate the single service; wait for Docker health; compare business baseline; check migration revision, reconciliation, HTTPS, exports and admin page. No production fixture/order/email is created. No relink or migration is required.

Automated readiness failure restores the prior image selection. For a later rollback, select `tiffintrack:gateway-fix-20261112` with the protected previous Compose and recreate the service. Since this release has no migration, the previous image can read the same schema. Retain new financial activity; do not restore an older DB for code rollback. Restore prior source separately if a host-side operational command must match the previous image.

Actual deployment completed successfully on 12 November 2026. Image `tiffintrack:audit-fix-20261112` (image SHA-256 beginning `4143b6823043`) was running healthy, zero restarts, 512 MiB/1.5 CPU limits. Pre/post production business counts and active amount matched exactly. Migration head remained `0004_whatsapp`; live current-week reconciliation reported no mismatches. Nginx configuration validation and trusted-temporary-certificate HTTPS `/health` passed. Certificate expires 9 February 2027 at 18:06:57 UTC.

A subsequent resource sample was 281.9 MiB / 512 MiB and 0.58% CPU; this is a sample, not peak capacity. At 15:44 IST the existing receiver was connected with a 2-second heartbeat age, worker heartbeat age 0, one processed inbox item, one review item and no pending/error backlog. The linked session was preserved.

Live HTTPS login-form rendering, security headers and the unauthenticated admin gate passed with certificate verification against the installed temporary certificate. The saved bootstrap credential was not usable for an authenticated read-only browser check (it may be changed or require password change). No password reset was performed. **Live authenticated dashboard/Operations/workbook checks with the user's current login remain NOT VERIFIED.** Their isolated regression tests passed. No live order, payment, group message or email was created.

The new disposable audit/test/recovery databases were explicitly removed after verification. Production and previous-release backups remain protected. Historical test databases from earlier tasks were not indiscriminately removed.

## 20. Prioritized roadmap and acceptance checklist

Completed scope: TT-01–09 code fixes/new diagnostics; relevant regressions; populated isolated restore; benchmark comparison. Deployment acceptance requires the execution evidence appended below.

Next business acceptance: observe one clear genuine order attributed correctly, verify totals/Excel, then authorized edit/cancel and invoice inbox receipt. These remain separate from synthetic fixtures. Do not send fake business messages or bulk email to manufacture acceptance.

Next operational phase: choose offsite encrypted backup/key destination and test full replacement-host recovery; define alert destinations and retention/access policy; agree measured availability/backlog/recovery objectives. No SLA/RPO/RTO is claimed.

Deferred user phase: owned hostname, trusted public HTTPS/tunnel or VPN, admin MFA/edge policy and external/mobile network validation. Record DNS/account permissions before configuration.

Future performance changes require profiles/query plans and repeatable tests, especially report connection use and peak combined process memory. Changes to partial payments, refunds, historical adjustments, lunch/dinner/cutoff rules or parser semantics need explicit business requirements, migration/rollback assessment and regression fixtures. Preserve historical prices and audit history throughout.


### Fresh benchmark comparison (P95 milliseconds)

| Operation | Before | After |
|---|---:|---:|
| Login | 117.56 | 142.80 |
| Member dashboard | 57.06 | 53.18 |
| Order creation | 44.74 | 46.44 |
| Weekly aggregation | 35.33 | 42.21 |
| Excel export | 567.58 | 593.54 |

Each run recorded zero sampled errors. Process peak RSS was approximately 275 MiB in each host-side run. Small samples and host scheduling do not establish that every timing difference is caused by the changes. Export consistency adds an intentional separate transaction/connection. No unsupported overall speedup or container peak-capacity claim is made.
