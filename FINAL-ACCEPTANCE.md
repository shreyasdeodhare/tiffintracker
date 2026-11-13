# TiffinTrack — final engineering and acceptance audit

Audit date: 12 November 2026, IST. **Application running; technical verification passed; full business acceptance pending.** This distinction is deliberate: production currently contains no confirmed orders. No synthetic group messages, financial records or summaries were created in production.

## 1. Current production, independently checked

| Item | Observed result |
|---|---|
| Image | `tiffintrack:stabilization-20261112` |
| Image SHA-256 | `7881a32451f8ae01e5eec06620ba156d52be9251c058b0a05dbd292c99205923` |
| Production code | `CODE-MANIFEST.sha256` matched all listed files |
| Schema | `0005_participants`; no migration performed |
| Container | One TiffinTrack application service, healthy, zero restarts, no OOM flag |
| Limits/security | 512 MiB, 1.5 CPU, 128 PIDs, user `app`, read-only root, unless-stopped recovery |
| Processes | Supervisor, Gunicorn master/workers, Node receiver and Python inbox worker observed |
| Ports | Host loopback `127.0.0.1:8087` to app 8000; bridge 8090 not published |
| WhatsApp | Connected; selected JID matches the previously approved group identifier; stored name `Amanora 507B Office The Meal Habit` |
| Heartbeats | Receiver 2.56 seconds and worker 0.74 seconds at live check |
| Business state | Users 2; participant 1; orders 0; message/inbox 1; review 1; payments 0; publications 0; pending/error inbox 0 |
| Session | 869 encrypted records in the consistent restore snapshot; preserved and authenticated decryption verified |
| Schedule | Friday 14:00 Asia/Kolkata only; Saturday–Friday snapshot; enabled |
| Other cron | Daily 02:00 backup and 03:00 login-attempt pruning preserved; retired email billing absent |
| Access | Server-side HTTPS verified at `https://192.168.1.101`; client requires LAN/VPN routing or a temporary authorized SSH forward |

Final recorded status at 23:14:25 IST was healthy with zero restarts. The image was not recreated or rebuilt during this audit. Private DNS/public exposure was not changed. Host PostgreSQL is shared by other applications; “one container” refers to TiffinTrack, not the entire server.

## 2. Scope, architecture and code quality

The working source was compared with the server code manifest. Baseline inventory contained 136 files and 78 Python source files; Python syntax was parsed and all bridge modules passed Node syntax checks. The audit reviewed factory/configuration, authentication, models/migrations, decision/identity paths, parser, bridge, worker, reports/workbooks, publication outbox, supervisor, Compose, Nginx, cron, backup and tests.

The existing layers are retained: presentation routes/templates validate HTTP and enforce roles; application services orchestrate transactions, participant mapping, orders, review, reporting and publication; domain modules contain deterministic parsing, quantity/date/price rules and summary formatting; infrastructure modules own SQLAlchemy/PostgreSQL, encrypted transport state, projections and workbook persistence. Compatibility exports and deferred imports preserve established entry points. No failing import cycle was demonstrated; broad relocation of compatibility code would add risk without correcting a verified defect.

The largest application decision module is 240 lines; existing focused responsibilities and pinned direct/transitive dependencies were preserved. Runtime `pip check` reported no broken requirements. Ruff passed. This is evidence of the executed checks, not a claim of exhaustive formal verification, complete type annotation or a new third-party vulnerability audit.

**No new runtime defect was demonstrated that justified changing or redeploying application code.** Existing duplicate protection and financial rules were verified again rather than rewritten. Documentation defects were corrected: obsolete candidate wording, guaranteed-local-forward claims and bootstrap-password assumptions. New recovery/domain/acceptance guidance records current facts and actual limitations.

Changed package files in this stage: `README.md`, `ACCESS.md`, `FINAL-ACCEPTANCE.md`, `ACCEPTANCE-MATRIX.csv`, `DOMAIN-READINESS.md`, `RECOVERY.md`, the documented `verification/` reproduction harness, and generated source inventory/checksum/evidence files. The harness Python was formatted and passed the final package Ruff checks. No Python application, bridge, migration, Docker, Nginx or production cron behavior changed. The previous stabilization corrections remain in the source and running image.

## 3. WhatsApp and order correctness

The Node transport filters the actual selected group JID, excludes private/other-group messages, disables full history synchronization, persists encrypted auth, writes durable unique inbox envelopes, refreshes heartbeat and reconnects with backoff. The supervisor recovers exited components. Encrypted state survived the prior approved restart and was independently decrypted from the isolated restore without opening a second session. Selected-group discovery avoids unnecessary all-group queries; tests cover discovery failure/backoff and pairing refresh.

Stable sender JIDs and verified aliases identify participants. Display-name collisions never imply identity. Guests get participant records, not login Users. Conflicting aliases and invalid/unknown identifiers require review. Deterministic English/Hindi/Hinglish rules accept explicit quantities and self-order/change/cancel phrases; uncertain wording, payment messages, media and multi-person requests are not silently counted. This does not promise recognition of every informal sentence.

Existing lifecycle coverage passed: create, multiple same-day originals, multiple tiffins, edit, modify, delete/cancel, exact-source ownership, late/out-of-order revision review, ambiguous cases, administrator splits/corrections, rollback and audit preservation. Only review-state messages are actionable. Before any original/split creation, the application checks creation audits for that group/source ID; ignoring, resetting or cancelling an earlier event does not make its source reusable. Durable transport uniqueness and transaction locks prevent repeated/concurrent ingestion from creating another order. Split allocations deliberately share one source while creating their separately approved allocations.

Transport acknowledgements are not proof that every participant received/read a summary. Disconnections, unsupported upstream changes and capture gaps remain limitations of the authorized unofficial connector. No live reconnect failure was deliberately induced; graceful recovery behavior was checked through source/tests and the already persisted running session.

## 4. Financial reconciliation and Excel

Order-record count, tiffin quantity and monetary amount remain separate. Amounts use stored historical Decimal prices; cancellations are excluded from active totals. The shared participant/source projection backs daily, Saturday–Friday weekly and monthly reports and exports. Independent raw-order aggregation checks participant and source/group totals and reference consistency. Pagination affects presentation rows, not full financial aggregates.

The reopened-workbook/dashboard/publication regression passed with 3 active records, 6 tiffins and INR 461.20, plus an excluded cancelled record, different stored prices and two people sharing a display name. All Daily/Weekly/Monthly worksheets agreed with dashboard totals; independent SQL and the immutable weekly snapshot agreed. Formula/control-character protection remains covered. The separate legacy Monday–Sunday payment/history export is retained for compatibility and is not the canonical weekly group report.

Read-only production reports and both Excel endpoints rendered successfully and their workbooks reopened. Current live totals reconcile at zero and are correctly provisional because capture does not cover the full reporting period and a review remains unresolved. These empty production totals cannot establish real order attribution accuracy.

## 5. Weekly publication

Cron and host timezone match the approved Friday 14:00 IST requirement, covering previous Saturday through Friday. No daily publisher or Monday individual email billing is active. Preparation records actual capture time, period, participants, record counts, quantities, money, cancellations, unresolved counts and explicit provisional reasons. Reconciliation mismatches block preparation; later business changes do not rewrite saved historical snapshots.

Database uniqueness plus serialized preparation prevents multiple publications for the same group/period/source. Snapshot/part immutability triggers remain installed. Outbox claim and acknowledgement tests cover concurrent senders, saved message identity, ordering, sent-part non-replay, timeout/unknown outcomes and evidence-based operator retry. The separate PostgreSQL outbox test uses mocked transport only.

**No real publication exists yet.** First scheduled delivery is Friday 13 November 2026 at 14:00 IST and is pending observation. The job was not manually invoked to fabricate completion; no synthetic financial summary reached the group.

## 6. Executed tests and security

| Gate | Result |
|---|---|
| `python -m compileall -q app scripts tests` | Passed in isolated verification stage |
| `ruff check .` | Passed |
| `ruff format --check .` | Passed; verification output reported 87 files formatted |
| Python suite with both disposable PostgreSQL URLs | **96 passed, no skips, 28.95 seconds** |
| `npm test` in bridge | **24 passed; zero failed/skipped** |
| PostgreSQL concurrent outbox/mock transport | Passed; no WhatsApp connection |
| All bridge module `node --check` | Passed |
| Runtime `python -m pip check` | No broken requirements |
| `docker compose config --quiet` | Passed against protected deployed configuration without printing secrets |
| Production source manifest | Matched |

Security tests exercise role/object isolation, cross-member orders/cancellation/export denial, login throttling, password/session invalidation, safe redirects, proxy handling, source ownership and invalid inputs. Real HTTPS returned login 200 and unauthenticated protected-route 302; an un-tokened POST was rejected with CSRF 400. Secure/HttpOnly/SameSite=Lax cookies, HSTS, CSP, nosniff and frame denial were observed. Database queries use parameterized ORM/SQL paths. Application database role is not superuser and cannot create databases or roles.

Read-only administrator rendering used `login_user` inside a test request context to verify views. **That is not password authentication.** Current production browser password login was not executed because current credentials/access were not available to that acceptance path. No password/session reset or bootstrap-credential substitution was attempted. Isolated password/login tests passed, but the production browser acceptance box remains pending.

The sampled hour of app logs contained 123 lines, zero supervisor component-exit lines, zero inbox processing errors and zero publication storage errors. A check against known protected environment values found no such secrets in the sampled logs. This scoped check is not a guarantee about every historical log or unknown sensitive value.

## 7. Actual performance and resource evidence

Live container samples were approximately 299.3–306 MiB of 512 MiB, 0.29–0.92% CPU and 19 PIDs, with no restarts/OOM flag. Five PostgreSQL connections were observed: four idle and the inspection query active. Disk usage was 83%, with approximately 154 GiB available. The shared host's aggregate Docker usage was inspected; nothing was pruned or removed.

Three privileged read-only production render samples: order dashboard 18.87/37.76/49.25 ms min/median/max; canonical Excel generation plus reopen 10.09/26.89/64.63 ms. These are in-process server checks on the current empty dataset, not browser/network latency or percentile capacity claims. Actual HTTPS login GET took 6.281 ms in the later sample; protected-route timings are redirects, not authenticated dashboard timings.

Controlled isolated PostgreSQL workload: 1,000 users, 13,986 seeded orders, at most five concurrent clients. Zero request errors; participant-dashboard P95 345.37 ms and legacy Excel-export P95 716.86 ms. Host test-client benchmark excludes Gunicorn/TLS/network and is not a 512 MiB container-capacity guarantee.

Isolated inbox workload: 20 synthetic participants, 200 deliveries, 100 unique events, 60 original creates, 20 edits and 20 cancellations. Final 40 active records/100 tiffins; 100 duplicate deliveries absorbed; zero errors. Duration 1.756 seconds, 56.94 events/second, P95 32.42 ms. It excludes actual WhatsApp delivery and container limits.

Production message-processing latency cannot be reported: the sole historical inbox entry lacks a usable processed timestamp. Publication execution time cannot be reported because no real publication has occurred. **Memory under genuine message-processing activity remains pending**; quiet connected snapshots neither prove capacity nor demonstrate a leak.

## 8. Backup, recovery and host security

Installed/native backup script hash matches source; directory 0700 and dump files 0600 were verified. Existing backup checksums passed. The installed script was executed against a separate protected output location; checksum and archive validation passed. Both an exact consistent-snapshot restore and a separate installed-script dump restore succeeded in isolated databases. All 19 production tables' counts/content hashes matched the exported snapshot, including users, messages, aliases, participants and encrypted auth. Financial history/publication tables were empty and therefore cannot demonstrate recovery of real populated history; the populated synthetic regression evidence is separate.

Restored application reconciliation and reopened Excel passed; recovered encrypted session credentials authenticated without initiating WhatsApp. No live database was restored, migrated or written by acceptance fixtures. Native read-only export and temporary isolated test databases do not change the production database architecture. See RECOVERY.md for recovery boundaries, rollback and custody of encryption secrets.

**High operational gap:** no configured application off-server destination/successful copy or independent failure alert was verified. A destination and controlled recovery-key custody are required; a local dump is not off-server disaster recovery.

**High shared-host finding:** PostgreSQL listens on all interfaces and UFW permits port 5432 from anywhere, IPv4/IPv6. Ordered HBA rules allow TiffinTrack only from localhost/dedicated app network and reject other remote access to its database/role. This protects application database authentication but does not close the network-exposed shared service. Changing global firewall/listening rules could affect other applications and requires the server owner's access inventory and approval. No global rules were changed or exposure introduced by this audit.

Monitoring distinguishes database liveness from receiver unavailable, worker unavailable, processing errors/backlog and review state. Publication history exposes delivery states; cron/host checks independently verify scheduling. The public health endpoint alone is not evidence of end-to-end order receipt. Automated off-server alert delivery and long-term uptime are not established by this audit.

## 9. Consolidated acceptance and remaining work

See ACCEPTANCE-MATRIX.csv. No new critical application-code defect was demonstrated. Technical verification and private server-side access passed; production business acceptance remains open for these specific reasons:

1. **Five genuine order records across participants:** zero currently available. Do not ask participants for unnecessary test orders. Observe normal activity and compare original group/source, stable participant, timestamp/date/quantity, database entry, dashboard and individual/group totals.
2. **Genuine edit/cancellation:** pending naturally occurring activity; synthetic lifecycle tests passed.
3. **First real weekly delivery:** pending Friday 13 November at 14:00 IST. Verify immutable snapshot totals, acknowledgement and no duplicate send through normal scheduled execution.
4. **Current browser password login:** pending access with current credentials; no password reset is justified by this audit.
5. **Real-message memory/latency and sustained reliability:** pending genuine activity; current quiet samples and isolated host workloads are explicitly limited.
6. **Off-server recovery/alerts and shared-host PostgreSQL firewall:** operational actions require destination/access details and owner-reviewed network restrictions.
7. **Public domain:** deliberately deferred; ownership/DNS approval not supplied. Preparation and record explanations are in DOMAIN-READINESS.md. Private access remains unchanged.

There were no automated test failures in the executed final suite. An initial host-inspection harness command omitted the PID column required by `docker top`; the harness was corrected and the baseline rerun successfully. This did not identify an application defect. Unexecuted real-world/browser tests above are pending, not passed. External inactivity is not a reason to rewrite the parser, relink WhatsApp or redeploy the same code.

## 10. Deliverables and rollback

The complete source ZIP includes application/bridge source, tests, migrations, Docker/deployment templates, active guidance, historical reports and sanitized final evidence. `CODE-MANIFEST.sha256` preserves production-code verification; the new complete-source manifest covers package contents including tests/docs, and the ZIP has an external SHA-256 checksum. No `.env`, credentials, keys, sessions, database dumps, installed dependencies or caches are included.

Evidence files `TiffinTrack-Final-{tests,baseline,restore,recovery,live,extra}.log`, performance JSON files and source inventory document actual execution. RECOVERY.md gives isolated restore and code rollback steps; DOMAIN-READINESS.md documents the separately approved DNS/HTTPS phase. No new deployment approval is requested because no runtime deployment is necessary.
