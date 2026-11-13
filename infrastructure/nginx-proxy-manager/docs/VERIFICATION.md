# Implementation and production verification — 13 November 2026

## Result and scope

The standalone manager is installed. The approved private HTTPS pilot is active at **https://192.168.1.101:18443**, proxied to the unchanged **127.0.0.1:8087** application binding. The single approved LAN-only UFW rule is installed. This does not provide public Internet hosting, DNS or Cloudflare ingress.

Server-side implementation and integration are verified. **Separate LAN-client reachability and full authenticated business-workflow acceptance remain outstanding.** No claim of 100% business-function verification is made. The existing application source, manual sites, Docker deployment, database schema/data, Redis, environment, WhatsApp code, scheduled jobs and backups were not changed by the installation.

## Automated tests

- Ubuntu Python 3.12: **80 passed**, no skips, 3.83 seconds on final regression run.
- Ruff check and formatting checks passed for the final package; complete output is recorded in the delivery evidence.
- Windows earlier run: 60 passed, 17 skipped; skipped tests require Linux symlinks/include behavior or Nginx. Later added Linux-specific regression tests were executed on Ubuntu.
- Unit integration tests mock system commands and use temporary files. Coverage includes registry CRUD, partial updates, all required validation categories, generation, WebSockets, source restrictions, disabled apps, manual/OS port conflicts, duplicate listener/name scopes, backup retention, failed staged validation, failed reload, failed HTTP health, verified rollback, failed rollback journal retention, interrupted recovery, tampering, confirmation and exclusive lock. Health diagnostics remain available during interrupted transactions or invalid desired registry state.
- An isolated real Nginx process confirmed upstream forwarding, Host including port, spoofed-forwarding-header replacement, 302 redirects, WebSocket 101 upgrade and unknown-Host rejection. Its own loopback ephemeral port, PID, configuration and logs were used; the system Nginx was not signaled by tests.
- Full candidate validation cloned the live include graph and ran the installed `nginx -t -c` before creating the production bridge or activating the pilot.
- Existing TiffinTrack application regression suites were not rerun against production: no application files/dependencies changed, and production DB-write testing was prohibited. The previous app-suite results are not presented as a new run.

## Before and after

Baseline: **17:16:38 IST**. Initial post-deployment inventory: **17:22:41 IST**. Additional source/Host/API/heartbeat checks followed. See JSON evidence for exact timestamps/results.

| Check | Before | After |
|---|---|---|
| Nginx service | active | active |
| Live nginx -t | pass | pass |
| Main nginx.conf | SHA-256 recorded | identical |
| Existing tiffintrack-ip.conf | SHA-256 recorded | identical |
| Existing default site / enabled symlink | recorded | identical |
| Existing compose / Dockerfile | SHA-256 recorded | identical |
| Billing/backup and summary cron files | SHA-256 recorded | identical |
| TiffinTrack container ID | dc643926… | identical |
| Image | tiffintrack:stabilization-20261112 | identical |
| Image content ID | sha256:7881a324… | identical |
| Health / running | healthy / true | healthy / true |
| Container restarts | 0 | 0 |
| Mapping | 127.0.0.1:8087 → 8000 | identical |
| Memory / CPU / PID limit | 512 MiB / 1.5 CPUs / 128 | identical |
| Default HTTP 80 | 200 | 200 |
| Original upstream / | 302 | 302 |
| Original upstream /health | 200 | 200 |
| Original HTTPS /auth/login | 200 | 200 |
| Original HTTPS /admin/orders unauthenticated | 302 | 302 |
| New HTTPS /auth/login and /health | absent | 200, certificate verified |
| New HTTPS / and protected orders/WhatsApp/status/export routes | absent | expected 302 authentication redirects |
| New listener binding | absent | 192.168.1.101:18443 only |
| Disallowed source 127.0.0.1 on new listener | absent | 403 |
| Unregistered Host on new listener | absent | closed connection (444) |
| WhatsApp state | connected in read-only inspection | connected, receiver age 2 s / worker age 0 s |
| WhatsApp inbox | 3 processed, no pending/errors | same at final read-only check |
| Sampled logs | 0 component exits / targeted errors | same; 111 sampled lines in initial post-inventory |
| Firewall | baseline hash | exactly one approved rule added |

Firewall comparison removes the exact newly added rule and compensates for UFW renumbering subsequent rules, then verifies the reconstructed original numbered-status SHA-256 against the baseline. Existing UFW policies/rules were not modified. Rule number 195 at installation is informational; numbers may change later. Match the full dedicated rule, never a stale number.

New rule: `192.168.1.0/24 → 192.168.1.101:18443/tcp on eno1`, comment `proxy-manager private HTTPS pilot`. No wildcard listener, public rule or new Tailscale rule was created.

## Deployment records

- New include: `/etc/nginx/conf.d/90-proxy-manager.conf`, root:root 0600.
- State directory: `/etc/nginx/apps.d`, root:root 0700.
- Audit directory: `/opt/nginx-proxy-manager/logs`, root:root 0700.
- Successful pilot release: `dc8583c055b4485a98558bdcec2f1eb4`.
- Retained prior-owned-state backup: `/etc/nginx/apps.d/backups/3c916aca11174f6e9d7a0891f9a13fc1`.
- Current registry has one enabled app, `tiffintrack-private`, and reports `configuration: applied`.
- No app-container restart, database migration/write command, package installation, source push or public-hosting change was performed.

## Issues found and corrected before pilot activation

1. Root path validation rejected group-writable `/var/log` on this host. Installation stopped before an Nginx change. The new manager log path was moved into its own protected `/opt` tree; existing directory permissions were preserved. The first unsuccessful installation audit was retained as `install-initial-attempt.log`. The newly created empty old log directory was removed after preserving that file. Error reporting was strengthened so an insecure log path is never opened after failed protection checks; regression test added.
2. Dry-run with the now-installed bridge exposed a candidate-graph edge case: the bridge's newly staged include was being re-expanded as an old graph member. Dry-run refused activation. The bridge is now substituted directly; regression test added and suite rerun before installation of the fix and live apply.
3. Linux `ss` returned interface-scoped IPv4/IPv6 endpoints. The parser was updated to interpret these conservatively, with regression coverage.
4. The first firewall report compared raw UFW numbering. Adding a new IPv4 rule shifts subsequent IPv6 rule indices; the verifier now reconstructs the prior indices before comparing the unchanged original rules. This was a reporting correction; no additional firewall mutation was made.

## Limits and remaining acceptance

- A direct TCP probe from this Windows workstation to 192.168.1.101:18443 failed. Its permitted LAN connectivity could not be established, so **reachability from an independent LAN client is not verified**. Server-local TLS checks and the specific listener/firewall rule are verified. Try the URL from an authorized 192.168.1.x client.
- Browser login with password, authenticated dashboard calculations, authenticated API responses, report download contents and order create/edit/cancel were not exercised. Login can write authentication-attempt records; creating test orders would modify production data, which this task prohibits. Existing authentication redirects and login-page delivery were verified.
- No WhatsApp message was sent, order injected or scheduled publication triggered as a proxy test. Connection/heartbeats/inbox were inspected using an explicit PostgreSQL read-only transaction. Existing worker/cron configuration was preserved. These checks establish liveness, not a new business-order acceptance test.
- Automatic rollback and interrupted recovery passed isolated mocked tests. An intentionally failed production deployment was not induced. The successful apply created a real backup and passed actual production health verification.
- The reused certificate is self-signed, with existing SANs and expiry 9 February 2027. Server probes verify it against the existing public certificate. Browser trust remains a client responsibility until a separately approved CA/certificate plan.
- Existing broad database/Redis UFW rules remain a preexisting host risk, unchanged by this enhancement. Public exposure and DNS remain outside this approved change.

## Evidence and file review

`docs/evidence/` contains sanitized before, after, preflight, postflight and health JSON, test/lint output, installed-code verification, final listener/permissions and a complete package file-purpose inventory. None contains application passwords, tokens, environment files, WhatsApp session material or private key contents. `MANIFEST.sha256` identifies every delivered source/document/evidence file. Production-generated directories are not bundled; existing app code/data are not copied into this independent manager package.
