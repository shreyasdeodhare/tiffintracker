# TiffinTrack 502 correction — 12 November 2026

Cause: an optional WhatsApp all-group listing request returned upstream status 500. Its rejection reached the receiver's generic fatal-task handler, and the original supervisor stopped Gunicorn when that receiver exited. This produced a container restart loop (747 observed restarts), leaving Nginx with no upstream.

Correction:
- Supervise/restart web, WhatsApp receiver and inbox worker independently in the same container, with bounded per-component backoff. Connector failures no longer stop the dashboard.
- Use the already-selected group's saved identity/name; do not request the entire group list on every reconnect.
- Run optional group discovery separately from durable message writes; contain API failures, back off, and prevent duplicate concurrent lookups.
- Record safe operation/type/status diagnostics without credentials or message content, instead of mislabeling every failure as storage failure.

Validation:
- Regression test proves a connector crash/restart does not stop Gunicorn or the inbox worker; orderly shutdown still signals live children.
- Three group-discovery tests cover selected-group API avoidance, upstream 500 containment/backoff, and concurrent refresh deduplication.
- All 17 Node tests passed; Python supervisor regression passed; Ruff, format and compile checks passed.
- Runtime returned healthy with zero container restarts after the fix and continued HTTP 200 responses through Nginx.
- The exact workstation URL https://127.0.0.1:8443/admin/whatsapp responded normally (unauthenticated access redirected to login and returned HTTP 200).

No database migration, order deletion, password reset or WhatsApp relink was used for this repair. One container, host PostgreSQL, existing port binding, 1.5 CPU and 512 MiB limit remain.

Image: tiffintrack:gateway-fix-20261112. The temporary local SSH forward was re-established for this deployment session.

Final live checks passed for the authenticated administrator dashboard and inbox worker. The ordinary member remained forbidden from pairing credentials. The receiver was connected to Amanora 507B Office The Meal Habit with no disconnect reported; observed usage was 265.2 MiB / 512 MiB and 0.41% CPU.
