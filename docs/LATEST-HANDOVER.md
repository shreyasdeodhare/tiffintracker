# TiffinTrack latest source handover - 13 November 2026

This delivery adds the independent host-level Nginx manager and a complete project file reference. All files already on master at b5935742446b2ad58c90169398c3dbdb090a569c are preserved byte-for-byte. Application code, models, migrations, APIs, Dockerfile, Compose, dependencies, WhatsApp connector, background workers and schedule templates are unchanged.

## Current production structure

One existing TiffinTrack application container remains healthy: tiffintrack:stabilization-20261112, host 127.0.0.1:8087 -> container 8000, 512 MiB, 1.5 CPUs and 128 PIDs. PostgreSQL, Nginx and cron remain host services. The new proxy manager is a standard-library Python CLI on the host, not another container.

The approved private endpoint is https://192.168.1.101:18443, restricted to 192.168.1.0/24 with the single approved eno1 UFW rule. Existing IP HTTPS 443 and access routes remain intact. The existing self-signed certificate is reused read-only. This is private hosting; public DNS/Cloudflare backend hosting remains deferred. The isolated website under website/ is a separate static landing page and does not run the Flask application.

## Preserved application behavior

The existing linked WhatsApp account receives messages for the selected group. Participant identities, order interpretation, review, edits, cancellations, calculations and Excel exports retain their current implementation. Weekly-only group publication remains Friday 14:00 Asia/Kolkata, covering the previous Saturday through Friday's snapshot cutoff. Each participant line reports that individual's order records, quantity and amount. Unresolved reviews produce the existing provisional report behavior. No daily publication was introduced. Retired automatic billing email was not re-enabled; historical compatibility files remain preserved.

## New infrastructure

infrastructure/nginx-proxy-manager/ contains the registry schema, validator, generator, controller, TCP/HTTP/TLS probes, atomic release deployment, backups, verified rollback, locking, audit logs, CLI, examples, tests and complete operations/security documentation. Its README is the entry point. The support-host examples contain paths and approved endpoints only, never credentials. Installation tools are guarded operator procedures, not automatic application deployment hooks.

Production manager code was matched by SHA-256 to the delivered source. Eighty Ubuntu tests passed, including isolated Nginx, expected redirects, forwarding headers, WebSocket upgrade and rollback failure/recovery tests. Existing protected configuration hashes, container identity/image/ports/resource limits and cron hashes matched the baseline. Existing and new HTTP/TLS canaries passed. WhatsApp was connected with fresh receiver/worker heartbeats at the latest live recheck.

## Verification limits

A separate allowed LAN client has not established reachability from this workstation. Actual authenticated password login, dashboard business actions, production order mutations, live WhatsApp messages and future scheduled publication were not rerun as proxy acceptance tests. No production test orders or database changes were made. The infrastructure evidence establishes preservation and selected endpoint/liveness behavior, not a new guarantee of complete order interpretation or message delivery.

## Complete file reference and package

TiffinTrack-Latest-Complete-File-Reference.pdf explains each source/document/evidence file, its purpose and full textual contents. Binary PDF artifacts are described instead of reproducing their bytes. Duplicate filenames are distinguished by responsibility. Historical guides and the prior PDF are retained and identified; they do not authorize applying obsolete instructions.

The downloadable ZIP is a complete snapshot of the Git tree, including the application, website, proxy manager, documentation and tests. It excludes .git, dependencies/caches, production .env, credentials, linked-device/session material, database dumps and runtime application data. The PDF's companion coverage record and source manifest allow independent file-by-file review. A GitHub push of these additions does not restart or redeploy the support application.
