# Complete delivered file inventory

All application and infrastructure files outside this independent package are unchanged by this source delivery.

| File | Purpose |
|---|---|
| `bin/proxy-manager` | Installed command entry point. |
| `docs/evidence/FILE-PURPOSES.md` | This complete per-file purpose inventory. |
| `docs/evidence/health-final.json` | Sanitized verification evidence: health-final.json. |
| `docs/evidence/installed-code-verification.json` | Sanitized verification evidence: installed-code-verification.json. |
| `docs/evidence/installed-code.sha256` | Sanitized verification evidence: installed-code.sha256. |
| `docs/evidence/proxy-manager-after.json` | Sanitized verification evidence: proxy-manager-after.json. |
| `docs/evidence/proxy-manager-before.json` | Sanitized verification evidence: proxy-manager-before.json. |
| `docs/evidence/proxy-manager-postflight.json` | Sanitized verification evidence: proxy-manager-postflight.json. |
| `docs/evidence/proxy-manager-preflight.json` | Sanitized verification evidence: proxy-manager-preflight.json. |
| `docs/evidence/tests-ubuntu.log` | Sanitized verification evidence: tests-ubuntu.log. |
| `docs/INSTALLATION.md` | Fresh-install procedure and private pilot acceptance. |
| `docs/OPERATIONS.md` | Onboarding, deployment sequence, rollback, retention and troubleshooting. |
| `docs/PRODUCTION-REVIEW.md` | Exact approved production files, settings, network change and rollback plan. |
| `docs/review-generated/00-manager-common.conf` | Exact generated pilot configuration presented for approval; not an instruction to copy over manual sites. |
| `docs/review-generated/app.tiffintrack-private.conf` | Exact generated pilot configuration presented for approval; not an instruction to copy over manual sites. |
| `docs/SCHEMA-SECURITY.md` | Exact field rules, permissions, trusted inputs and security limitations. |
| `docs/VERIFICATION.md` | Measured before/after outcomes, test results and outstanding acceptance. |
| `examples/applications.empty.json` | Safe empty desired registry for initialization. |
| `examples/manager.support.json` | Reviewed host-specific executable paths, approved upstream, canaries, certificate roots and audit location. |
| `examples/tiffintrack-pilot.json` | Reviewed private HTTPS pilot registry entry; no credentials. |
| `pyproject.toml` | Test discovery, lint and formatting configuration. |
| `README.md` | Architecture, layout, commands and release status. |
| `requirements.txt` | Documents no third-party runtime dependencies. |
| `src/proxy_manager/__init__.py` | Standalone package version. |
| `src/proxy_manager/cli.py` | Root privilege checks, argument parsing, confirmations, desired-state commands and health diagnostics. |
| `src/proxy_manager/deployment.py` | Approved settings, immutable releases, journal, backup, atomic activation and verified rollback. |
| `src/proxy_manager/generator.py` | Fixed safe Nginx syntax and isolated per-application output. |
| `src/proxy_manager/health_checker.py` | Bounded TCP/HTTP/TLS probes without response bodies or redirect following. |
| `src/proxy_manager/nginx_controller.py` | Nginx/systemd subprocess arrays, include graph and manual/socket conflicts. |
| `src/proxy_manager/registry.py` | Atomic desired registry CRUD with recursive partial updates. |
| `src/proxy_manager/storage.py` | Strict JSON, secure paths, atomic fsync writes, OS lock and rotating audit. |
| `src/proxy_manager/validator.py` | Deny-by-default registry field validation and exposure restrictions. |
| `tests/conftest.py` | Independent fixtures and package import setup. |
| `tests/test_controller.py` | Manual/OS conflicts, cloned include graph, scoped sockets, validation-before-reload and locking. |
| `tests/test_deployment.py` | Immutable releases, backups, rollback/recovery, ownership, health, confirmations and security failures. |
| `tests/test_health.py` | TCP/HTTP failures, expected redirects and verified TLS probes. |
| `tests/test_real_nginx.py` | Short-lived isolated Nginx with real HTTP forwarding, WebSocket upgrade and Host rejection. |
| `tests/test_validation.py` | Schema rejection, injection protection, generation, CRUD and partial updates. |
| `tools/install_support.py` | Guarded, explicitly approved installation and pilot operations. |
| `tools/postflight_support.py` | Read-only private route, source/Host restriction, firewall difference and WhatsApp liveness verification. |
| `tools/preflight_support.py` | Read live configuration, stage cloned candidate and validate without a production reload. |
| `tools/read_only_inventory.py` | Sanitized before/after inventory with protected configuration hashes. |
| `MANIFEST.sha256` | SHA-256 of every other delivered file for independent integrity review. |
