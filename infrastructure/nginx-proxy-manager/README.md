# Nginx Proxy Manager CLI

A standalone Python 3.12 standard-library CLI for an existing Ubuntu Nginx installation. It manages HTTP applications through a JSON registry. It does not replace Nginx, start a daemon, install a container, or touch application deployments.

**Release status:** installed on the support host after explicit Phase 3 approval on 13 November 2026. Private pilot: `https://192.168.1.101:18443`, allowed source subnet `192.168.1.0/24`. Server-side TLS, HTTP and preservation checks passed. Separate LAN-client reachability and authenticated business workflows remain unverified; see `docs/VERIFICATION.md`.

## Architecture

`applications.json → validator → generator → immutable release → complete Nginx candidate test → atomic current symlink → nginx -t → graceful reload → HTTP verification`

The CLI separates desired state from applied state. Add, update, enable, disable and remove change only the registry. `apply` activates changes after confirmation. `apply --dry-run`, `generate` and `test` stage and validate a complete candidate without changing the active link or reloading. Staging and audit files are the only dry-run writes.

Files are split by responsibility:

| File | Purpose |
|---|---|
| `bin/proxy-manager` | Executable entry point; resolves installed package relative to itself |
| `src/proxy_manager/cli.py` | Argument parsing, explicit confirmations, serialized operations and command output |
| `registry.py` | Strict loading and validated atomic CRUD with recursive partial updates |
| `validator.py` | Schema, safe names/hosts/paths/CIDRs, bounds and exposure restrictions |
| `generator.py` | Fixed Nginx syntax, per-app configuration and optional WebSocket/rate-limit helpers |
| `storage.py` | Strict JSON, protected paths, durable atomic file writes, lock and rotating audit log |
| `nginx_controller.py` | Fixed subprocess arguments, conflict detection and cloned complete include graph |
| `health_checker.py` | TCP/HTTP checks, expected redirects and certificate-verified HTTPS |
| `deployment.py` | Immutable releases, manifests, backups, journal, activation and verified rollback |
| `__init__.py` | Package version |
| `tests/` | Schema, generation, CRUD, preservation, failure/recovery, locking and isolated real-Nginx tests |
| `tools/preflight_support.py` | Server-specific read-only production preflight; stages only under `/var/tmp` |
| `tools/read_only_inventory.py` | Sanitized baseline/post-check inventory, protected-file hashes and service/container state |
| `tools/postflight_support.py` | Read-only route, source-denial, firewall-difference and WhatsApp heartbeat acceptance |
| `tools/install_support.py` | Explicitly approved, guarded support-host installation and pilot steps |
| `examples/` | Empty registry, proposed support-host settings and proposed TiffinTrack pilot |
| `pyproject.toml` | Test and lint settings, no runtime dependencies |
| `requirements.txt` | Documents the dependency-free runtime |
| `docs/` | Installation, operations, schema, security, production review and verification evidence |

## Installed layout

```text
/opt/nginx-proxy-manager/bin/proxy-manager
/opt/nginx-proxy-manager/src/proxy_manager/*.py
/usr/local/sbin/proxy-manager -> /opt/nginx-proxy-manager/bin/proxy-manager
/etc/nginx/apps.d/manager.json
/etc/nginx/apps.d/applications.json
/etc/nginx/apps.d/current -> releases/<generation>
/etc/nginx/apps.d/releases/<generation>/{app.*.conf,00-manager-common.conf,snapshot.json,manifest.json}
/etc/nginx/apps.d/backups/<transaction>/
/etc/nginx/apps.d/staging/<generation>/
/etc/nginx/apps.d/{state.json,deploy.lock,pending.json when recovering}
/etc/nginx/conf.d/90-proxy-manager.conf
/opt/nginx-proxy-manager/logs/audit.log
```

Immutable releases and one symlink replace a mutable generated directory so all application configurations switch together. Nginx's existing `conf.d/*.conf` include loads the one new bridge. No edit to `nginx.conf` or existing sites is required on the inspected host. Backups and staged evidence are retained; monitor disk use and archive reviewed inactive evidence manually. There is no automatic backup deletion.

## Commands

Run operational commands with `sudo proxy-manager`. The settings default is `/etc/nginx/apps.d/manager.json`; override with `--settings /approved/manager.json` before the subcommand.

```sh
sudo proxy-manager list
sudo proxy-manager add --file /root/new-application.json
sudo proxy-manager update example-api --file /root/partial-update.json
sudo proxy-manager enable example-api
sudo proxy-manager disable example-api
sudo proxy-manager remove example-api                  # asks for confirmation
sudo proxy-manager validate
sudo proxy-manager show example-api
sudo proxy-manager generate                           # complete staged test
sudo proxy-manager test                               # complete staged test
sudo proxy-manager apply --dry-run
sudo proxy-manager apply                              # asks for confirmation
sudo proxy-manager reload                             # validates first; asks for confirmation
sudo proxy-manager health
sudo proxy-manager rollback                           # asks for confirmation
```

`--yes` is available for reviewed apply/reload/rollback/remove operations. It is not a bypass for validation, conflicts or health failures. A successful process exits 0; validation/operational failures exit 1. Health returns 1 if Nginx, an enabled app's upstream/proxy or a preservation canary fails. Output is JSON; errors go to stderr. An enabled app with health checking disabled cannot be certified by apply; keep it disabled in the registry until its health rule is defined.

See [operations](docs/OPERATIONS.md), [schema and security](docs/SCHEMA-SECURITY.md), [installation](docs/INSTALLATION.md), [production review](docs/PRODUCTION-REVIEW.md) and [verification](docs/VERIFICATION.md).

## Development

The runtime requires only Ubuntu Python 3.12 and existing Nginx/systemd/iproute2. In a separate development virtual environment install pytest and Ruff, then run:

```sh
python -m pytest -q
ruff check .
ruff format --check .
```

Linux is the supported operational platform. Windows skips Linux atomic-deployment/include-graph tests. The real-Nginx test uses its own short-lived process, loopback ephemeral port, config, PID and logs; it never signals the system Nginx or contacts production applications. Unit integration tests mock all system commands and use temporary state.

Nginx behavior is based on the official [command options](https://nginx.org/en/docs/switches.html), [WebSocket proxying](https://nginx.org/en/docs/http/websocket.html) and [proxy module](https://nginx.org/en/docs/http/ngx_http_proxy_module.html) documentation.
