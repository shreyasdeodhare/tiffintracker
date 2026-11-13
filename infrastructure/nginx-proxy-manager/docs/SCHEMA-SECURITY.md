# Registry and privilege boundary

Registry root has exactly `version: 1` and `applications: []`, at most 100 entries. Unknown fields and duplicate JSON keys are rejected. Do not put credentials, environment variables, tokens, request bodies or private key contents in either JSON file.

| Application field | Rules |
|---|---|
| `name` | Unique lowercase letter followed by lowercase letters, digits or hyphens, max 48 characters |
| `enabled`, `websocket` | JSON booleans |
| `upstream.host` | Exact IP or DNS name; no URLs, credentials, paths or directives |
| `upstream.port` | Integer 1–65535; known infrastructure ports denied; endpoint must also be explicitly approved in protected manager settings |
| `listen.mode` | Only `private`; public mode is unsupported |
| `listen.address` | Explicit RFC1918, IPv4/IPv6 loopback, IPv6 ULA or Tailscale 100.64/10 address; must currently belong to the host |
| `listen.port` | Integer 1–65535, infrastructure ports forbidden; one application per explicit IP/port |
| `server_names` | 1–10 unique exact hostnames/IPs; wildcard/regex names forbidden |
| `allowed_networks` | 1–32 canonical CIDRs; nonempty, no `/0`; deny all follows allow directives. Include listener IP so server-local verification can pass |
| `health_check.enabled` | Boolean. Enabled applications require successful checks for deployment certification |
| `health_check.path` | Simple absolute URL path; no traversal, percent escapes, queries or fragments |
| `health_check.expected_statuses` | Nonempty list of integers 200–399; e.g. `[200,302]`. Redirects are never followed |
| `health_check.timeout` | Integer seconds 1–15 |
| `proxy_settings.max_body_size` | 1–9999 followed by lowercase `m` or `k` |
| `proxy_settings.connect_timeout/read_timeout/send_timeout` | Integer seconds 1–300 |
| `headers` | Optional, up to 12 literal `X-App-*` headers, safe nonsecret values only. Authentication/forwarding headers cannot be supplied |
| `tls` | Optional object: `certificate`, `key`, `ca_file`; absolute paths restricted to approved roots. TLS 1.2/1.3. All paths must be root-protected at preflight. CA is used to verify probes; verification is never disabled |
| `rate_limit` | Optional exact path, requests_per_minute 1–10000, burst 1–1000; HTTP 429 on excess |

The example pilot keeps the existing application's body limit and login rate policy. Arbitrary directives, custom templates, path-to-different-upstream routing, shared listeners, load balancing, upstream TLS and public ingress are deliberately unsupported in this release. Rejected configurations require review rather than automatic adoption of a manual site. Future services require only registry/settings configuration when they fit the supported model.

## Settings

`manager.json` explicitly specifies root state directory, Nginx binary/main configuration, systemctl, ss, ip, dedicated bridge, audit log, certificate roots, approved upstream objects `{host,port}` and nonempty preservation canaries `{host,port,path,statuses,tls?,ca_file?}`. These privileged settings are administered by root. The endpoint allowlist prevents accidentally proxying an internal infrastructure service on a custom port. Register only confirmed HTTP applications. DNS-backed upstreams are trusted only after administrator approval; stable local IPs are preferable.

## Security controls

Operational CLI requires root on Linux. Registry/state/log directories are root-owned 0700, files 0600; code and executable paths must be root-owned without group/world write. Every ancestor of privileged inputs is checked. No daemon, HTTP management API, extra container, password storage, package upgrade, sudoers change or firewall command is installed. Use ordinary restricted sudo access, not passwordless access granted to arbitrary scripts.

Each new listener has a default server that closes unmatched Host requests with 444, plus one named proxy server restricted by source CIDRs. Host includes the port to preserve origin/CSRF behavior. X-Real-IP and X-Forwarded-For overwrite incoming values with Nginx's client address; X-Forwarded-Proto is computed from the connection. Alternate Forwarded/X-Forwarded-Host/X-Forwarded-Port values are cleared. Registry headers cannot override these decisions. Configure any global real-IP trust independently and only for explicitly trusted proxy addresses; the manager does not add trusted proxies or infer them from request headers.

Existing global Nginx directives, including HTTP-level real-IP settings, modules, logging and timeouts, continue to apply. Review those before introducing a new external ingress. The inspected server had no real-IP trust directives. This version's private-only listener policy is not an approval for Cloudflare/public tunnel exposure.

No automatic UFW change is made. Nginx CIDR restrictions and host firewall restrictions are separate controls. A private listener is not reachable from another LAN client until the host/network firewall permits it. HTTP is unencrypted even on a LAN. Use a trusted internal CA, managed private certificate or separately reviewed Tailscale HTTPS setup for browser trust. The pilot reuses the existing self-signed certificate read-only and will show a browser trust warning unless that certificate is trusted.

Audit records UTC timestamps, operation, application name/release identifiers and error classes, never response bodies, raw configuration dumps or credentials. Logs rotate at 2 MB with five retained rotations. Existing Nginx access/error logging is inherited unchanged. Registry values are visible to root and in generated config; therefore headers must never contain secrets.

## Transaction guarantees and limitations

All CLI operations share a nonblocking OS lock. Manual administrator edits and external deployment tools do not honor it; avoid simultaneous Nginx maintenance. Before activation the manager fingerprints manual files and include membership; a detected concurrent change stops the transaction. Backups contain only owned state. Files without the expected ownership marker/manifest are never adopted or overwritten. An interrupted transaction leaves a recovery journal and blocks state-changing operations until verified rollback; health remains available.

The system cannot guarantee that an independent root administrator or external deployment tool will not change Nginx between checks. An OS crash or SIGKILL can leave the on-disk pointer and running workers at different releases; use the retained journal and rollback after recovery. It never claims success solely from a reload command. HTTP probes establish selected endpoint behavior, not complete authenticated business workflow correctness.
