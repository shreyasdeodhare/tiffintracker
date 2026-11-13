# Phase 3: exact proposed production changes

Prepared and explicitly approved 13 November 2026. **Integration and the LAN-only firewall rule are now applied and verified server-side.** This document records the exact approved plan; see VERIFICATION.md for the resulting evidence. The audit-log directory moved to the manager-owned /opt directory after the permission check rejected this host’s group-writable /var/log ancestor; no existing directory permissions were altered.

## Proposed outcome

- New private HTTPS endpoint: **https://192.168.1.101:18443**.
- Explicit listener: **192.168.1.101**, TCP **18443**, no wildcard bind.
- Sources: **192.168.1.0/24**, deny all other sources.
- Existing upstream: **http://127.0.0.1:8087**, unchanged.
- Reuse existing `/etc/tiffintrack/tls/server.crt` and `server.key` read-only; self-signed cert, expires 9 February 2027. No new secrets or certificate replacements.
- Preserve body limit **1m**, connect timeout **5s**, read/send timeout **60s**, exact `/auth/login` rate **10 requests/minute**, burst **15**, excess **429**.
- Existing IP HTTPS 443 and default HTTP 80 remain untouched. WhatsApp receiver is outbound inside the existing application and needs no new inbound WebSocket setting; pilot websocket=false does not change it.

## Existing files to modify

**None.** Do not edit nginx.conf, tiffintrack-ip.conf, default site, application/DB/Redis config, Docker files, environment/secrets, cron, backups, pipelines or firewall rules already present. Graceful Nginx reload affects its configuration globally, so before/after canaries and rollback are mandatory. No application container restart is proposed.

## New files and directories

```text
/opt/nginx-proxy-manager/bin/proxy-manager
/opt/nginx-proxy-manager/src/proxy_manager/__init__.py
/opt/nginx-proxy-manager/src/proxy_manager/cli.py
/opt/nginx-proxy-manager/src/proxy_manager/deployment.py
/opt/nginx-proxy-manager/src/proxy_manager/generator.py
/opt/nginx-proxy-manager/src/proxy_manager/health_checker.py
/opt/nginx-proxy-manager/src/proxy_manager/nginx_controller.py
/opt/nginx-proxy-manager/src/proxy_manager/registry.py
/opt/nginx-proxy-manager/src/proxy_manager/storage.py
/opt/nginx-proxy-manager/src/proxy_manager/validator.py
/usr/local/sbin/proxy-manager -> /opt/nginx-proxy-manager/bin/proxy-manager
/etc/nginx/apps.d/manager.json                      # exact content: examples/manager.support.json
/etc/nginx/apps.d/applications.json                 # empty initially, then the one pilot entry
/etc/nginx/apps.d/deploy.lock
/etc/nginx/apps.d/state.json
/etc/nginx/apps.d/current -> releases/<generation>
/etc/nginx/apps.d/releases/<generation>/00-manager-common.conf
/etc/nginx/apps.d/releases/<generation>/app.tiffintrack-private.conf  # only enabled pilot release
/etc/nginx/apps.d/releases/<generation>/snapshot.json
/etc/nginx/apps.d/releases/<generation>/manifest.json
/etc/nginx/apps.d/staging/<generation>/<hashed-existing-config-filenames>.conf
/etc/nginx/apps.d/backups/<transaction>/<previous-owned-release-files>
/etc/nginx/apps.d/backups/<transaction>/desired-registry.json
/etc/nginx/apps.d/pending.json                      # only during activation/recovery
/etc/nginx/conf.d/90-proxy-manager.conf
/opt/nginx-proxy-manager/logs/audit.log               # five automatic rotated audit logs maximum
```

Release and transaction identifiers are generated UUID hex values. Staging copies include existing configuration text solely for validation; originals are not edited. State/registry/staging/backups/logs are root-only. Code is root-owned without group/world write.

## Exact include addition

New `/etc/nginx/conf.d/90-proxy-manager.conf`:

```nginx
# Owned by nginx-proxy-manager v1; do not edit.
include /etc/nginx/apps.d/current/*.conf;
```

No existing-file diff. See `review-generated/` for the exact proposed application/common configuration. See `examples/tiffintrack-pilot.json` for the complete registry entry. Each listener has an unknown-Host rejection server plus the CIDR-restricted application server.

## Proposed firewall addition, separately reviewable

UFW currently denies new inbound traffic by default and has no 18443 allow rule. For actual LAN client reachability, add exactly:

```sh
ufw allow in on eno1 proto tcp from 192.168.1.0/24 to 192.168.1.101 port 18443 comment 'proxy-manager private HTTPS pilot'
```

Without this approval the manager can be installed/tested locally, but LAN client access must not be claimed. No IPv6/public/Tailscale rule is requested. Existing broad PostgreSQL/Redis rules are a preexisting host risk; they are reported but will not be changed by this task.

## Conflicts and verification

The inspected port 18443 is free. 80/443 are occupied and will not be adopted. The complete staged candidate passed the installed Nginx parser while existing site hashes stayed identical. Existing upstream/login/default/protected-route response canaries passed. Recheck immediately before installation because IP/ports/config may change.

The self-signed cert requires client trust. Login and dashboard will be checked through the new origin, but successful login HTML alone does not prove authenticated workflow acceptance. WhatsApp message placement, billing/business totals, session behavior and denied-source LAN checks remain explicitly unverified until tested by an authorized client. No production data will be created merely to test proxy installation.

## Rollback

Apply retains the prior immutable owned release plus a backup. Failed validation before activation performs no live switch. Failed reload or HTTP verification automatically restores the previous pointer, validates, gracefully reloads and verifies prior canaries; failure to verify is reported and the journal is retained. Explicit rollback restores the previous known-good release without changing application data or manual sites.

For complete integration withdrawal, remove only the new bridge from the active include pattern after verifying its exact ownership; validate/reload and verify original endpoints. Preserve installed code/state/backups for review. If the optional firewall rule was added, separately remove only that exact new rule. Never overwrite or delete a preexisting site/configuration.

**Approval received and executed:** install this standalone manager, add this one include, gracefully reload Nginx and apply this private TLS pilot. Optional second approval: add the one source-restricted UFW rule so LAN clients can reach it. No public deployment, application change, container restart, database operation or existing-site edit is included.
