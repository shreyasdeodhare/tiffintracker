# Operations, onboarding and recovery

## Add a service

1. Verify its existing HTTP upstream and a currently assigned private listener IP. Do not change the application binding.
2. Root adds the exact `{host,port}` to `manager.json` approved_upstreams after confirming it is an HTTP app, not a database or management service. No code change is required.
3. Copy the example app JSON, choose a unique name and unused explicit private IP/port, exact server_names and minimal source CIDRs. Preserve application body/timeouts/authentication requirements. Use TLS when secure cookies require it.
4. `sudo proxy-manager add --file /root/new-application.json`
5. `sudo proxy-manager show example-api`, then `sudo proxy-manager apply --dry-run`.
6. Review generated files and planned exposure. Open only the necessary firewall rule after separate approval.
7. `sudo proxy-manager apply`, then `sudo proxy-manager health`. Verify a real client and required authenticated workflows.

Example partial update: `{"proxy_settings":{"read_timeout":90}}`. Unspecified fields remain intact. `null` may remove optional TLS/rate policy only if that is the reviewed desired state; never downgrade an authenticated app to HTTP without checking secure-cookie requirements.

Disable or remove edits desired state first, then dry-run/apply. Only manager-owned listener configurations disappear; upstream application processes, data, manual Nginx sites and backups are untouched. After removal, root may separately review whether a dedicated UFW rule should be removed; the CLI does not alter it.

## Deployment sequence

The manager verifies its bridge/release ownership, validates desired state and endpoint approval, checks assigned IPs/certificate permissions and proves current endpoint health. It probes candidate upstreams, reads `nginx -T`, detects manual/OS socket conflicts, stages an immutable release and clones the full include graph. Only the manager include is redirected in that candidate. It runs `nginx -t -c <candidate>` before a live switch.

Apply then creates a unique backup, writes a pending journal, atomically switches current, rechecks manual-file stability, runs live `nginx -t`, gracefully reloads, checks systemd activity and probes application upstream/proxy plus preservation canaries. State records success only after all checks. Failed activation restores the previous immutable release, validates, reloads and verifies its health. Any unverified rollback keeps the journal and returns failure.

`reload` validates active configuration and checks active release health. It does not apply pending registry edits. Nginx reload is graceful; existing application containers are never restarted. An unsuccessful reload operation is reported as failure; investigate configuration before retrying.

## Rollback

`sudo proxy-manager rollback` restores the previous successful manager-owned release, or the journal's previous release after interruption. It validates the full live configuration before reloading and verifies restored HTTP behavior. Registry desired state intentionally remains unchanged for review; reapplying without correcting it can repeat the failure. The previous/current release IDs are recorded in state.json, and immutable releases plus uniquely named backups are retained.

If rollback reports NOT verified, do not run repeated apply commands. Inspect `sudo proxy-manager health`, protected audit logs, `sudo nginx -t` and `sudo journalctl -u nginx`. Correct unrelated manual changes through their owner; the manager will not overwrite them. Resolve missing IPs/certificates or upstream faults, then retry rollback. A corrupt current release or ownership violation requires a root operator to review backup manifests before restoring any pointer. Never bypass ownership checks by marking an unknown file as owned.

For complete removal of the *new integration*, first apply an empty enabled-app set and verify original sites. Move only the exact verified manager bridge to a protected recovery location, run `nginx -t`, then gracefully reload and verify existing canaries. Restore that bridge and validate/reload if removal fails. Retain code, releases, logs and backups until reviewed; no recursive deletion or manual-site removal is needed. Any firewall rule removal is separate and requires the exact dedicated rule to be reviewed.

## Troubleshooting

| Symptom | Action |
|---|---|
| Lock held | Another manager operation is running. Do not delete the lock to bypass OS locking |
| Pending transaction | Run rollback after inspecting logs; new deployments remain blocked; health remains available for diagnosis |
| Upstream not approved | Confirm HTTP ownership, then root updates protected settings |
| Duplicate/manual listener | Choose an unused explicit IP/port; do not disable the manual site |
| Assigned IP missing | Reconfirm current LAN/Tailscale address; review registry and certificate SAN |
| Browser certificate warning | Trust an approved internal CA/certificate; never disable TLS probe verification |
| 403 | Check client source CIDR, selected server name and proxy chain |
| 444 / closed connection | Host is not one of the exact registered server names |
| LAN timeout but local check works | Review UFW/network rules separately; CLI does not open ports |
| 502 | Check original upstream TCP/HTTP health; do not restart the app as a routine proxy action |
| Login/CSRF failure | Preserve HTTPS, Host including port and application origin rules; stop/rollback before app changes |
| Candidate validation failure | No live switch occurred. Inspect staged root-protected config and run nginx -t -c on it for local detail |
| Nginx active but HTTP probe fails | Service activity alone is insufficient; apply fails and triggers rollback |

Backups have no automatic retention deletion. Monitor `/etc/nginx/apps.d` disk use. Archive reviewed inactive releases/staging offline during a locked maintenance window; never remove current, state.previous or journal-referenced releases. Audit logs rotate automatically without changing system cron/logrotate.
