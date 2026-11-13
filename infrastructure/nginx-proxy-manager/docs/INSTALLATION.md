# Installation after explicit Phase 3 approval

This is the reference procedure for a reviewed fresh installation. The support host has now been installed using the guarded installer and a verified resume after its permission check stopped the first attempt; do not rerun initialization on that existing installation. See VERIFICATION.md. Run fresh installations from a reviewed release directory. Recheck IP, port availability, Nginx validation, current health and file absence immediately before installation. Stop if any target already exists; do not overwrite another installation. Review scripts before running as root. There is no dependency installation or app-container action.

## 1. Install standalone root-owned code and protected empty state

The release must be copied to a controlled directory and checked against its supplied SHA-256 manifest. Then root installs only the listed code:

```sh
sudo test ! -e /opt/nginx-proxy-manager
sudo test ! -e /etc/nginx/apps.d
sudo test ! -e /usr/local/sbin/proxy-manager
sudo test ! -e /etc/nginx/conf.d/90-proxy-manager.conf
```

All four tests must succeed before continuing. During a serialized root maintenance session:

```sh
sudo install -d -o root -g root -m 0755 /opt/nginx-proxy-manager/bin /opt/nginx-proxy-manager/src/proxy_manager
sudo install -o root -g root -m 0755 bin/proxy-manager /opt/nginx-proxy-manager/bin/proxy-manager
sudo install -o root -g root -m 0644 src/proxy_manager/*.py /opt/nginx-proxy-manager/src/proxy_manager/
sudo ln -s /opt/nginx-proxy-manager/bin/proxy-manager /usr/local/sbin/proxy-manager
sudo install -d -o root -g root -m 0700 /etc/nginx/apps.d /opt/nginx-proxy-manager/logs
sudo install -o root -g root -m 0600 examples/manager.support.json /etc/nginx/apps.d/manager.json
sudo proxy-manager initialize
```

`initialize` creates empty registry/state, releases/staging/backups directories, a current release and audit log. It does not create the Nginx include, expose a listener or reload Nginx.

## 2. Validate and add the one new include

Run `tools/preflight_support.py` with root using the reviewed release to validate the prospective full include graph without live changes. The support-host `conf.d/*.conf` inclusion was verified; no main configuration edit is needed. Create a new file only, using exclusive creation in a root maintenance session:

```python
import os

os.umask(0o077)
with open("/etc/nginx/conf.d/90-proxy-manager.conf", "x", encoding="utf-8") as stream:
    stream.write(
        "# Owned by nginx-proxy-manager v1; do not edit.\ninclude /etc/nginx/apps.d/current/*.conf;\n"
    )
    stream.flush()
    os.fsync(stream.fileno())
```

Run `sudo nginx -t`. If it fails, move **only that newly created bridge** to a protected recovery location, revalidate the unchanged original configuration, and stop; do not reload invalid configuration. If it succeeds, `sudo proxy-manager reload` validates again, gracefully reloads and checks existing endpoints. If reload or checks fail, restore the previous include state, validate and gracefully reload, then verify original endpoints. Do not proceed to the pilot until the empty integration is confirmed healthy.

## 3. Register and apply the private TLS pilot

```sh
sudo proxy-manager add --file /reviewed-release/examples/tiffintrack-pilot.json
sudo proxy-manager show tiffintrack-private
sudo proxy-manager apply --dry-run
sudo proxy-manager apply
sudo proxy-manager health
```

Replace `/reviewed-release` with the actual controlled release directory. The deployment prompts before changes. The proposed endpoint is `https://192.168.1.101:18443`; source allowlist is `192.168.1.0/24`. Existing `https://192.168.1.101/`, Tailscale HTTPS and `http://127.0.0.1:8087` remain as before. UFW remains unchanged unless the following separate rule is explicitly approved.

## 4. Optional, separately approved LAN firewall rule

```sh
sudo ufw allow in on eno1 proto tcp from 192.168.1.0/24 to 192.168.1.101 port 18443 comment 'proxy-manager private HTTPS pilot'
```

No public rule, default policy change, database rule, Docker rule or Tailscale rule is proposed. Record the exact added rule. If pilot deployment fails, remove only that exact new rule after reviewing its current number or equivalent exact rule specification; retain every preexisting rule.

## 5. Acceptance

Repeat the baseline inventory and protected-file hashes; compare Nginx manual sites, container ID/image/health/ports/restart count, workers, cron and firewall rules. Test the new endpoint locally with trusted certificate verification, then from an allowed LAN client. Check denied source/unknown Host behavior without exposing the listener publicly. Complete authenticated login/dashboard/API and user-owned WhatsApp workflow acceptance; those cannot be inferred from login-page availability. Stop and roll back if any regression is observed.

For future hosts, replace the settings and registry with reviewed host-specific configuration; do not copy the support host IP/certificate/upstream assumptions. Confirm the bridge location is covered by an existing HTTP include and choose an unused explicit private listener before installation.
