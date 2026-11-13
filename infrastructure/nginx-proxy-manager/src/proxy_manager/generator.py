"""Generate only fixed, reviewed Nginx syntax from validated fields."""

MARKER = "# Owned by nginx-proxy-manager v1; do not edit.\n"


def endpoint(address, port):
    return f"[{address}]:{port}" if ":" in address else f"{address}:{port}"


def proxy_lines(app):
    settings = app["proxy_settings"]
    upstream = endpoint(app["upstream"]["host"], app["upstream"]["port"])
    lines = [
        f"proxy_pass http://{upstream};",
        "proxy_http_version 1.1;",
        "proxy_set_header Host $http_host;",
        "proxy_set_header X-Real-IP $remote_addr;",
        "proxy_set_header X-Forwarded-For $remote_addr;",
        "proxy_set_header X-Forwarded-Proto $scheme;",
        # Clear potentially spoofed alternate forwarding information.
        'proxy_set_header Forwarded "";',
        'proxy_set_header X-Forwarded-Host "";',
        'proxy_set_header X-Forwarded-Port "";',
    ]
    if app["websocket"]:
        lines += [
            "proxy_set_header Upgrade $http_upgrade;",
            "proxy_set_header Connection $proxy_manager_connection_upgrade;",
        ]
    else:
        lines += ['proxy_set_header Connection "";']
    for key in ("connect_timeout", "read_timeout", "send_timeout"):
        lines.append(f"proxy_{key} {settings[key]}s;")
    for key, value in sorted(app.get("headers", {}).items()):
        lines.append(f'proxy_set_header {key} "{value}";')
    return lines


def render_application(app):
    listener = endpoint(app["listen"]["address"], app["listen"]["port"])
    suffix = " ssl" if app.get("tls") else ""
    tls_lines = []
    if app.get("tls"):
        tls_lines = [
            f"ssl_certificate {app['tls']['certificate']};",
            f"ssl_certificate_key {app['tls']['key']};",
            "ssl_protocols TLSv1.2 TLSv1.3;",
        ]
    # Own a new explicit endpoint, never the existing manual site's default.
    lines = (
        [
            MARKER.rstrip(),
            "server {",
            f"    listen {listener}{suffix} default_server;",
            "    server_name _;",
        ]
        + ["    " + line for line in tls_lines]
        + [
            "    return 444;",
            "}",
            "server {",
            f"    listen {listener}{suffix};",
            "    server_name " + " ".join(app["server_names"]) + ";",
        ]
    )
    lines += ["    " + line for line in tls_lines]
    lines += [f"    client_max_body_size {app['proxy_settings']['max_body_size']};"]
    for network in app["allowed_networks"]:
        lines.append(f"    allow {network};")
    lines.append("    deny all;")
    rate = app.get("rate_limit")
    if rate:
        lines += [
            f"    location = {rate['path']} {{",
            f"        limit_req zone=pm_{app['name'].replace('-', '_')} burst={rate['burst']} nodelay;",
            "        limit_req_status 429;",
        ]
        lines += ["        " + line for line in proxy_lines(app)] + ["    }"]
    lines += (
        ["    location / {"] + ["        " + line for line in proxy_lines(app)] + ["    }", "}", ""]
    )
    return "\n".join(lines)


def generate(registry):
    files, common = {}, [MARKER.rstrip()]
    enabled = [app for app in registry["applications"] if app["enabled"]]
    if any(app["websocket"] for app in enabled):
        common += [
            "map $http_upgrade $proxy_manager_connection_upgrade {",
            "    default upgrade;",
            "    '' close;",
            "}",
        ]
    for app in enabled:
        rate = app.get("rate_limit")
        if rate:
            common.append(
                f"limit_req_zone $binary_remote_addr zone=pm_{app['name'].replace('-', '_')}:1m rate={rate['requests_per_minute']}r/m;"
            )
        files[f"app.{app['name']}.conf"] = render_application(app)
    files["00-manager-common.conf"] = "\n".join(common) + "\n"
    return files
