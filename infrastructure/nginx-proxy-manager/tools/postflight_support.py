"""Read-only acceptance of the approved private pilot; no login or business mutations."""

import hashlib
import http.client
import json
import re
import ssl
import subprocess
from pathlib import Path


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    context = ssl.create_default_context(cafile="/etc/tiffintrack/tls/server.crt")
    statuses = {}
    for path, expected in (
        ("/auth/login", 200),
        ("/health", 200),
        ("/", 302),
        ("/admin/orders", 302),
        ("/admin/whatsapp", 302),
        ("/admin/whatsapp/status.json", 302),
        ("/admin/whatsapp/export.xlsx", 302),
    ):
        connection = http.client.HTTPSConnection("192.168.1.101", 18443, timeout=5, context=context)
        try:
            connection.request("GET", path)
            response = connection.getresponse()
            statuses[path] = response.status
            require(response.status == expected, f"Unexpected status for {path}: {response.status}")
        finally:
            connection.close()
    connection = http.client.HTTPSConnection(
        "192.168.1.101", 18443, timeout=5, context=context, source_address=("127.0.0.1", 0)
    )
    try:
        connection.request("GET", "/auth/login")
        denied = connection.getresponse().status
        require(denied == 403, f"Source ACL did not deny: {denied}")
    finally:
        connection.close()
    connection = http.client.HTTPSConnection("192.168.1.101", 18443, timeout=5, context=context)
    rejected_host = False
    try:
        connection.request("GET", "/", headers={"Host": "unregistered.invalid"})
        try:
            connection.getresponse()
        except http.client.RemoteDisconnected:
            rejected_host = True
        require(rejected_host, "Unknown Host was not rejected")
    finally:
        connection.close()
    before = json.loads(Path("/var/tmp/proxy-manager-before.json").read_text())
    after = json.loads(Path("/var/tmp/proxy-manager-after.json").read_text())
    require(
        before["manual_and_protected_hashes"] == after["manual_and_protected_hashes"],
        "Protected file changed",
    )
    require(before["container"] == after["container"], "Container identity/configuration changed")
    ufw = subprocess.run(
        ["ufw", "status", "numbered"], check=True, capture_output=True, text=True
    ).stdout.strip()
    addition = [
        line
        for line in ufw.splitlines()
        if "18443" in line and "proxy-manager private HTTPS pilot" in line
    ]
    require(len(addition) == 1, "Missing/duplicate approved firewall rule")
    new_index = int(re.match(r"\[\s*(\d+)\]", addition[0])[1])
    original_lines = []
    for line in ufw.splitlines():
        if line == addition[0]:
            continue
        match = re.match(r"\[\s*(\d+)\]", line)
        if match and int(match[1]) > new_index:
            line = f"[{int(match[1]) - 1:2d}]" + line[match.end() :]
        original_lines.append(line)
    remaining = "\n".join(original_lines).strip()
    require(
        hashlib.sha256(remaining.encode()).hexdigest() == before["ufw_sha256"],
        "Unexpected firewall difference",
    )
    sql = """BEGIN READ ONLY;
    SELECT status, floor(extract(epoch FROM now()-heartbeat))::int AS receiver_age_seconds,
    floor(extract(epoch FROM now()-worker_heartbeat))::int AS worker_age_seconds FROM wa_connection;
    SELECT state,count(*) FROM wa_inbox GROUP BY state;
    COMMIT;"""
    query = subprocess.run(
        [
            "runuser",
            "-u",
            "postgres",
            "--",
            "psql",
            "-X",
            "-v",
            "ON_ERROR_STOP=1",
            "-d",
            "tiffintrack",
            "-A",
            "-t",
        ],
        input=sql,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    print(
        json.dumps(
            {
                "tls_verified_route_statuses": statuses,
                "denied_loopback_source_status": denied,
                "unknown_host_rejected": rejected_host,
                "protected_hashes_unchanged": True,
                "container_identity_image_health_resources_and_ports_unchanged": True,
                "only_approved_firewall_addition": True,
                "whatsapp_read_only_status": query.stdout.strip().splitlines(),
                "authenticated_login_dashboard_order_mutations": "not exercised; preservation requires no database changes",
                "lan_client_reachability": "not verified from a separate permitted LAN client",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
