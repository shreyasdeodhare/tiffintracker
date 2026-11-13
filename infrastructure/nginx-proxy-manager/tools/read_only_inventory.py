"""Sanitized baseline/post-check. No writes, requests with bodies, or service mutations."""

import datetime
import hashlib
import json
import subprocess
from pathlib import Path


def command(*args):
    return subprocess.run(
        list(args), check=True, capture_output=True, text=True, timeout=30
    ).stdout.strip()


def main():
    paths = [
        "/etc/nginx/nginx.conf",
        "/etc/nginx/sites-available/default",
        "/etc/nginx/conf.d/tiffintrack-ip.conf",
        "/home/support/tiffintrack/docker-compose.yml",
        "/home/support/tiffintrack/Dockerfile",
        "/etc/cron.d/tiffintrack",
        "/etc/cron.d/tiffintrack-summary",
    ]
    container = json.loads(command("docker", "inspect", "tiffintrack"))[0]
    state = container["State"]
    logs = subprocess.run(
        ["docker", "logs", "--since", "30m", "--tail", "200", "tiffintrack"],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    lines = (logs.stdout + logs.stderr).splitlines()
    ufw = command("ufw", "status", "numbered")
    print(
        json.dumps(
            {
                "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
                "nginx": command("systemctl", "is-active", "nginx"),
                "nginx_test_passed": subprocess.run(
                    ["nginx", "-t"], capture_output=True, timeout=30
                ).returncode
                == 0,
                "manual_and_protected_hashes": {
                    p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths
                },
                "enabled_sites": {
                    str(p): str(p.resolve()) for p in Path("/etc/nginx/sites-enabled").iterdir()
                },
                "nginx_conf_d": sorted(p.name for p in Path("/etc/nginx/conf.d").glob("*.conf")),
                "container": {
                    "id": container["Id"],
                    "image": container["Image"],
                    "configured_image": container["Config"]["Image"],
                    "running": state["Running"],
                    "health": state.get("Health", {}).get("Status"),
                    "restarts": container["RestartCount"],
                    "ports": container["HostConfig"]["PortBindings"],
                    "memory": container["HostConfig"]["Memory"],
                    "nano_cpus": container["HostConfig"]["NanoCpus"],
                    "pids_limit": container["HostConfig"]["PidsLimit"],
                },
                "addresses": [
                    {
                        "interface": a["ifname"],
                        "addresses": [i["local"] for i in a.get("addr_info", [])],
                    }
                    for a in json.loads(command("ip", "-j", "address", "show"))
                    if a["ifname"] in ("eno1", "tailscale0", "lo")
                ],
                "listening_tcp_sha256": hashlib.sha256(
                    command("ss", "-H", "-lnt").encode()
                ).hexdigest(),
                "ufw_sha256": hashlib.sha256(ufw.encode()).hexdigest(),
                "ufw_pilot_lines": [line for line in ufw.splitlines() if "18443" in line],
                "logs_sample": {
                    "lines": len(lines),
                    "component_exits": sum("exited" in line.lower() for line in lines),
                    "errors": sum(
                        any(
                            word in line.lower()
                            for word in ("traceback", "intake-error", "publication-storage-error")
                        )
                        for line in lines
                    ),
                },
                "production_app_or_database_writes": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
