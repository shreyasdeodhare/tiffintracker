"""Explicitly approved support-host installation. No app, DB, Docker or cron mutations."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def run(*arguments):
    result = subprocess.run(list(arguments), capture_output=True, text=True, timeout=120)
    if result.returncode:
        raise RuntimeError(f"{arguments[0]} failed: {result.stderr.strip()}")
    return result.stdout


def baseline():
    before = json.loads(Path("/var/tmp/proxy-manager-before.json").read_text())
    for name, checksum in before["manual_and_protected_hashes"].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != checksum:
            raise RuntimeError(f"Protected baseline file changed: {name}")
    container = json.loads(run("docker", "inspect", "tiffintrack"))[0]
    if (
        container["Id"] != before["container"]["id"]
        or container["Image"] != before["container"]["image"]
        or container["RestartCount"] != before["container"]["restarts"]
    ):
        raise RuntimeError("Existing application container changed")
    if container["State"].get("Health", {}).get("Status") != "healthy":
        raise RuntimeError("Existing application is not healthy")
    run("/usr/sbin/nginx", "-t")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["install", "pilot"])
    parser.add_argument("--approved", action="store_true", required=True)
    args = parser.parse_args()
    if os.geteuid() != 0:
        raise RuntimeError("Root is required")
    os.umask(0o077)
    baseline()
    source = Path(__file__).resolve().parents[1]
    if args.phase == "install":
        code = Path("/opt/nginx-proxy-manager")
        root = Path("/etc/nginx/apps.d")
        link = Path("/usr/local/sbin/proxy-manager")
        bridge = Path("/etc/nginx/conf.d/90-proxy-manager.conf")
        logs = Path("/opt/nginx-proxy-manager/logs")
        for path in (code, root, link, bridge, logs):
            if path.exists() or path.is_symlink():
                raise RuntimeError(f"Refusing existing installation target: {path}")
        # Revalidate the full hypothetical include before creating its live bridge.
        run("/usr/bin/python3", str(source / "tools/preflight_support.py"))
        (code / "bin").mkdir(parents=True, mode=0o755)
        (code / "src/proxy_manager").mkdir(parents=True, mode=0o755)
        for directory in (code, code / "bin", code / "src", code / "src/proxy_manager"):
            directory.chmod(0o755)
        for item in (source / "src/proxy_manager").glob("*.py"):
            target = code / "src/proxy_manager" / item.name
            shutil.copyfile(item, target)
            target.chmod(0o644)
        shutil.copyfile(source / "bin/proxy-manager", code / "bin/proxy-manager")
        (code / "bin/proxy-manager").chmod(0o755)
        link.symlink_to(code / "bin/proxy-manager")
        root.mkdir(mode=0o700)
        logs.mkdir(mode=0o700)
        shutil.copyfile(source / "examples/manager.support.json", root / "manager.json")
        (root / "manager.json").chmod(0o600)
        print(run(str(link), "initialize"))
        sys.path.insert(0, str(code / "src"))
        from proxy_manager.deployment import Manager, settings_from

        manager = Manager(settings_from(root / "manager.json"))
        with bridge.open("x", encoding="utf-8") as stream:
            stream.write(manager.bridge())
            stream.flush()
            os.fsync(stream.fileno())
        try:
            print(run(str(link), "reload", "--yes"))
            baseline()
        except Exception:
            recovery = root / "bridge.recovery"
            if bridge.read_text() != manager.bridge():
                raise RuntimeError("Bridge changed externally; manual recovery required") from None
            bridge.rename(recovery)
            run("/usr/sbin/nginx", "-t")
            run("/usr/bin/systemctl", "reload", "nginx")
            manager.verified_health({"version": 1, "applications": []})
            raise
        print("EMPTY MANAGER INTEGRATION VERIFIED; APPLICATION UNCHANGED")
    else:
        cli = "/usr/local/sbin/proxy-manager"
        print(run(cli, "add", "--file", str(source / "examples/tiffintrack-pilot.json")))
        print(run(cli, "apply", "--dry-run"))
        print(run(cli, "apply", "--yes"))
        baseline()
        print(run(cli, "health"))
        print("PRIVATE PILOT VERIFIED LOCALLY; FIREWALL UNCHANGED BY THIS SCRIPT")


if __name__ == "__main__":
    main()
