"""Read live Nginx and test a full cloned candidate; never install, reload or mutate it."""

import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from proxy_manager.deployment import EMPTY, Manager  # noqa: E402
from proxy_manager.storage import load  # noqa: E402
from proxy_manager.validator import validate_registry  # noqa: E402


def main():
    os.umask(0o077)
    base = Path(__file__).resolve().parents[1]
    settings = load(base / "examples/manager.support.json")
    app = load(base / "examples/tiffintrack-pilot.json")
    registry = validate_registry(
        {"version": 1, "applications": [app]}, settings["certificate_roots"]
    )
    # Retain isolated evidence for review. Nothing is placed under /etc or /opt.
    root = Path(tempfile.mkdtemp(prefix="proxy-manager-candidate-", dir="/var/tmp"))
    settings["root"] = str(root)
    settings["audit_log"] = str(root / "audit.log")
    manager = Manager(settings)
    manager.initialize()
    manager.network_preflight(registry)
    canaries = manager.canaries()
    if not all(c["ok"] for c in canaries):
        raise RuntimeError("Existing endpoint baseline failed")
    upstream = manager.health(registry, proxy=False)
    if not all(c["upstream"]["ok"] for c in upstream.values()):
        raise RuntimeError("Existing application upstream failed")
    files, manual, hashes = manager.controller.snapshot(root)
    manager.controller.conflicts(registry, manual, EMPTY)
    release = manager.new_release(registry)
    candidate = manager.controller.candidate(
        files, manager.release(release), root / "staging" / release
    )
    manager.controller.test(candidate)
    manager.controller.unchanged(hashes)
    manager.controller.test()
    print(
        json.dumps(
            {
                "candidate_valid": True,
                "nginx_reloaded": False,
                "production_files_changed": False,
                "manual_config_hashes": hashes,
                "candidate": str(candidate),
                "preservation_canaries": canaries,
                "upstream": upstream,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
