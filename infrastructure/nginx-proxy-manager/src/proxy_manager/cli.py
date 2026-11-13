"""Root-only operational CLI. Input files never execute commands or templates."""

import argparse
import json
import os
import sys
from pathlib import Path

from .deployment import Manager, settings_from
from .generator import generate
from .registry import edit
from .storage import load, lock, protected
from .validator import Error


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--settings", default="/etc/nginx/apps.d/manager.json")
    commands = result.add_subparsers(dest="command", required=True)
    for name in ("list", "validate", "generate", "test", "health", "initialize"):
        commands.add_parser(name)
    add = commands.add_parser("add")
    add.add_argument("--file", required=True, help="JSON object describing one application")
    update = commands.add_parser("update")
    update.add_argument("name")
    update.add_argument(
        "--file",
        required=True,
        help="JSON merge patch; unspecified fields are preserved",
    )
    for name in ("enable", "disable", "remove", "show"):
        command = commands.add_parser(name)
        command.add_argument("name")
        if name == "remove":
            command.add_argument("--yes", action="store_true")
    for name in ("apply", "rollback", "reload"):
        command = commands.add_parser(name)
        command.add_argument("--yes", action="store_true")
        if name == "apply":
            command.add_argument("--dry-run", action="store_true")
    return result


def confirm(args):
    if getattr(args, "yes", False) or getattr(args, "dry_run", False):
        return
    if (
        not sys.stdin.isatty()
        or input(f"Confirm {args.command} manager-owned configuration? Type yes: ") != "yes"
    ):
        raise Error("Confirmation required; use --yes for an explicitly reviewed operation")


def execute(args, manager):
    command, root = args.command, manager.root
    if command == "initialize":
        return manager.initialize()
    if command not in ("rollback", "health"):
        manager.pending()
    if command in ("add", "update", "enable", "disable", "remove"):
        if command == "remove":
            confirm(args)
        value = load(args.file) if command in ("add", "update") else None
        # Enforce endpoint approval before committing even desired state.
        if command == "add" and value.get("upstream") not in manager.settings["approved_upstreams"]:
            raise Error("Upstream requires explicit approval in protected manager settings")
        before = (root / "applications.json").read_text(encoding="utf-8")
        try:
            edit(
                root / "applications.json",
                command,
                getattr(args, "name", None),
                value,
                manager.settings["certificate_roots"],
            )
            manager.registry()
        except Exception:
            from .storage import write

            write(root / "applications.json", before)
            raise
        manager.log(command, application=getattr(args, "name", None) or value["name"])
        return {
            "desired_state_updated": True,
            "live_nginx_unchanged": True,
            "next": "apply --dry-run",
        }
    if command == "rollback":
        confirm(args)
        return manager.rollback()
    if command == "health":
        applied = manager.verify_release(manager.current())
        results = {
            "nginx_active": manager.controller.active(),
            "applications": manager.health(applied),
            "canaries": manager.canaries(),
        }
        manager.log("health", nginx_active=results["nginx_active"])
        return results
    registry = manager.registry()
    if command == "validate":
        manager.log("validate", applications=len(registry["applications"]))
        return {"valid": True}
    if command == "show":
        app = next((a for a in registry["applications"] if a["name"] == args.name), None)
        if not app:
            raise Error("Application not found")
        return generate({"version": 1, "applications": [{**app, "enabled": True}]})
    if command == "list":
        current = manager.verify_release(manager.current())
        active = {a["name"]: a for a in current["applications"]}
        return [
            {
                "name": a["name"],
                "enabled": a["enabled"],
                "upstream": a["upstream"],
                "listen": a["listen"],
                "allowed_networks": a["allowed_networks"],
                "configuration": "applied" if active.get(a["name"]) == a else "pending apply",
            }
            for a in registry["applications"]
        ]
    if command in ("generate", "test"):
        # Both validate the complete staged configuration; neither activates it.
        return manager.apply(dry_run=True)
    if command in ("apply", "reload", "rollback"):
        confirm(args)
        if command == "apply":
            return manager.apply(args.dry_run)
        if command == "rollback":
            return manager.rollback()
        manager.verify_bridge()
        applied = manager.verify_release(manager.current())
        manager.log("reload-attempt")
        manager.controller.reload()
        result = manager.verified_health(applied)
        manager.log("reload-success")
        return result
    raise Error("Unsupported command")


def main(argv=None):
    args = parser().parse_args(argv)
    manager = None
    audit_ready = False
    os.umask(0o077)
    try:
        if os.name != "posix" or os.geteuid() != 0:
            raise Error(
                "Operational CLI requires Ubuntu root via sudo; no daemon or permissive sudo rule is installed"
            )
        manager = Manager(settings_from(args.settings))
        protected(manager.root)
        for entry in manager.root.rglob("*"):
            if not entry.is_symlink():
                protected(entry)
        protected(Path(manager.settings["audit_log"]).parent)
        if Path(manager.settings["audit_log"]).exists():
            protected(manager.settings["audit_log"])
        audit_ready = True
        with lock(manager.root):
            result = execute(args, manager)
        print(json.dumps(result, indent=2, sort_keys=True))
        if args.command == "health":
            success = (
                result["nginx_active"]
                and all(x["ok"] for x in result["canaries"])
                and all(x["ok"] for a in result["applications"].values() for x in a.values())
            )
            return 0 if success else 1
        return 0
    except (Error, OSError, ValueError, KeyError, TypeError) as exc:
        if manager and audit_ready:
            try:
                manager.log("operation-error", command=args.command, failure=type(exc).__name__)
            except OSError:
                pass
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
