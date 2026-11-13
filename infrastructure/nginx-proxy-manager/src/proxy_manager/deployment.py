"""Immutable releases, preflight validation, atomic pointer switch and verified rollback."""

import contextlib
import copy
import ipaddress
import os
import re
import shutil
import socket
import uuid
from pathlib import Path

from .generator import MARKER, generate
from .health_checker import check_app, probe
from .nginx_controller import Controller, run
from .registry import read
from .storage import audit, digest, load, protected, write, write_json
from .validator import Error, fields, validate_registry

EMPTY = {"version": 1, "applications": []}


def settings_from(path, privileged=True):
    if privileged:
        protected(path)
        protected(Path(__file__))
    settings = load(path)
    keys = {
        "root",
        "nginx",
        "nginx_config",
        "systemctl",
        "ss",
        "ip",
        "bridge",
        "certificate_roots",
        "approved_upstreams",
        "canaries",
        "audit_log",
    }
    fields(settings, keys, keys, "manager settings")
    for key in keys - {"certificate_roots", "approved_upstreams", "canaries"}:
        if (
            not isinstance(settings[key], str)
            or not re.fullmatch(r"/[A-Za-z0-9_./-]+", settings[key])
            or ".." in Path(settings[key]).parts
        ):
            raise Error("Manager paths must be simple absolute paths")
    if (
        not isinstance(settings["certificate_roots"], list)
        or not isinstance(settings["approved_upstreams"], list)
        or not isinstance(settings["canaries"], list)
        or not settings["canaries"]
    ):
        raise Error(
            "Certificate roots, approved upstreams and nonempty preservation canaries are required"
        )
    if privileged:
        for key in ("root", "nginx", "systemctl", "ss", "ip", "nginx_config"):
            protected(settings[key])
    return settings


class Manager:
    def __init__(self, settings, controller=None):
        self.settings = settings
        self.root = Path(settings["root"])
        self.controller = controller or Controller(settings)

    def log(self, event, **details):
        audit(self.settings["audit_log"], event, **details)

    def registry(self):
        value = read(self.root / "applications.json", self.settings["certificate_roots"])
        for app in value["applications"]:
            if app["upstream"] not in self.settings["approved_upstreams"]:
                raise Error("Upstream is not explicitly approved as an HTTP application endpoint")
        return value

    def release(self, identifier):
        if not isinstance(identifier, str) or not re.fullmatch(r"[a-f0-9]{32}", identifier):
            raise Error("Invalid release identifier")
        path = self.root / "releases" / identifier
        if path.is_symlink() or path.resolve().parent != (self.root / "releases").resolve():
            raise Error("Unsafe release path")
        return path

    def verify_release(self, identifier):
        path = self.release(identifier)
        manifest = load(path / "manifest.json")
        if {p.name for p in path.iterdir()} != set(manifest) | {"manifest.json"}:
            raise Error("Unexpected files in manager release")
        for name, checksum in manifest.items():
            if not re.fullmatch(
                r"(?:app\.[a-z][a-z0-9-]{0,47}|00-manager-common)\.conf|snapshot\.json",
                name,
            ):
                raise Error("Unsafe manifest filename")
            target = path / name
            if target.is_symlink() or digest(target) != checksum:
                raise Error("Manager-owned release was modified; refusing overwrite")
            if name.endswith(".conf") and not target.read_text(encoding="utf-8").startswith(MARKER):
                raise Error("Missing ownership marker")
        snapshot = validate_registry(
            load(path / "snapshot.json"), self.settings["certificate_roots"]
        )
        expected = generate(snapshot)
        if {k for k in manifest if k.endswith(".conf")} != set(expected):
            raise Error("Incomplete release manifest")
        for name, content in expected.items():
            if (path / name).read_text(encoding="utf-8") != content:
                raise Error("Release does not match its validated snapshot")
        return snapshot

    def current(self):
        link = self.root / "current"
        if not link.is_symlink():
            raise Error("Current release must be a manager-owned symlink")
        identifier = link.resolve().name
        if link.resolve() != self.release(identifier).resolve():
            raise Error("Current release escaped manager directory")
        self.verify_release(identifier)
        return identifier

    def bridge(self):
        return MARKER + f"include {self.root}/current/*.conf;\n"

    def verify_bridge(self):
        path = Path(self.settings["bridge"])
        if (
            path.is_symlink()
            or not path.is_file()
            or path.read_text(encoding="utf-8") != self.bridge()
        ):
            raise Error("Dedicated include is missing or not manager-owned")

    def new_release(self, registry):
        identifier = uuid.uuid4().hex
        path = self.release(identifier)
        path.mkdir(mode=0o700)
        for name, content in generate(registry).items():
            write(path / name, content)
        write_json(path / "snapshot.json", registry)
        write_json(path / "manifest.json", {p.name: digest(p) for p in path.iterdir()})
        return identifier

    def point(self, identifier):
        temporary = self.root / (".current-" + uuid.uuid4().hex)
        temporary.symlink_to(self.release(identifier), target_is_directory=True)
        try:
            os.replace(temporary, self.root / "current")
            fd = os.open(self.root, os.O_DIRECTORY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        finally:
            if temporary.is_symlink():
                temporary.unlink()

    def network_preflight(self, registry):
        import json

        addresses = {
            item["local"]
            for interface in json.loads(run([self.settings["ip"], "-j", "address", "show"]))
            for item in interface.get("addr_info", [])
        }
        for app in registry["applications"]:
            if not app["enabled"]:
                continue
            address = app["listen"]["address"]
            if address not in addresses:
                raise Error("Configured listening IP is not currently assigned to this host")
            # A local probe must traverse the same source restriction as clients.
            if not any(
                ipaddress.ip_address(address) in ipaddress.ip_network(n)
                for n in app["allowed_networks"]
            ):
                raise Error(
                    "Source allowlist must include the listener IP for verified local probes"
                )
            for path in (app.get("tls") or {}).values():
                protected(path)
            for result in socket.getaddrinfo(
                app["upstream"]["host"],
                app["upstream"]["port"],
                type=socket.SOCK_STREAM,
            ):
                if (
                    ipaddress.ip_address(result[4][0]).is_multicast
                    or ipaddress.ip_address(result[4][0]).is_unspecified
                ):
                    raise Error("Unsafe upstream DNS resolution")

    def health(self, registry, proxy=True):
        results = {}
        for app in registry["applications"]:
            if app["enabled"]:
                results[app["name"]] = {
                    "upstream": check_app(app),
                    **({"proxy": check_app(app, True)} if proxy else {}),
                }
        return results

    def canaries(self):
        results = []
        for item in self.settings["canaries"]:
            fields(
                item,
                {"host", "port", "path", "statuses", "tls", "ca_file"},
                {"host", "port", "path", "statuses"},
                "preservation canary",
            )
            results.append(
                probe(
                    item["host"],
                    item["port"],
                    item["path"],
                    item["statuses"],
                    tls=item.get("tls", False),
                    ca_file=item.get("ca_file"),
                )
            )
        return results

    def verified_health(self, registry):
        if not self.controller.active():
            raise Error("Nginx inactive")
        apps = self.health(registry)
        canaries = self.canaries()
        if not all(item["ok"] for item in canaries) or not all(
            result["ok"] for app in apps.values() for result in app.values()
        ):
            raise Error("Application or preservation HTTP verification failed")
        return {"applications": apps, "preservation_canaries": canaries}

    def pending(self):
        if (self.root / "pending.json").exists():
            raise Error(
                "Interrupted transaction detected; run rollback --yes before another deployment"
            )

    def apply(self, dry_run=False):
        self.pending()
        self.verify_bridge()
        old = self.current()
        previous = self.verify_release(old)
        registry = self.registry()
        self.network_preflight(registry)
        # Prove current behavior is good before claiming that it can be restored.
        self.verified_health(previous)
        upstream = self.health(registry, proxy=False)
        if not all(result["upstream"]["ok"] for result in upstream.values()):
            raise Error("Candidate upstream health failed")
        files, manual, hashes = self.controller.snapshot(self.root)
        self.controller.conflicts(registry, manual, previous)
        new = self.new_release(registry)
        candidate = self.controller.candidate(files, self.release(new), self.root / "staging" / new)
        self.controller.test(candidate)
        self.controller.unchanged(hashes)
        self.log(
            "dry-run" if dry_run else "preflight",
            release=new,
            applications=len(upstream),
        )
        if dry_run:
            return {
                "dry_run": True,
                "release": new,
                "files": sorted(generate(registry)),
                "reload": False,
            }
        backup = self.root / "backups" / uuid.uuid4().hex
        shutil.copytree(self.release(old), backup)
        write_json(backup / "desired-registry.json", registry)
        old_state = load(self.root / "state.json")
        transaction = {"old": old, "new": new, "manual_hashes": hashes, "old_state": old_state}
        write_json(self.root / "pending.json", transaction)
        try:
            self.point(new)
            self.controller.unchanged(hashes)
            self.controller.reload()
            result = self.verified_health(registry)
            self.controller.unchanged(hashes)
            write_json(self.root / "state.json", {"current": new, "previous": old})
            (self.root / "pending.json").unlink()
            self.log("apply-success", release=new)
            return {"release": new, "backup": str(backup), **result}
        except Exception as exc:
            with contextlib.suppress(OSError):
                self.log("apply-failure", failure=type(exc).__name__)
            try:
                self.restore(old, hashes)
                write_json(self.root / "state.json", old_state)
                (self.root / "pending.json").unlink()
            except Exception as rollback_error:  # noqa: BLE001 -- rollback must report every failure
                self.log("rollback-failure", failure=type(rollback_error).__name__)
                raise Error(
                    f"Deployment failed ({exc}); rollback NOT verified ({rollback_error}); journal retained"
                ) from exc
            raise Error(
                f"Deployment failed ({exc}); previous release restored and rollback verified"
            ) from exc

    def restore(self, identifier, hashes):
        registry = self.verify_release(identifier)
        self.controller.unchanged(hashes)
        self.point(identifier)
        self.controller.reload()
        self.verified_health(registry)
        self.log("rollback-verified", release=identifier)

    def rollback(self):
        self.verify_bridge()
        pending = self.root / "pending.json"
        if pending.exists():
            transaction = load(pending)
            target, hashes = transaction["old"], transaction["manual_hashes"]
        else:
            state = load(self.root / "state.json")
            target = state.get("previous")
            if not target:
                raise Error("No previous successful release")
            _, _, hashes = self.controller.snapshot(self.root)
        old = self.current()
        self.restore(target, hashes)
        write_json(self.root / "state.json", {"current": target, "previous": old})
        if pending.exists():
            pending.unlink()
        return {
            "restored": target,
            "verified": True,
            "desired_registry": "unchanged; review before applying again",
        }

    def initialize(self):
        """Create only manager state. Never writes the Nginx bridge or reloads."""
        if (self.root / "current").exists() or (self.root / "applications.json").exists():
            raise Error("Manager already initialized")
        for directory in ("releases", "staging", "backups"):
            (self.root / directory).mkdir(mode=0o700)
        write_json(self.root / "applications.json", copy.deepcopy(EMPTY))
        identifier = self.new_release(EMPTY)
        self.point(identifier)
        write_json(self.root / "state.json", {"current": identifier, "previous": None})
        self.log("initialize", release=identifier)
        return {
            "initialized": True,
            "nginx_bridge_not_installed": True,
            "bridge_content": self.bridge(),
        }
