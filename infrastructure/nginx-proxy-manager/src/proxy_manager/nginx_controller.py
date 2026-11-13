"""Argument-array subprocess boundary, complete candidate view, conservative conflicts."""

import fnmatch
import glob
import hashlib
import ipaddress
import re
import shlex
import subprocess
from pathlib import Path

from .storage import digest, write
from .validator import Error

INCLUDE = re.compile(
    r'(?m)^(\s*)include\s+([A-Za-z0-9_./*?-]+|"[^"\n]+"|\x27[^\x27\n]+\x27)\s*;\s*(?:#[^\n]*)?$'
)


def run(arguments):
    try:
        process = subprocess.run(arguments, capture_output=True, text=True, timeout=30, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise Error(f"Command failed: {Path(arguments[0]).name}: {type(exc).__name__}") from exc
    if process.returncode:
        # Nginx stderr can contain arbitrary configuration; do not leak it to audit logs.
        raise Error(
            f"{Path(arguments[0]).name} failed with exit {process.returncode}; inspect its protected configuration locally"
        )
    return process.stdout


def dump_files(output):
    matches = list(re.finditer(r"(?m)^# configuration file (/.+):\n", output))
    if not matches:
        raise Error("Cannot enumerate Nginx configuration")
    return {
        m.group(1): output[
            m.end() : matches[i + 1].start() if i + 1 < len(matches) else len(output)
        ]
        for i, m in enumerate(matches)
    }


def listener(value):
    # ss can append a Linux interface scope after IPv4 or bracketed IPv6.
    value = re.sub(r"%[A-Za-z0-9_.-]+(?=:)", "", value)
    if value.isdigit():
        return "0.0.0.0", int(value)
    match = re.fullmatch(r"\[([^]]+)\]:(\d+)", value)
    if match:
        return str(ipaddress.ip_address(match[1])), int(match[2])
    host, separator, port = value.rpartition(":")
    if separator and port.isdigit():
        if host == "*":
            host = "0.0.0.0"
        try:
            return str(ipaddress.ip_address(host)), int(port)
        except ValueError:
            pass
    raise Error("Unsupported manual listener; review before integration")


def servers(text):
    lexer = shlex.shlex(text, posix=True, punctuation_chars="{};")
    lexer.whitespace_split = True
    tokens = list(lexer)
    # Split punctuation runs such as }; into separate tokens.
    tokens = [
        piece
        for token in tokens
        for piece in (list(token) if token and set(token) <= set("{};") else [token])
    ]
    result, stack, statement = [], [], []
    for token in tokens:
        if token == "{":
            parent = next((x for x in reversed(stack) if isinstance(x, dict)), None)
            node = {"listen": [], "names": []} if statement == ["server"] else None
            if node is not None:
                result.append(node)
            stack.append(node if node is not None else parent)
            statement = []
        elif token == "}":
            if not stack:
                raise Error("Unbalanced Nginx configuration")
            stack.pop()
            statement = []
        elif token == ";":
            node = stack[-1] if stack else None
            if node is not None and statement:
                if statement[0] == "listen":
                    node["listen"].append(listener(statement[1]))
                elif statement[0] == "server_name":
                    node["names"].extend(statement[1:])
            statement = []
        else:
            statement.append(token)
    for node in result:
        if not node["listen"]:
            node["listen"] = [("0.0.0.0", 80)]
    return result


class Controller:
    def __init__(self, settings):
        self.settings = settings

    def test(self, candidate=None):
        return run(
            [
                self.settings["nginx"],
                "-t",
                "-c",
                str(candidate or self.settings["nginx_config"]),
            ]
        )

    def active(self):
        return run([self.settings["systemctl"], "is-active", "nginx"]).strip() == "active"

    def reload(self):
        self.test()
        run([self.settings["systemctl"], "reload", "nginx"])
        if not self.active():
            raise Error("Nginx is not active after reload")

    def snapshot(self, root):
        self.test()
        output = run([self.settings["nginx"], "-T", "-c", self.settings["nginx_config"]])
        files = dump_files(output)
        manual = {
            name: text
            for name, text in files.items()
            if not Path(name).is_relative_to(root) and name != self.settings["bridge"]
        }
        hashes = {name: digest(name) for name in manual}
        self.include_membership = {}
        base = Path(self.settings["nginx_config"]).parent
        for content in manual.values():
            for match in INCLUDE.finditer(content):
                pattern = match[2].strip("\"'")
                pattern = str(base / pattern) if not pattern.startswith("/") else pattern
                self.include_membership[pattern] = sorted(glob.glob(pattern))
        return files, manual, hashes

    def unchanged(self, hashes):
        if any(not Path(name).is_file() or digest(name) != value for name, value in hashes.items()):
            raise Error("A manually maintained Nginx file changed during the transaction")
        if any(
            sorted(glob.glob(pattern)) != members
            for pattern, members in getattr(self, "include_membership", {}).items()
        ):
            raise Error("Nginx include membership changed during the transaction")

    def conflicts(self, registry, manual, previous):
        existing = [server for text in manual.values() for server in servers(text)]
        occupied = run([self.settings["ss"], "-H", "-lnt"])
        old = {
            (a["listen"]["address"], a["listen"]["port"])
            for a in previous["applications"]
            if a["enabled"]
        }
        sockets = []
        for line in occupied.splitlines():
            parts = line.split()
            if len(parts) >= 4:
                try:
                    sockets.append(listener(parts[3]))
                except Error:
                    raise Error("Cannot safely interpret an existing TCP listener") from None
        for app in registry["applications"]:
            if not app["enabled"]:
                continue
            address, port = app["listen"]["address"], app["listen"]["port"]
            for server in existing:
                for host, number in server["listen"]:
                    if number == port and (host in (address, "0.0.0.0", "::")):
                        overlap = set(app["server_names"]) & set(server["names"])
                        raise Error(
                            "Existing manual listener conflict"
                            + (" and duplicate server name" if overlap else "")
                        )
            for host, number in sockets:
                if (
                    number == port
                    and host in (address, "0.0.0.0", "::")
                    and (address, port) not in old
                ):
                    raise Error("Port already occupied by an unmanaged service")

    def candidate(self, files, release, destination):
        """Clone the include graph, substituting only the manager's dedicated bridge."""
        destination.mkdir(mode=0o700)
        bridge = self.settings["bridge"]
        mapping = {
            name: destination / (hashlib.sha256(name.encode()).hexdigest() + ".conf")
            for name in files
        }
        base = Path(self.settings["nginx_config"]).parent
        found = False
        for name, content in files.items():
            if name == bridge:
                write(mapping[name], f"include {release}/*.conf;\n")
                continue

            def replace(match):
                nonlocal found
                pattern = match[2].strip("\"'")
                pattern = str(base / pattern) if not pattern.startswith("/") else pattern
                matches = sorted(glob.glob(pattern))
                statements = []
                for item in matches:
                    if item == bridge:
                        found = True
                        statements.append(f"include {release}/*.conf;")
                    elif item in mapping:
                        statements.append(f"include {mapping[item]};")
                    else:
                        # Every matched include must have appeared in nginx -T.
                        raise Error("Include graph changed or contains an unreadable configuration")
                if bridge not in matches and fnmatch.fnmatch(bridge, pattern):
                    found = True
                    statements.append(f"include {release}/*.conf;")
                return (
                    "\n".join(statements) if statements else "# Empty include retained in candidate"
                )

            write(mapping[name], INCLUDE.sub(replace, content))
        if not found:
            raise Error("Dedicated bridge is not covered by an existing include")
        return mapping[self.settings["nginx_config"]]
