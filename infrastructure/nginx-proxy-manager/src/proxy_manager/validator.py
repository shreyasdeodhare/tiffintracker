"""Strict, deny-by-default schema. Registry values cannot supply directives."""

import ipaddress
import re
from pathlib import Path


class Error(ValueError):
    """An actionable validation or safety failure."""


FORBIDDEN_PORTS = {
    21,
    22,
    53,
    111,
    139,
    389,
    445,
    636,
    990,
    1433,
    2049,
    2375,
    2376,
    3306,
    5432,
    5433,
    6379,
    6380,
    6382,
    6443,
    9200,
    11211,
    26379,
    27017,
}
NAME = re.compile(r"[a-z][a-z0-9-]{0,47}\Z")
HOST = re.compile(
    r"(?=.{1,253}\Z)(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*\Z"
)
PATH = re.compile(r"/[A-Za-z0-9/_~.%-]*\Z")


def fields(value, allowed, required, label):
    if not isinstance(value, dict) or set(value) - set(allowed) or set(required) - set(value):
        raise Error(f"{label}: missing or unsupported fields")


def integer(value, low, high, label):
    if type(value) is not int or not low <= value <= high:
        raise Error(f"{label}: expected integer {low}–{high}")
    return value


def boolean(value, label):
    if type(value) is not bool:
        raise Error(f"{label}: expected true or false")


def hostname(value):
    if not isinstance(value, str):
        raise Error("Host must be a literal IP or DNS hostname")
    try:
        ipaddress.ip_address(value)
        return
    except ValueError:
        if not HOST.fullmatch(value) or re.fullmatch(r"[0-9.]+", value):
            raise Error("Invalid upstream/server hostname") from None


def path_value(value):
    if not isinstance(value, str) or not PATH.fullmatch(value) or ".." in value or "%" in value:
        raise Error("Path must be a simple absolute URL path without traversal or escapes")


def certificate_path(value, roots):
    if (
        not isinstance(value, str)
        or not re.fullmatch(r"/[A-Za-z0-9_./-]+", value)
        or ".." in Path(value).parts
    ):
        raise Error("Unsafe certificate path")
    resolved = Path(value).resolve()
    if not any(resolved.is_relative_to(Path(root).resolve()) for root in roots):
        raise Error("Certificate path is outside approved certificate roots")


def validate_registry(registry, certificate_roots=()):
    fields(registry, {"version", "applications"}, {"version", "applications"}, "registry")
    if registry["version"] != 1 or type(registry["version"]) is not int:
        raise Error("Unsupported registry version")
    if not isinstance(registry["applications"], list) or len(registry["applications"]) > 100:
        raise Error("applications must be a list of at most 100 entries")
    names, listeners = set(), set()
    for app in registry["applications"]:
        fields(
            app,
            {
                "name",
                "enabled",
                "upstream",
                "listen",
                "server_names",
                "allowed_networks",
                "websocket",
                "health_check",
                "proxy_settings",
                "headers",
                "tls",
                "rate_limit",
            },
            {
                "name",
                "enabled",
                "upstream",
                "listen",
                "server_names",
                "allowed_networks",
                "websocket",
                "health_check",
                "proxy_settings",
            },
            "application",
        )
        name = app["name"]
        if not isinstance(name, str) or not NAME.fullmatch(name) or name in names:
            raise Error("Invalid or duplicate application name")
        names.add(name)
        boolean(app["enabled"], "enabled")
        boolean(app["websocket"], "websocket")
        fields(app["upstream"], {"host", "port"}, {"host", "port"}, "upstream")
        hostname(app["upstream"]["host"])
        port = integer(app["upstream"]["port"], 1, 65535, "upstream port")
        if port in FORBIDDEN_PORTS:
            raise Error("Infrastructure service ports cannot be proxied")
        fields(
            app["listen"],
            {"mode", "address", "port"},
            {"mode", "address", "port"},
            "listen",
        )
        if app["listen"]["mode"] != "private":
            raise Error("Only private mode is supported; public exposure is not authorized")
        if not isinstance(app["listen"]["address"], str):
            raise Error("listen.address must be an explicit literal IP string")
        try:
            address = ipaddress.ip_address(app["listen"]["address"])
        except (ValueError, TypeError):
            raise Error("listen.address must be an explicit literal IP") from None
        tailnet = address.version == 4 and address in ipaddress.ip_network("100.64.0.0/10")
        if (
            address.is_unspecified
            or address.is_multicast
            or address.is_link_local
            or not (
                address.is_loopback
                or tailnet
                or any(
                    address.version == n.version and address in n
                    for n in map(
                        ipaddress.ip_network,
                        ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "fc00::/7"),
                    )
                )
            )
        ):
            raise Error("Listener must be a specific private, loopback or Tailscale IP")
        listen_port = integer(app["listen"]["port"], 1, 65535, "listen port")
        if listen_port in FORBIDDEN_PORTS:
            raise Error("Infrastructure ports cannot be used as proxy listeners")
        listener = (str(address), listen_port)
        if app["enabled"] and listener in listeners:
            raise Error(
                "Duplicate listener/server-name scope; shared listener adoption is unsupported"
            )
        if app["enabled"]:
            listeners.add(listener)
        hosts = app["server_names"]
        if (
            not isinstance(hosts, list)
            or not 1 <= len(hosts) <= 10
            or any(not isinstance(h, str) for h in hosts)
            or len({h.lower() for h in hosts}) != len(hosts)
        ):
            raise Error("server_names must contain 1–10 unique exact hosts")
        for host in hosts:
            hostname(host)
        networks = app["allowed_networks"]
        if not isinstance(networks, list) or not networks or len(networks) > 32:
            raise Error("An explicit nonempty source network allowlist is required")
        for network in networks:
            if not isinstance(network, str) or "/" not in network:
                raise Error("Source networks must be CIDR strings")
            try:
                cidr = ipaddress.ip_network(network, strict=True)
            except (ValueError, TypeError):
                raise Error("Invalid or noncanonical source CIDR") from None
            if cidr.prefixlen == 0 or cidr.is_multicast:
                raise Error("Unrestricted source networks are forbidden")
        fields(
            app["health_check"],
            {"enabled", "path", "expected_statuses", "timeout"},
            {"enabled", "path", "expected_statuses", "timeout"},
            "health_check",
        )
        health = app["health_check"]
        boolean(health["enabled"], "health enabled")
        path_value(health["path"])
        integer(health["timeout"], 1, 15, "health timeout")
        if not isinstance(health["expected_statuses"], list) or not health["expected_statuses"]:
            raise Error("Expected health statuses must be specified")
        for status in health["expected_statuses"]:
            integer(status, 200, 399, "expected HTTP status")
        settings = app["proxy_settings"]
        fields(
            settings,
            {"max_body_size", "connect_timeout", "read_timeout", "send_timeout"},
            {"max_body_size", "connect_timeout", "read_timeout", "send_timeout"},
            "proxy_settings",
        )
        if not isinstance(settings["max_body_size"], str) or not re.fullmatch(
            r"[1-9][0-9]{0,3}[mk]", settings["max_body_size"]
        ):
            raise Error("Invalid bounded body size, for example 1m or 512k")
        for key in ("connect_timeout", "read_timeout", "send_timeout"):
            integer(settings[key], 1, 300, key)
        headers = app.get("headers", {})
        if not isinstance(headers, dict) or len(headers) > 12:
            raise Error("headers must be a bounded dictionary")
        for key, value in headers.items():
            if (
                not re.fullmatch(r"X-App-[A-Za-z0-9-]{1,40}", key)
                or not isinstance(value, str)
                or not re.fullmatch(r"[A-Za-z0-9 ._/-]{1,100}", value)
            ):
                raise Error("Only non-secret X-App-* literal headers are supported")
        if app.get("tls") is not None:
            fields(
                app["tls"],
                {"certificate", "key", "ca_file"},
                {"certificate", "key", "ca_file"},
                "tls",
            )
            for value in app["tls"].values():
                certificate_path(value, certificate_roots)
        rate = app.get("rate_limit")
        if rate is not None:
            fields(
                rate,
                {"path", "requests_per_minute", "burst"},
                {"path", "requests_per_minute", "burst"},
                "rate_limit",
            )
            path_value(rate["path"])
            integer(rate["requests_per_minute"], 1, 10000, "rate")
            integer(rate["burst"], 1, 1000, "burst")
    return registry
