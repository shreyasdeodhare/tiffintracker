"""Bounded GET probes; redirects are evaluated, never followed; TLS is verified."""

import http.client
import socket
import ssl
import time

from .generator import endpoint


class PinnedHTTPS(http.client.HTTPSConnection):
    def __init__(self, host, port, address, **kwargs):
        super().__init__(host, port, **kwargs)
        self.address = address

    def connect(self):
        self.sock = socket.create_connection((self.address, self.port), self.timeout)
        self.sock = self._context.wrap_socket(self.sock, server_hostname=self.host)


def probe(host, port, path, statuses, timeout=5, tls=False, ca_file=None, address=None):
    start = time.monotonic()
    result = {"tcp": False, "ok": False, "status": None}
    connection = None
    try:
        with socket.create_connection((address or host, port), timeout):
            result["tcp"] = True
        if tls:
            connection = PinnedHTTPS(
                host,
                port,
                address or host,
                timeout=timeout,
                context=ssl.create_default_context(cafile=ca_file),
            )
        else:
            connection = http.client.HTTPConnection(address or host, port, timeout=timeout)
        connection.request(
            "GET",
            path,
            headers={
                "Host": endpoint(host, port),
                "User-Agent": "proxy-manager-health/1",
            },
        )
        response = connection.getresponse()
        result.update(status=response.status, ok=response.status in statuses)
        # Never consume or log application response bodies.
    except (OSError, ValueError, http.client.HTTPException) as exc:
        result["error"] = type(exc).__name__
    finally:
        if connection:
            connection.close()
    result["milliseconds"] = round((time.monotonic() - start) * 1000, 1)
    return result


def check_app(app, proxy=False):
    health = app["health_check"]
    if not health["enabled"]:
        return {
            "ok": False,
            "skipped": True,
            "reason": "Health disabled; cannot certify deployment",
        }
    if proxy:
        tls = app.get("tls")
        return probe(
            app["server_names"][0],
            app["listen"]["port"],
            health["path"],
            health["expected_statuses"],
            health["timeout"],
            tls=bool(tls),
            ca_file=tls["ca_file"] if tls else None,
            address=app["listen"]["address"],
        )
    return probe(
        app["upstream"]["host"],
        app["upstream"]["port"],
        health["path"],
        health["expected_statuses"],
        health["timeout"],
    )
