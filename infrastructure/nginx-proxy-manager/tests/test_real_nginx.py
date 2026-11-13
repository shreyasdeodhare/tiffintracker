"""Isolated Nginx process only: separate config, PID, logs and ephemeral loopback port."""

import http.client
import json
import os
import shutil
import socket
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from proxy_manager.generator import generate
from proxy_manager.validator import validate_registry

pytestmark = pytest.mark.skipif(
    os.name != "posix" or not shutil.which("nginx"),
    reason="Isolated real Nginx requires Linux Nginx",
)


def test_real_nginx_headers_access_redirect_and_websocket(tmp_path, app):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self):
            if self.path == "/upgrade":
                self.send_response(101)
                self.send_header("Upgrade", self.headers.get("Upgrade", "missing"))
                self.send_header("Connection", "upgrade")
                self.end_headers()
            elif self.path == "/redirect":
                self.send_response(302)
                self.send_header("Location", "/login")
                self.send_header("Content-Length", "0")
                self.end_headers()
            else:
                body = json.dumps(dict(self.headers)).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        def log_message(self, *args):
            pass

    upstream = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    thread.start()
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    app["upstream"]["port"] = upstream.server_port
    app["listen"]["port"] = port
    app["websocket"] = True
    registry = validate_registry({"version": 1, "applications": [app]})
    generated = tmp_path / "generated"
    generated.mkdir()
    for name, content in generate(registry).items():
        (generated / name).write_text(content, encoding="utf-8")
    conf = tmp_path / "nginx.conf"
    conf.write_text(
        f"worker_processes 1;\npid {tmp_path}/nginx.pid;\nerror_log {tmp_path}/error.log notice;\n"
        f"events {{ worker_connections 32; }}\nhttp {{ access_log off; include {generated}/*.conf; }}\n",
        encoding="utf-8",
    )
    subprocess.run(
        [shutil.which("nginx"), "-t", "-p", str(tmp_path), "-c", str(conf)],
        check=True,
        capture_output=True,
    )
    process = subprocess.Popen(
        [shutil.which("nginx"), "-p", str(tmp_path), "-c", str(conf), "-g", "daemon off;"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        for _ in range(40):
            try:
                with socket.create_connection(("127.0.0.1", port), 0.2):
                    break
            except OSError:
                time.sleep(0.05)
        client = http.client.HTTPConnection("127.0.0.1", port, timeout=3)
        client.request(
            "GET",
            "/",
            headers={
                "X-Forwarded-For": "8.8.8.8",
                "X-Forwarded-Proto": "evil",
                "Forwarded": "for=evil",
            },
        )
        response = client.getresponse()
        assert response.status == 200
        headers = json.loads(response.read())
        assert headers["Host"] == f"127.0.0.1:{port}"
        assert headers["X-Forwarded-For"] == "127.0.0.1"
        assert headers["X-Real-IP"] == "127.0.0.1"
        assert headers["X-Forwarded-Proto"] == "http"
        assert "Forwarded" not in headers
        client.request("GET", "/redirect")
        response = client.getresponse()
        assert response.status == 302
        response.read()
        client.close()
        client = http.client.HTTPConnection("127.0.0.1", port, timeout=3)
        client.request("GET", "/upgrade", headers={"Upgrade": "websocket", "Connection": "upgrade"})
        response = client.getresponse()
        assert response.status == 101
        assert response.getheader("Upgrade") == "websocket"
        client.close()
        client = http.client.HTTPConnection("127.0.0.1", port, timeout=3)
        client.request("GET", "/", headers={"Host": "unregistered.invalid"})
        with pytest.raises(http.client.RemoteDisconnected):
            client.getresponse()
        client.close()
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        upstream.shutdown()
        upstream.server_close()
