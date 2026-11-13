from urllib.request import Request, urlopen

from flask import current_app


def bridge(path, method="GET"):
    token = current_app.config["WA_BRIDGE_TOKEN"]
    if not token:
        raise ValueError("WhatsApp connector is not configured")
    req = Request(
        "http://127.0.0.1:8090" + path, headers={"Authorization": "Bearer " + token}, method=method
    )
    with urlopen(req, timeout=5) as response:
        return response.read(1024 * 1024), response.headers.get_content_type()
