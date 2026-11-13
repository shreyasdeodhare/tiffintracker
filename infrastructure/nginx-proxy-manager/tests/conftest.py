import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture
def app():
    return {
        "name": "example-api",
        "enabled": True,
        "upstream": {"host": "127.0.0.1", "port": 8001},
        "listen": {"mode": "private", "address": "127.0.0.1", "port": 18081},
        "server_names": ["127.0.0.1"],
        "allowed_networks": ["127.0.0.0/8"],
        "websocket": False,
        "health_check": {
            "enabled": True,
            "path": "/",
            "expected_statuses": [200, 302],
            "timeout": 2,
        },
        "proxy_settings": {
            "max_body_size": "1m",
            "connect_timeout": 5,
            "read_timeout": 60,
            "send_timeout": 60,
        },
    }


@pytest.fixture
def registry(app):
    return {"version": 1, "applications": [copy.deepcopy(app)]}
