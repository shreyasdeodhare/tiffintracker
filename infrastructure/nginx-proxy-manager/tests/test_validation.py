import copy

import pytest

from proxy_manager.generator import generate
from proxy_manager.registry import edit, read
from proxy_manager.storage import load, write_json
from proxy_manager.validator import Error, validate_registry


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        (None, "name", "../bad"),
        (None, "name", "x;include"),
        (None, "enabled", 1),
        (None, "server_names", [{}]),
        (None, "server_names", ["x", "X"]),
        ("upstream", "host", "999.1.1.1"),
        ("upstream", "host", "x;return 200;"),
        ("upstream", "port", True),
        ("upstream", "port", 0),
        ("upstream", "port", 65536),
        ("upstream", "port", 5432),
        ("upstream", "port", 6379),
        ("upstream", "port", 22),
        ("listen", "address", "0.0.0.0"),
        ("listen", "address", "8.8.8.8"),
        ("listen", "address", "192.0.2.1"),
        ("listen", "address", "169.254.1.1"),
        ("listen", "mode", "public"),
        ("listen", "port", 443.5),
        (None, "allowed_networks", []),
        (None, "allowed_networks", ["0.0.0.0/0"]),
        (None, "allowed_networks", ["192.168.1.1/24"]),
        (None, "allowed_networks", ["invalid"]),
        ("health_check", "path", "/../secret"),
        ("health_check", "path", "/x%0A"),
        ("health_check", "expected_statuses", [500]),
        ("health_check", "timeout", 999),
        ("proxy_settings", "max_body_size", "1m;return 200"),
        ("proxy_settings", "read_timeout", 0),
        (None, "headers", {"Authorization": "secret"}),
        (None, "headers", {"X-App-Tag": "x\nreturn 200;"}),
        (None, "arbitrary_directive", "x"),
    ],
)
def test_rejects_invalid(registry, section, key, value):
    target = registry["applications"][0]
    if section:
        target = target[section]
    target[key] = value
    with pytest.raises(Error):
        validate_registry(registry)


def test_duplicate_name(registry):
    registry["applications"].append(copy.deepcopy(registry["applications"][0]))
    with pytest.raises(Error, match="duplicate application"):
        validate_registry(registry)


def test_duplicate_listener(registry):
    second = copy.deepcopy(registry["applications"][0])
    second["name"] = "another"
    registry["applications"].append(second)
    with pytest.raises(Error, match="Duplicate listener"):
        validate_registry(registry)


def test_multiple_and_disabled(registry):
    second = copy.deepcopy(registry["applications"][0])
    second.update(name="disabled", enabled=False)
    registry["applications"].append(second)
    validate_registry(registry)
    assert set(generate(registry)) == {"app.example-api.conf", "00-manager-common.conf"}


def test_generator_safe_headers_and_access(registry):
    validate_registry(registry)
    config = generate(registry)["app.example-api.conf"]
    assert "allow 127.0.0.0/8;\n    deny all;" in config
    assert "proxy_set_header X-Forwarded-For $remote_addr;" in config
    assert "$proxy_add_x_forwarded_for" not in config
    assert "proxy_set_header Host $http_host;" in config
    assert "default_server" in config and "return 444;" in config
    assert "proxy_set_header Upgrade" not in config


def test_websocket_and_rate(registry):
    registry["applications"][0].update(
        websocket=True, rate_limit={"path": "/login", "requests_per_minute": 10, "burst": 15}
    )
    validate_registry(registry)
    files = generate(registry)
    assert "map $http_upgrade" in files["00-manager-common.conf"]
    assert "limit_req_zone" in files["00-manager-common.conf"]
    assert "proxy_set_header Upgrade $http_upgrade" in files["app.example-api.conf"]
    assert "location = /login" in files["app.example-api.conf"]


def test_tls_path_escape(registry):
    registry["applications"][0]["tls"] = {
        "certificate": "/etc/certs/a.crt",
        "key": "/etc/certs/../secret",
        "ca_file": "/etc/certs/a.crt",
    }
    with pytest.raises(Error):
        validate_registry(registry, ["/etc/certs"])


def test_registry_crud_and_partial_update(tmp_path, registry):
    path = tmp_path / "registry.json"
    write_json(path, registry)
    edit(path, "update", "example-api", {"proxy_settings": {"read_timeout": 90}})
    app = read(path)["applications"][0]
    assert app["proxy_settings"]["connect_timeout"] == 5
    assert app["proxy_settings"]["read_timeout"] == 90
    edit(path, "disable", "example-api")
    assert not read(path)["applications"][0]["enabled"]
    edit(path, "enable", "example-api")
    assert read(path)["applications"][0]["enabled"]
    before = path.read_bytes()
    with pytest.raises(Error):
        edit(path, "update", "example-api", {"upstream": {"port": 22}})
    assert path.read_bytes() == before
    edit(path, "remove", "example-api")
    assert read(path)["applications"] == []
    edit(path, "add", value=registry["applications"][0])
    assert len(read(path)["applications"]) == 1


def test_json_duplicate_keys_and_bad_json(tmp_path):
    path = tmp_path / "data.json"
    for content in ('{"version":1,"version":2}', "{bad", "[] garbage"):
        path.write_text(content, encoding="utf-8")
        with pytest.raises(Error):
            load(path)


@pytest.mark.parametrize("value", [[123], ["127.0.0.1"]])
def test_network_requires_cidr_string(registry, value):
    registry["applications"][0]["allowed_networks"] = value
    with pytest.raises(Error):
        validate_registry(registry)


def test_listener_rejects_integer_address(registry):
    registry["applications"][0]["listen"]["address"] = 2130706433
    with pytest.raises(Error):
        validate_registry(registry)
