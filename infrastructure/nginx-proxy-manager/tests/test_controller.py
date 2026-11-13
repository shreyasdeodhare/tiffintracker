from unittest.mock import patch

import pytest

from proxy_manager.nginx_controller import Controller, dump_files, listener, servers
from proxy_manager.storage import lock
from proxy_manager.validator import Error


def test_parser_handles_comments_quotes_and_nested_locations():
    text = """# server { listen 22; }
    server {listen 443 ssl; server_name a.test b.test; location / { proxy_pass http://localhost:8001; }}
    server { listen [::]:80 default_server; server_name "_"; }"""
    values = servers(text)
    assert len(values) == 2
    assert values[0] == {"listen": [("0.0.0.0", 443)], "names": ["a.test", "b.test"]}
    assert values[1]["listen"] == [("::", 80)]


@pytest.mark.parametrize("value", ["*:80", "80", "0.0.0.0:80"])
def test_wildcard_listener(value):
    assert listener(value) == ("0.0.0.0", 80)


def test_config_dump():
    assert dump_files("# configuration file /etc/nginx/nginx.conf:\nhttp {}\n") == {
        "/etc/nginx/nginx.conf": "http {}\n"
    }
    with pytest.raises(Error):
        dump_files("unexpected")


@pytest.mark.parametrize(
    "manual",
    [
        "server {listen 18081; server_name other;}",
        "server {listen 127.0.0.1:18081; server_name 127.0.0.1;}",
        "server {listen [::]:18081; server_name other;}",
    ],
)
def test_manual_conflict(registry, manual):
    with patch("proxy_manager.nginx_controller.run", return_value=""):
        with pytest.raises(Error, match="conflict"):
            Controller({"ss": "ss"}).conflicts(registry, {"manual": manual}, {"applications": []})


def test_occupied_non_nginx_port(registry):
    with patch(
        "proxy_manager.nginx_controller.run", return_value="LISTEN 0 128 0.0.0.0:18081 0.0.0.0:*"
    ):
        with pytest.raises(Error, match="occupied"):
            Controller({"ss": "ss"}).conflicts(registry, {}, {"applications": []})


def test_reload_validates_first():
    controller = Controller(
        {"nginx": "nginx", "nginx_config": "/etc/nginx/nginx.conf", "systemctl": "systemctl"}
    )
    with patch("proxy_manager.nginx_controller.run", side_effect=Error("invalid")) as runner:
        with pytest.raises(Error):
            controller.reload()
        assert runner.call_count == 1
        assert runner.call_args.args[0][1] == "-t"


def test_candidate_clones_graph_without_touching_original(tmp_path):
    conf = tmp_path / "nginx.conf"
    directory = tmp_path / "conf.d"
    directory.mkdir()
    manual = directory / "manual.conf"
    manual.write_text("server { listen 80; }", encoding="utf-8")
    bridge = directory / "90-manager.conf"
    conf.write_text(
        f"events {{}}\nhttp {{\n include {directory.as_posix()}/*.conf;\n}}", encoding="utf-8"
    )
    originals = {str(conf): conf.read_text(), str(manual): manual.read_text()}
    # POSIX paths are the production platform; Windows glob paths differ.
    if __import__("os").name != "posix":
        pytest.skip("Linux Nginx include graph")
    controller = Controller({"nginx_config": str(conf), "bridge": str(bridge)})
    candidate = controller.candidate(originals, tmp_path / "release", tmp_path / "staged")
    assert "release/*.conf" in candidate.read_text()
    assert conf.read_text() == originals[str(conf)]
    assert manual.read_text() == originals[str(manual)]
    assert not bridge.exists()


def test_exclusive_lock(tmp_path):
    with lock(tmp_path):
        with pytest.raises(Error, match="Another manager"):
            with lock(tmp_path):
                pass


def test_scoped_operating_system_sockets():
    assert listener("127.0.0.53%lo:53") == ("127.0.0.53", 53)
    assert listener("[fd7a:115c:a1e0::1]%tailscale0:38826") == ("fd7a:115c:a1e0::1", 38826)


@pytest.mark.skipif(__import__("os").name != "posix", reason="Linux include graph")
def test_candidate_with_existing_manager_bridge(tmp_path):
    conf = tmp_path / "nginx.conf"
    confd = tmp_path / "conf.d"
    confd.mkdir()
    old = tmp_path / "old"
    old.mkdir()
    active = old / "common.conf"
    active.write_text("# old active", encoding="utf-8")
    release = tmp_path / "new"
    release.mkdir()
    (release / "new.conf").write_text("# staged candidate", encoding="utf-8")
    bridge = confd / "90-manager.conf"
    bridge.write_text(f"include {old}/*.conf;\n", encoding="utf-8")
    conf.write_text(f"events {{}}\nhttp {{\n include {confd}/*.conf;\n}}", encoding="utf-8")
    files = {str(p): p.read_text() for p in (conf, bridge, active)}
    controller = Controller({"nginx_config": str(conf), "bridge": str(bridge)})
    candidate = controller.candidate(files, release, tmp_path / "staged")
    assert f"include {release}/*.conf;" in candidate.read_text()
    assert bridge.read_text() == files[str(bridge)]
