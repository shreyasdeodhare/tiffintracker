from unittest.mock import Mock, patch

from proxy_manager.health_checker import check_app, probe


def test_expected_redirect_no_follow():
    connection = Mock()
    connection.getresponse.return_value.status = 302
    with (
        patch("socket.create_connection"),
        patch("http.client.HTTPConnection", return_value=connection),
    ):
        result = probe("127.0.0.1", 8001, "/", [302])
    assert result["ok"] and result["tcp"] and result["status"] == 302
    assert connection.request.call_count == 1
    connection.getresponse.return_value.read.assert_not_called()


def test_bad_http_not_healthy():
    connection = Mock()
    connection.getresponse.return_value.status = 500
    with (
        patch("socket.create_connection"),
        patch("http.client.HTTPConnection", return_value=connection),
    ):
        assert not probe("127.0.0.1", 8001, "/", [200])["ok"]


def test_tcp_failure_not_healthy():
    with patch("socket.create_connection", side_effect=ConnectionRefusedError):
        result = probe("127.0.0.1", 8001, "/", [200])
    assert not result["ok"] and not result["tcp"]
    assert result["error"] == "ConnectionRefusedError"


def test_disabled_probe_is_not_certified(app):
    app["health_check"]["enabled"] = False
    assert check_app(app)["skipped"]
    assert not check_app(app)["ok"]


def test_tls_verification_enabled():
    connection = Mock()
    connection.getresponse.return_value.status = 200
    with (
        patch("socket.create_connection"),
        patch("ssl.create_default_context") as context,
        patch("proxy_manager.health_checker.PinnedHTTPS", return_value=connection),
    ):
        assert probe(
            "app.test", 18443, "/", [200], tls=True, ca_file="/cert/ca.crt", address="192.168.1.101"
        )["ok"]
    context.assert_called_once_with(cafile="/cert/ca.crt")
