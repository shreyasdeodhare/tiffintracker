import os
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from proxy_manager.deployment import EMPTY, Manager
from proxy_manager.storage import load, write_json
from proxy_manager.validator import Error

pytestmark = pytest.mark.skipif(
    os.name != "posix", reason="Atomic Linux symlinks and fsync verified on Ubuntu"
)


@pytest.fixture
def manager(tmp_path, app):
    root = tmp_path / "owned"
    root.mkdir(mode=0o700)
    controller = Mock()
    controller.snapshot.return_value = ({}, {}, {})
    controller.candidate.return_value = tmp_path / "candidate.conf"
    controller.active.return_value = True
    manager = Manager(
        {
            "root": str(root),
            "bridge": str(tmp_path / "90-manager.conf"),
            "certificate_roots": [],
            "approved_upstreams": [app["upstream"]],
            "audit_log": str(tmp_path / "audit.log"),
        },
        controller,
    )
    manager.initialize()
    Path(manager.settings["bridge"]).write_text(manager.bridge(), encoding="utf-8")
    manager.network_preflight = Mock()
    manager.health = Mock(return_value={"app": {"upstream": {"ok": True}, "proxy": {"ok": True}}})
    manager.canaries = Mock(return_value=[{"ok": True}])
    return manager


def test_dry_run_no_live_change(manager, registry):
    old = manager.current()
    write_json(manager.root / "applications.json", registry)
    result = manager.apply(dry_run=True)
    assert result["dry_run"]
    assert manager.current() == old
    manager.controller.reload.assert_not_called()
    assert not list((manager.root / "backups").iterdir())


def test_apply_backup_and_rollback(manager, registry):
    old = manager.current()
    write_json(manager.root / "applications.json", registry)
    result = manager.apply()
    assert manager.current() == result["release"] != old
    assert Path(result["backup"]).is_dir()
    assert load(Path(result["backup"]) / "snapshot.json") == EMPTY
    assert manager.rollback()["verified"]
    assert manager.current() == old
    assert load(manager.root / "applications.json") == registry


def test_failed_stage_validation_never_switches(manager, registry):
    old = manager.current()
    write_json(manager.root / "applications.json", registry)
    manager.controller.test.side_effect = Error("invalid Nginx")
    with pytest.raises(Error, match="invalid Nginx"):
        manager.apply()
    assert manager.current() == old
    manager.controller.reload.assert_not_called()


def test_failed_reload_restores_verified_previous(manager, registry):
    old = manager.current()
    write_json(manager.root / "applications.json", registry)
    manager.controller.reload.side_effect = [Error("reload failed"), None]
    with pytest.raises(Error, match="rollback verified"):
        manager.apply()
    assert manager.current() == old
    assert manager.controller.reload.call_count == 2
    assert not (manager.root / "pending.json").exists()


def test_failed_health_rolls_back(manager, registry):
    old = manager.current()
    write_json(manager.root / "applications.json", registry)
    manager.verified_health = Mock(side_effect=[{}, Error("bad proxy"), {}])
    with pytest.raises(Error, match="rollback verified"):
        manager.apply()
    assert manager.current() == old


def test_rollback_failure_retains_journal(manager, registry):
    write_json(manager.root / "applications.json", registry)
    manager.controller.reload.side_effect = Error("cannot reload")
    with pytest.raises(Error, match="rollback NOT verified"):
        manager.apply()
    assert (manager.root / "pending.json").exists()
    with pytest.raises(Error, match="Interrupted"):
        manager.apply()


def test_recover_interrupted_transaction(manager, registry):
    old = manager.current()
    new = manager.new_release(registry)
    write_json(manager.root / "pending.json", {"old": old, "new": new, "manual_hashes": {}})
    manager.point(new)
    assert manager.rollback()["verified"]
    assert manager.current() == old
    assert not (manager.root / "pending.json").exists()


def test_modified_owned_file_refused(manager):
    path = manager.release(manager.current()) / "00-manager-common.conf"
    path.write_text("# manual change", encoding="utf-8")
    with pytest.raises(Error, match="modified"):
        manager.current()


def test_unowned_bridge_refused(manager):
    Path(manager.settings["bridge"]).write_text("# manual", encoding="utf-8")
    with pytest.raises(Error, match="manager-owned"):
        manager.apply()


def test_registry_invalid_after_failure_does_not_prevent_recovery(manager):
    from proxy_manager.cli import execute, parser

    old = manager.current()
    write_json(manager.root / "pending.json", {"old": old, "new": old, "manual_hashes": {}})
    (manager.root / "applications.json").write_text("broken", encoding="utf-8")
    assert execute(parser().parse_args(["rollback", "--yes"]), manager)["verified"]


def test_unapproved_upstream(manager, registry):
    registry["applications"][0]["upstream"]["port"] = 3000
    write_json(manager.root / "applications.json", registry)
    with pytest.raises(Error, match="explicitly approved"):
        manager.registry()


def test_preexisting_preservation_failure_prevents_apply(manager, registry):
    old = manager.current()
    write_json(manager.root / "applications.json", registry)
    manager.canaries.return_value = [{"ok": False}]
    with pytest.raises(Error, match="preservation"):
        manager.apply()
    assert manager.current() == old
    manager.controller.reload.assert_not_called()


def test_registry_changes_do_not_reload(manager, registry):
    from proxy_manager.cli import execute, parser

    write_json(manager.root / "applications.json", registry)
    result = execute(parser().parse_args(["disable", "example-api"]), manager)
    assert result["live_nginx_unchanged"]
    assert not manager.registry()["applications"][0]["enabled"]
    manager.controller.reload.assert_not_called()


def test_remove_requires_confirmation(manager, registry):
    from proxy_manager.cli import execute, parser

    write_json(manager.root / "applications.json", registry)
    with patch("sys.stdin.isatty", return_value=False):
        with pytest.raises(Error, match="Confirmation"):
            execute(parser().parse_args(["remove", "example-api"]), manager)


def test_release_path_escape(manager):
    with pytest.raises(Error):
        manager.release("../../etc/nginx")


def test_insecure_path_never_opens_audit_log(manager):
    from proxy_manager.cli import main

    with (
        patch("proxy_manager.cli.os.geteuid", return_value=0),
        patch("proxy_manager.cli.settings_from", return_value=manager.settings),
        patch("proxy_manager.cli.Manager", return_value=manager),
        patch("proxy_manager.cli.protected", side_effect=Error("insecure path")),
    ):
        manager.log = Mock()
        assert main(["validate"]) == 1
        manager.log.assert_not_called()


def test_health_remains_available_with_invalid_desired_registry_and_pending_journal(manager):
    from proxy_manager.cli import execute, parser

    (manager.root / "applications.json").write_text("broken", encoding="utf-8")
    write_json(manager.root / "pending.json", {"old": manager.current()})
    result = execute(parser().parse_args(["health"]), manager)
    assert result["nginx_active"]
