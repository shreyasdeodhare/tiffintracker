"""A connector crash must not shut down the dashboard or inbox worker."""

import signal

from scripts import run_container


def test_connector_restart_does_not_stop_web(monkeypatch):
    clock = [0.0]
    launches = []
    kills = []
    handlers = {}
    ticks = [0]

    class Child:
        def __init__(self, command):
            self.command = command
            self.pid = 100 + len(launches)
            self.returncode = None
            self.first_connector = command[0] == "node" and not any(
                item.command[0] == "node" for item in launches
            )

        def poll(self):
            if self.first_connector:
                self.returncode = 1
            return self.returncode

        def wait(self, timeout=None):
            self.returncode = 0
            return 0

    def spawn(command, **kwargs):
        child = Child(command)
        launches.append(child)
        return child

    def sleep(seconds):
        clock[0] += seconds
        ticks[0] += 1
        if ticks[0] == 4:
            assert sum(child.command[0] == "gunicorn" for child in launches) == 1
            assert sum(child.command[0] == "python" for child in launches) == 1
            assert sum(child.command[0] == "node" for child in launches) == 2
            assert kills == []
            handlers[signal.SIGTERM](signal.SIGTERM, None)

    monkeypatch.setattr(run_container.subprocess, "Popen", spawn)
    monkeypatch.setattr(run_container.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(run_container.time, "sleep", sleep)
    monkeypatch.setattr(
        run_container.signal,
        "signal",
        lambda number, handler: handlers.__setitem__(number, handler),
    )
    monkeypatch.setattr(
        run_container.os, "killpg", lambda pid, sig: kills.append((pid, sig)), raising=False
    )
    run_container.main()
    assert len(kills) == 3  # All live children receive shutdown only after the stop request.
