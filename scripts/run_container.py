"""Supervise web, one WhatsApp receiver, and one durable queue worker in one container."""

import os
import signal
import subprocess
import sys
import time


def main():
    commands = [
        [
            "gunicorn",
            "--bind=0.0.0.0:8000",
            "--workers=2",
            "--threads=2",
            "--timeout=60",
            "--graceful-timeout=30",
            "--access-logfile=-",
            "--error-logfile=-",
            "wsgi:app",
        ],
        ["node", "--max-old-space-size=160", "bridge/index.mjs"],
        ["python", "-m", "scripts.whatsapp_worker"],
    ]
    children = []
    names = ["web", "whatsapp", "inbox_worker"]
    attempts = [0] * len(commands)
    start_times = [0.0] * len(commands)
    next_starts = [0.0] * len(commands)
    stopping = False

    def stop(_signal=None, _frame=None):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        children = [None] * len(commands)
        while not stopping:
            now = time.monotonic()
            for index, command in enumerate(commands):
                child = children[index]
                if child is not None and child.poll() is not None:
                    if now - start_times[index] > 60:
                        attempts[index] = 0
                    attempts[index] += 1
                    delay = min(60, 2 ** min(attempts[index] - 1, 6))
                    print(
                        f"component={names[index]} exited code={child.returncode} retry_seconds={delay}",
                        file=sys.stderr,
                        flush=True,
                    )
                    next_starts[index] = now + delay
                    children[index] = None
                if children[index] is None and now >= next_starts[index]:
                    children[index] = subprocess.Popen(command, start_new_session=True)
                    start_times[index] = now
            time.sleep(1)
    finally:
        for child in children:
            if child is not None and child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
        deadline = time.monotonic() + 30
        for child in children:
            if child is None:
                continue
            try:
                child.wait(timeout=max(1, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)


if __name__ == "__main__":
    main()
