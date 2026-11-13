"""Protected state, atomic writes, strict JSON and a cross-process deployment lock."""

import contextlib
import hashlib
import json
import logging
import logging.handlers
import os
import tempfile
import time
from pathlib import Path

from .validator import Error


def pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise Error("Duplicate JSON key")
        result[key] = value
    return result


def load(path):
    try:
        if Path(path).stat().st_size > 1_000_000:
            raise Error("JSON exceeds 1 MB limit")
        return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=pairs)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Error(f"Cannot read JSON: {type(exc).__name__}") from exc


def write(path, data):
    path = Path(path)
    if path.is_symlink():
        raise Error("Refusing to replace a symlink file")
    fd, temporary = tempfile.mkstemp(prefix=".pm-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        if os.name == "posix":
            directory = os.open(path.parent, os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_json(path, value):
    write(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def protected(path):
    """Every ancestor of privileged input must be root-owned and not writable by others."""
    path = Path(path).resolve(strict=True)
    if os.name != "posix":
        return
    for item in [path, *path.parents]:
        stat = item.stat()
        if stat.st_uid != 0 or stat.st_mode & 0o022:
            raise Error(f"Insecure privileged path: {item}")


@contextlib.contextmanager
def lock(root):
    path = Path(root) / "deploy.lock"
    if path.is_symlink():
        raise Error("Unsafe lock path")
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        if os.name == "posix":
            import fcntl

            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise Error("Another manager operation holds the deployment lock") from None
        else:
            import msvcrt

            if os.fstat(fd).st_size == 0:
                os.write(fd, b"0")
            os.lseek(fd, 0, 0)
            try:
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            except OSError:
                raise Error("Another manager operation holds the deployment lock") from None
        yield
    finally:
        os.close(fd)


def audit(path, event, **details):
    logger = logging.getLogger("proxy-manager")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = logging.handlers.RotatingFileHandler(
        path, maxBytes=2_000_000, backupCount=5, encoding="utf-8"
    )
    formatter = logging.Formatter("%(asctime)sZ %(message)s")
    formatter.converter = time.gmtime
    handler.setFormatter(formatter)
    try:
        logger.addHandler(handler)
        logger.info(json.dumps({"event": event, **details}, sort_keys=True))
    finally:
        logger.removeHandler(handler)
        handler.close()
