"""Single-instance lock and pause switch for the scheduled mail job."""

import fcntl
import os
from contextlib import contextmanager
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parents[2] / "logs"
LOCK_PATH = LOG_DIR / "mail-job.lock"
PAUSE_PATH = LOG_DIR / "mail-job.paused"


@contextmanager
def exclusive_run(path=LOCK_PATH):
    """Yield True while holding the lock, or False if another run holds it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        try:
            handle.truncate(0)
            handle.write(str(os.getpid()))
            handle.flush()
            yield True
        finally:
            handle.truncate(0)
            fcntl.flock(handle, fcntl.LOCK_UN)


def running_pid(path=LOCK_PATH):
    """Return the PID of the run holding the lock, or None when idle."""
    path = Path(path)
    try:
        handle = path.open("r", encoding="utf-8")
    except FileNotFoundError:
        return None
    with handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            text = handle.read().strip()
            return int(text) if text.isdigit() else None
        fcntl.flock(handle, fcntl.LOCK_UN)
        return None


def is_paused(path=PAUSE_PATH):
    return Path(path).exists()


def set_paused(paused, path=PAUSE_PATH):
    path = Path(path)
    if paused:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    else:
        path.unlink(missing_ok=True)
