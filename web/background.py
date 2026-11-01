"""Status and control of the background mail, labelling, and web jobs."""

import os
import plistlib
import re
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from src.background import job_control

MAIL_AGENT = "com.robinweiss.wildl-id-mail-downloader"
WEB_AGENT = "com.robinweiss.wildl-id-web"
LAUNCH_AGENTS_DIR = Path.home() / "Library/LaunchAgents"
DEFAULT_LOG_DIR = Path.home() / "Library/Logs/wildl-id"
LOG_TAIL_BYTES = 65536
RUN_HEADER = "WILDLIFE CAMERA IMAGE ANALYSIS SYSTEM"
EVENT_PREFIXES = (
    "Downloaded ",
    "Moving ",
    "Automatically labelling ",
    "No new images require",
    "Sent ",
    "✗",
    "Automatic mail job is paused",
    "Mail job already running",
)
PROGRESS = re.compile(r"(\d+)/(\d+) \[[^<\]]*<([^,\]]+)")


@dataclass
class MailJobStatus:
    running_pid: int | None
    paused: bool
    progress: tuple[int, int, str] | None
    events: list[str] = field(default_factory=list)
    last_activity: datetime | None = None


def _plist(plist_path):
    try:
        return plistlib.loads(Path(plist_path).read_bytes())
    except (FileNotFoundError, plistlib.InvalidFileException):
        return None


def _tail(path):
    try:
        with Path(path).open("rb") as source:
            source.seek(0, os.SEEK_END)
            source.seek(max(0, source.tell() - LOG_TAIL_BYTES))
            return source.read().decode("utf-8", errors="replace")
    except FileNotFoundError:
        return ""


def _log_paths(plist):
    plist = plist or {}
    return (
        Path(plist.get("StandardOutPath", DEFAULT_LOG_DIR / "stdout.log")),
        Path(plist.get("StandardErrorPath", DEFAULT_LOG_DIR / "stderr.log")),
    )


def mail_job_status(
    lock_path=job_control.LOCK_PATH,
    pause_path=job_control.PAUSE_PATH,
    plist_path=LAUNCH_AGENTS_DIR / f"{MAIL_AGENT}.plist",
):
    stdout_path, stderr_path = _log_paths(_plist(plist_path))
    latest_run = _tail(stdout_path).rsplit(RUN_HEADER, 1)[-1]
    events = [
        line.strip().split(" into ", 1)[0]
        for line in latest_run.splitlines()
        if line.strip().startswith(EVENT_PREFIXES)
    ]
    running_pid = job_control.running_pid(lock_path)
    progress = None
    if running_pid is not None:
        matches = PROGRESS.findall(_tail(stderr_path).replace("\r", "\n"))
        if matches:
            done, total, remaining = matches[-1]
            if done != total:
                progress = (int(done), int(total), remaining.strip())
    modified = [
        path.stat().st_mtime for path in (stdout_path, stderr_path) if path.exists()
    ]
    return MailJobStatus(
        running_pid=running_pid,
        paused=job_control.is_paused(pause_path),
        progress=progress,
        events=events,
        last_activity=datetime.fromtimestamp(max(modified)) if modified else None,
    )


def agent_status(label, runner=subprocess.run):
    try:
        result = runner(
            ["/bin/launchctl", "print", f"gui/{os.getuid()}/{label}"],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return "unavailable"
    if result.returncode != 0:
        return "not installed"
    fields = dict(re.findall(r"^\t(state|pid) = (.+)$", result.stdout, re.MULTILINE))
    state = fields.get("state", "unknown")
    return f"{state} (pid {fields['pid']})" if "pid" in fields else state


def run_now(
    plist_path=LAUNCH_AGENTS_DIR / f"{MAIL_AGENT}.plist",
    lock_path=job_control.LOCK_PATH,
    popen=subprocess.Popen,
):
    """Start one manual mail run with the installed agent's command and logs."""
    if job_control.running_pid(lock_path) is not None:
        return "Mail job is already running"
    plist = _plist(plist_path)
    if plist is None:
        return "Mail job is not installed on this machine"
    stdout_path, stderr_path = _log_paths(plist)
    with stdout_path.open("a") as stdout, stderr_path.open("a") as stderr:
        popen(
            [*plist["ProgramArguments"], "--manual"],
            cwd=plist.get("WorkingDirectory"),
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )
    return "Mail job started"
