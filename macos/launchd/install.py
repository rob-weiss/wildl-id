"""Install or uninstall the per-user email download LaunchAgent."""

import argparse
import os
import plistlib
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

LABEL = "com.robinweiss.wildl-id-mail-downloader"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEMPLATE_PATH = Path(__file__).with_name(f"{LABEL}.plist.template")
LAUNCH_AGENTS_DIR = Path.home() / "Library/LaunchAgents"
LOG_DIR = Path.home() / "Library/Logs/wildl-id"


def install_agent(
    *,
    project_root: Path | None = None,
    interpreter: str | None = None,
    launch_agents_dir: Path | None = None,
    log_dir: Path | None = None,
    template_path: Path | None = None,
    launchctl_runner: Callable | None = None,
):
    project_root = Path(project_root or PROJECT_ROOT).resolve()
    interpreter = str(Path(interpreter or sys.executable).absolute())
    launch_agents_dir = Path(launch_agents_dir or LAUNCH_AGENTS_DIR).expanduser()
    log_dir = Path(log_dir or LOG_DIR).expanduser()
    template_path = Path(template_path or TEMPLATE_PATH)
    plist_path = launch_agents_dir / f"{LABEL}.plist"
    script_path = project_root / "wildl_id.py"
    if not script_path.is_file():
        raise FileNotFoundError(f"project entry point not found: {script_path}")

    template = plistlib.loads(template_path.read_bytes())
    replacements = {
        "__PYTHON__": interpreter,
        "__SCRIPT__": str(script_path),
        "__PROJECT_ROOT__": str(project_root),
        "__STDOUT__": str(log_dir / "stdout.log"),
        "__STDERR__": str(log_dir / "stderr.log"),
    }
    plist = _replace_placeholders(template, replacements)
    if plist.get("Label") != LABEL:
        raise ValueError("LaunchAgent template has an unexpected label")
    if plist.get("ProgramArguments") != [
        interpreter,
        str(script_path),
        "--download",
        "--notify",
    ]:
        raise ValueError("LaunchAgent template has unexpected program arguments")

    launch_agents_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    temporary = plist_path.with_name(f".{plist_path.name}.tmp")
    try:
        temporary.write_bytes(plistlib.dumps(plist, sort_keys=False))
        os.replace(temporary, plist_path)
    finally:
        temporary.unlink(missing_ok=True)

    runner = launchctl_runner or subprocess.run
    domain = f"gui/{os.getuid()}"
    runner(
        ["/bin/launchctl", "bootout", domain, str(plist_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    try:
        runner(
            ["/bin/launchctl", "bootstrap", domain, str(plist_path)],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as error:
        detail = (error.stderr or "").strip()
        message = f"could not load LaunchAgent at {plist_path}"
        if detail:
            message += f": {detail}"
        raise RuntimeError(message) from error
    return plist_path


def uninstall_agent(
    *,
    launch_agents_dir: Path | None = None,
    launchctl_runner: Callable | None = None,
):
    plist_path = (
        Path(launch_agents_dir or LAUNCH_AGENTS_DIR).expanduser() / f"{LABEL}.plist"
    )
    runner = launchctl_runner or subprocess.run
    runner(
        ["/bin/launchctl", "bootout", f"gui/{os.getuid()}", str(plist_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    plist_path.unlink(missing_ok=True)
    return plist_path


def _replace_placeholders(value, replacements):
    if isinstance(value, dict):
        return {
            key: _replace_placeholders(item, replacements)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_replace_placeholders(item, replacements) for item in value]
    if isinstance(value, str):
        return replacements.get(value, value)
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    subparsers.add_parser("install", help="install and load the LaunchAgent")
    subparsers.add_parser("uninstall", help="unload and remove the LaunchAgent")
    args = parser.parse_args(argv)
    try:
        if args.action == "install":
            path = install_agent()
            print(f"Installed LaunchAgent: {path}")
        else:
            path = uninstall_agent()
            print(f"Removed LaunchAgent: {path}")
    except (OSError, RuntimeError, ValueError) as error:
        parser.exit(1, f"{parser.prog}: error: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
