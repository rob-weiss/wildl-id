import plistlib
import sys
import tempfile
import unittest
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

installer = import_module("macos.launchd.install")


class LaunchAgentInstallerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.project_root = self.root / "project root"
        self.project_root.mkdir()
        (self.project_root / "wildl_id.py").write_text("pass\n", encoding="utf-8")
        self.launch_agents_dir = self.root / "LaunchAgents"
        self.logs_dir = self.root / "Logs" / "wildl-id"
        self.calls = []

    def launchctl(self, arguments, **kwargs):
        self.calls.append(arguments)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    def install(self, interpreter="/env/bin/python"):
        return installer.install_agent(
            project_root=self.project_root,
            interpreter=interpreter,
            launch_agents_dir=self.launch_agents_dir,
            log_dir=self.logs_dir,
            launchctl_runner=self.launchctl,
        )

    def test_install_renders_absolute_paths_and_one_minute_schedule(self):
        plist_path = self.install()
        with plist_path.open("rb") as source:
            plist = plistlib.load(source)

        self.assertEqual(
            plist["ProgramArguments"],
            [
                "/env/bin/python",
                str((self.project_root / "wildl_id.py").resolve()),
                "--download",
                "--notify",
            ],
        )
        self.assertEqual(plist["WorkingDirectory"], str(self.project_root.resolve()))
        self.assertIs(plist["RunAtLoad"], True)
        self.assertEqual(plist["StartInterval"], 60)
        self.assertEqual(plist["StandardOutPath"], str(self.logs_dir / "stdout.log"))
        self.assertEqual(plist["StandardErrorPath"], str(self.logs_dir / "stderr.log"))
        self.assertEqual(self.calls[-1][0:2], ["/bin/launchctl", "bootstrap"])

    def test_generated_plist_contains_no_credentials(self):
        plist_path = self.install()
        content = plist_path.read_text(encoding="utf-8").lower()

        self.assertNotIn("password", content)
        self.assertNotIn("secret", content)
        self.assertNotIn("wildlife@robinweiss.de", content)

    def test_install_preserves_virtual_environment_interpreter_symlink(self):
        target = self.root / "python3"
        target.write_text("interpreter target", encoding="utf-8")
        interpreter = self.root / "venv" / "bin" / "python"
        interpreter.parent.mkdir(parents=True)
        interpreter.symlink_to(target)

        plist_path = self.install(str(interpreter))
        with plist_path.open("rb") as source:
            plist = plistlib.load(source)

        self.assertEqual(plist["ProgramArguments"][0], str(interpreter))

    def test_uninstall_boots_out_and_removes_generated_plist(self):
        plist_path = self.install()
        self.calls.clear()

        removed = installer.uninstall_agent(
            launch_agents_dir=self.launch_agents_dir,
            launchctl_runner=self.launchctl,
        )

        self.assertEqual(removed, plist_path)
        self.assertFalse(plist_path.exists())
        self.assertEqual(self.calls[0][0:2], ["/bin/launchctl", "bootout"])


if __name__ == "__main__":
    unittest.main()
