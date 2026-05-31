from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "cli"))
import install_anima_fast as cli  # noqa: E402


class InstallAnimaFastCliTests(unittest.TestCase):
    def test_find_project_root_from_repo(self):
        root = cli.find_project_root(Path(__file__).resolve().parents[2])
        self.assertTrue((root / "gui.py").is_file())
        self.assertTrue((root / "config" / "anima_fast_backend.toml").is_file())

    def test_dry_run_prints_plan(self):
        with tempfile.TemporaryDirectory() as td:
            project = Path(td)
            (project / "gui.py").write_text("", encoding="utf-8")
            (project / "config").mkdir()
            (project / "config" / "anima_fast_backend.toml").write_text(
                '[backend]\nsource_commit = "abc123"\n', encoding="utf-8"
            )
            env_dir = project / "config" / "anima_fast_environment"
            env_dir.mkdir()
            (env_dir / "anima-constraints-cu130.txt").write_text("torch\n", encoding="utf-8")
            (env_dir / "anima-overrides-cu130.txt").write_text("numpy\n", encoding="utf-8")
            source = project / "upstream"
            source.mkdir()
            (source / "train.py").write_text("print('ok')\n", encoding="utf-8")

            fake_plan = mock.Mock()
            fake_plan.as_dict.return_value = {}
            with mock.patch.object(cli, "resolve_source_root", return_value=source), mock.patch.object(
                cli.os, "chdir"
            ), mock.patch(
                "mikazuki.anima_fast_backend.settings.feature_enabled", return_value=True
            ), mock.patch(
                "mikazuki.anima_fast_backend.settings.discover_runtime",
                return_value=mock.Mock(source_commit="abc123"),
            ), mock.patch(
                "mikazuki.anima_fast_backend.environment.build_environment_install_plan",
                return_value=fake_plan,
            ):
                rc = cli.main(["--project-root", str(project), "--dry-run"])
            self.assertEqual(rc, 0)
