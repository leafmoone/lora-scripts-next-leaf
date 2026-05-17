import sys
import tempfile
import unittest
from pathlib import Path

from mikazuki.anima_backend.upstream import (
    prefer_upstream_imports,
    resolve_upstream_path,
    upstream_entrypoint,
    verify_pinned_commit,
)


class AnimaBackendUpstreamTests(unittest.TestCase):
    def test_resolve_upstream_path_from_config(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "repo"
            upstream = root / "vendor" / "sd-scripts"
            upstream.mkdir(parents=True)
            config = root / "config" / "anima_backend.toml"
            config.parent.mkdir()
            config.write_text(
                '[backend]\nupstream_path = "vendor/sd-scripts"\nentrypoint = "anima_train_network.py"\n',
                encoding="utf-8",
            )

            self.assertEqual(resolve_upstream_path(root, config), upstream.resolve())

    def test_upstream_entrypoint_uses_configured_script(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            upstream = Path(temp_dir) / "vendor" / "sd-scripts"
            upstream.mkdir(parents=True)
            script = upstream / "anima_train_network.py"
            script.write_text("", encoding="utf-8")

            self.assertEqual(upstream_entrypoint(upstream, "anima_train_network.py"), script)

    def test_upstream_entrypoint_fails_when_missing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            upstream = Path(temp_dir)

            with self.assertRaises(FileNotFoundError):
                upstream_entrypoint(upstream, "anima_train_network.py")

    def test_prefer_upstream_imports_places_path_first(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            upstream = Path(temp_dir).resolve()
            original = list(sys.path)
            try:
                prefer_upstream_imports(upstream)

                self.assertEqual(sys.path[0], str(upstream))
                self.assertEqual(sys.path.count(str(upstream)), 1)
            finally:
                sys.path[:] = original

    def test_verify_pinned_commit_matches_submodule_head(self):
        root = Path.cwd()

        self.assertEqual(
            verify_pinned_commit(root),
            "502cc3fab2aa22c106580e2e05c4692cfde5e5ff",
        )


if __name__ == "__main__":
    unittest.main()
