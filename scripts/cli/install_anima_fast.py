#!/usr/bin/env python3
"""Install Anima LoRA Fast plugin environment without WebUI (CLI / portable)."""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path


def ensure_project_import_path(project_root: Path) -> None:
    root = str(project_root.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)


def find_project_root(start: Path | None = None) -> Path:
    candidates = [start or Path.cwd()]
    here = Path(__file__).resolve().parent
    candidates.append(here.parent.parent)
    for base in candidates:
        root = base.resolve()
        if (root / "gui.py").is_file() and (root / "config" / "anima_fast_backend.toml").is_file():
            return root
    raise SystemExit(
        "Cannot locate SD-Trainer project root (need gui.py and config/anima_fast_backend.toml). "
        "Run from repo / SD-Trainer directory or pass --project-root."
    )


def resolve_source_root(project_root: Path, explicit: Path | None, source_commit: str | None) -> Path:
    from mikazuki.anima_fast_backend.source_root import InstallSourceError, resolve_install_source_root

    try:
        return resolve_install_source_root(
            project_root,
            explicit,
            source_commit,
            allow_clone=True,
            log=print,
        )
    except InstallSourceError as exc:
        raise SystemExit(str(exc)) from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Install Anima LoRA Fast plugin (extensions/anima_lora) for CLI training."
    )
    parser.add_argument("--project-root", type=Path, default=None, help="SD-Trainer root (default: auto-detect)")
    parser.add_argument("--source-root", type=Path, default=None, help="Existing sorryhyun/anima_lora clone")
    parser.add_argument("--source-commit", default="", help="Pin upstream commit (default: config/anima_fast_backend.toml)")
    parser.add_argument("--dry-run", action="store_true", help="Print install plan only")
    args = parser.parse_args(argv)

    project_root = (args.project_root or find_project_root()).resolve()
    ensure_project_import_path(project_root)
    os.chdir(project_root)

    from mikazuki.anima_fast_backend.environment import build_environment_install_plan, install_environment
    from mikazuki.anima_fast_backend.extension_state import default_layout, read_extension_status
    from mikazuki.anima_fast_backend.settings import discover_runtime, feature_enabled

    if not feature_enabled():
        raise SystemExit("Anima Fast is disabled (LORA_ENABLE_ANIMA_FAST=0).")

    runtime = discover_runtime(lora_next_root=project_root)
    commit = (args.source_commit or runtime.source_commit or "").strip() or None
    source_root = resolve_source_root(project_root, args.source_root, commit)
    layout = default_layout(project_root)
    plan = build_environment_install_plan(
        project_root, layout, source_root, dry_run=args.dry_run, source_commit=commit
    )

    print(f"Project root : {project_root}")
    print(f"Source root  : {source_root}")
    print(f"Target source: {layout.source}")
    print(f"Venv python  : {layout.venv_python}")
    if commit:
        print(f"Pin commit   : {commit}")

    if args.dry_run:
        print("[dry-run] No changes made.")
        return 0

    if not shutil.which("uv"):
        raise SystemExit(
            "uv is required but not found in PATH. Install: https://docs.astral.sh/uv/getting-started/installation/"
        )

    def log(line: str) -> None:
        print(line, flush=True)

    result = install_environment(plan, log)
    status = read_extension_status(layout)
    print(f"Status: {status.state} ({status.reason})")
    if not result.ok:
        for err in result.errors:
            print(f"[error] {err}", file=sys.stderr)
        return 1
    print("")
    print("Fast plugin ready. Train with:")
    if sys.platform == "win32":
        print(r"  scripts\cli\train_anima_fast_by_toml.bat <config.toml>")
    else:
        print("  bash scripts/cli/train_anima_fast_by_toml.sh <config.toml>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
