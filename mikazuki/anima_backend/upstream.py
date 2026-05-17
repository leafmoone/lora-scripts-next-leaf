from __future__ import annotations

import sys
import subprocess
from pathlib import Path


def load_toml_file(path: Path) -> dict:
    try:
        import tomllib

        with path.open("rb") as config_file:
            return tomllib.load(config_file)
    except ModuleNotFoundError:
        import toml

        return toml.load(path)


def resolve_upstream_path(root: Path, config_path: Path | None = None) -> Path:
    config_path = config_path or root / "config" / "anima_backend.toml"
    data = load_toml_file(config_path)

    upstream_path = Path(data["backend"]["upstream_path"])
    if not upstream_path.is_absolute():
        upstream_path = root / upstream_path
    return upstream_path.resolve()


def pinned_commit(root: Path, config_path: Path | None = None) -> str:
    config_path = config_path or root / "config" / "anima_backend.toml"
    data = load_toml_file(config_path)
    return str(data["backend"]["pinned_commit"])


def current_upstream_commit(upstream_path: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(upstream_path), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=True,
    )
    return result.stdout.strip()


def verify_pinned_commit(root: Path, config_path: Path | None = None) -> str:
    upstream_path = resolve_upstream_path(root, config_path)
    expected = pinned_commit(root, config_path)
    actual = current_upstream_commit(upstream_path)
    if actual != expected:
        raise RuntimeError(
            "Pinned sd-scripts commit mismatch: "
            f"config expects {expected}, but {upstream_path} is at {actual}"
        )
    return actual


def upstream_entrypoint(upstream_path: Path, entrypoint: str) -> Path:
    script = upstream_path / entrypoint
    if not script.exists():
        raise FileNotFoundError(f"Upstream Anima trainer not found: {script}")
    return script


def prefer_upstream_imports(upstream_path: Path) -> None:
    upstream = str(upstream_path.resolve())
    if upstream in sys.path:
        sys.path.remove(upstream)
    sys.path.insert(0, upstream)
