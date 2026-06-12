#!/usr/bin/env python3
"""Build SQLite character alias index from JSON resources."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

from anima_caption_pipeline.alias_index import (  # noqa: E402
    CHARACTER_ALIASES_PATH,
    DEFAULT_DB_PATH,
    build_sqlite_index,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build character alias SQLite index")
    parser.add_argument(
        "--json",
        type=Path,
        default=CHARACTER_ALIASES_PATH,
        help="Source danbooru_character_aliases.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_DB_PATH,
        help="Output SQLite database path",
    )
    args = parser.parse_args()
    if not args.json.is_file():
        print(f"JSON not found: {args.json}", file=sys.stderr)
        return 1
    count = build_sqlite_index(args.json, args.output)
    print(f"Wrote {count} alias rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
