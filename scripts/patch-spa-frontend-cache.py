#!/usr/bin/env python3
"""Fix SPA tagger route + cache bust for patched frontend assets."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "frontend" / "dist"
VERSION = (ROOT / "VERSION").read_text(encoding="utf-8").strip()


def read_version() -> str:
    if VERSION:
        return VERSION
    return "0"


def patch_html_navigation_and_scripts() -> None:
    ver = read_version()
    count_md = 0
    count_script = 0
    for path in DIST.rglob("*.html"):
        html = path.read_text(encoding="utf-8")
        original = html
        html = html.replace('href="/tagger.md"', 'href="/tagger.html"')
        if html != original:
            count_md += 1
        html = re.sub(
            r'src="/assets/sd-trainer-brand\.js(\?v=[^"]*)?"',
            f'src="/assets/sd-trainer-brand.js?v={ver}"',
            html,
        )
        html = re.sub(
            r'src="/assets/tagger-progress\.js(\?v=[^"]*)?"',
            f'src="/assets/tagger-progress.js?v={ver}"',
            html,
        )
        if "tagger-progress.js" in html and f"tagger-progress.js?v={ver}" not in html:
            html = html.replace(
                '<script type="module" src="/assets/app.',
                f'<script src="/assets/tagger-progress.js?v={ver}" defer></script>\n    <script type="module" src="/assets/app.',
                1,
            )
        if html != original or f'sd-trainer-brand.js?v={ver}' in html:
            path.write_text(html, encoding="utf-8")
            count_script += 1
    print(f"patched tagger.md->tagger.html in {count_md} file(s)")
    print(f"updated script cache bust (v={ver}) in html files")


def main() -> None:
    patch_html_navigation_and_scripts()


if __name__ == "__main__":
    main()
