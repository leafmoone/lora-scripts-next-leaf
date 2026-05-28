#!/usr/bin/env python3
"""Tagger progress UI: script tag + CSS sync into style bundle. Dock built at runtime."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "frontend" / "dist"
BUILD_VERSION = (ROOT / "VERSION").read_text(encoding="utf-8").strip() or "0"
ASSETS = DIST / "assets"
TAGGER_HTML = DIST / "tagger.html"
TAGGER_VUE_JS = ASSETS / "tagger.html.0daaef4e.js"
TAGGER_PROGRESS_JS = ASSETS / "tagger-progress.js"
POLISH_CSS = ASSETS / "sd-trainer-ui-polish.css"
STYLE_CSS = ASSETS / "style.874872ce.css"
MARKER = "sd-tagger-dock"
LEGACY_MARKERS = ("sd-tagger-status", "sd-tagger-dock")


def sync_polish_into_style() -> None:
    polish = POLISH_CSS.read_text(encoding="utf-8")
    block_match = re.search(
        r"/\* ----- Tagger：底部操作坞.*",
        polish,
        re.DOTALL,
    )
    if not block_match:
        print("skip style sync: tagger dock block missing in polish css")
        return
    block = block_match.group(0).rstrip() + "\n"
    style = STYLE_CSS.read_text(encoding="utf-8")
    if MARKER in style:
        style = re.sub(
            r"/\* ----- Tagger：.*",
            block.rstrip(),
            style,
            count=1,
            flags=re.DOTALL,
        )
    else:
        if not style.endswith("\n"):
            style += "\n"
        style += "\n" + block
    STYLE_CSS.write_text(style, encoding="utf-8")
    print("synced tagger dock css into style.874872ce.css")


def strip_legacy_status_html(html: str) -> str:
    html = re.sub(
        r'<section class="sd-tagger-status"[^>]*>.*?</section>',
        "",
        html,
        flags=re.DOTALL,
    )
    html = re.sub(
        r'<footer class="sd-tagger-dock"[^>]*>.*?</footer>',
        "",
        html,
        flags=re.DOTALL,
    )
    return html


def patch_tagger_html() -> None:
    html = strip_legacy_status_html(TAGGER_HTML.read_text(encoding="utf-8"))
    cache_key = int(TAGGER_PROGRESS_JS.stat().st_mtime)
    tagger_src = f'/assets/tagger-progress.js?v={BUILD_VERSION}-{cache_key}'
    style_cache_key = int(STYLE_CSS.stat().st_mtime)
    style_src = f'/assets/style.874872ce.css?v={BUILD_VERSION}-{style_cache_key}'
    html = re.sub(
        r'href="/assets/style\.874872ce\.css(?:\?v=[^"]+)?"',
        f'href="{style_src}"',
        html,
        count=1,
    )
    if re.search(r'<script src="/assets/tagger-progress\.js\?v=[^"]+" defer></script>', html):
        html = re.sub(
            r'<script src="/assets/tagger-progress\.js\?v=[^"]+" defer></script>',
            f'<script src="{tagger_src}" defer></script>',
            html,
            count=1,
        )
    else:
        html = html.replace(
            '<script type="module" src="/assets/app.',
            f'<script src="{tagger_src}" defer></script>\n    <script type="module" src="/assets/app.',
            1,
        )
    html = html.replace('href="/tagger.md"', 'href="/tagger.html"')
    TAGGER_HTML.write_text(html, encoding="utf-8")
    print("patched tagger.html (script only, dock is runtime)")


def patch_tagger_vue_js() -> None:
    js = TAGGER_VUE_JS.read_text(encoding="utf-8")
    for marker in LEGACY_MARKERS:
        if marker in js:
            js = re.sub(
                r'<section class="sd-tagger[^"]*"[^>]*>.*?</section>',
                "",
                js,
            )
            TAGGER_VUE_JS.write_text(js, encoding="utf-8")
            print("stripped legacy status HTML from tagger vue js")
            return
    print("tagger vue js: no legacy status block")


def main() -> None:
    patch_tagger_html()
    patch_tagger_vue_js()
    sync_polish_into_style()


if __name__ == "__main__":
    main()
