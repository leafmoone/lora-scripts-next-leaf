"""Tag-Edit-Leaf helper tests (no GPU / no live API)."""

import json
from io import BytesIO
from pathlib import Path

from PIL import Image

from mikazuki.utils.tag_edit_leaf_helpers import parse_progress_line, prepare_image_for_api


def test_prepare_image_for_api_keeps_small_jpeg(tmp_path: Path):
    img_path = tmp_path / "small.jpg"
    img_path.write_bytes(b"\xff\xd8\xff" + b"x" * 100)

    data, subtype = prepare_image_for_api(img_path, max_bytes=4096, max_edge=512)
    assert subtype == "jpeg"
    assert data == img_path.read_bytes()


def test_prepare_image_for_api_resizes_large_image(tmp_path: Path):
    img_path = tmp_path / "large.bmp"
    Image.new("RGB", (3000, 2000), color=(255, 0, 0)).save(img_path)

    data, subtype = prepare_image_for_api(img_path, max_bytes=512 * 1024, max_edge=1536)
    assert subtype == "jpeg"
    assert len(data) <= 512 * 1024
    with Image.open(BytesIO(data)) as img:
        assert max(img.size) <= 1536


def test_parse_progress_line_json():
    state = {"progress": 0, "phase": "", "message": ""}
    line = json.dumps(
        {"type": "progress", "phase": "tagging", "current": 3, "total": 10, "message": "Tagging 3/10"},
        ensure_ascii=False,
    )
    assert parse_progress_line(line, state) is True
    assert state["progress"] == 30
    assert state["phase"] == "tagging"
    assert "Tagging" in state["message"]


def test_parse_progress_line_bracket_fallback():
    state = {"progress": 0}
    assert parse_progress_line("  [5/20] 25.0% - Tagging image", state) is True
    assert state["progress"] == 25
