"""Process UI deliverables from doc/local into committed assets."""
from __future__ import annotations

from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "doc" / "local" / "Next Trainer" / "Next Trainer"
ICON_SRC = SRC / "icon_4.png"
BANNER_SRC = SRC / "banner.png"


def cover_resize(img: Image.Image, size: tuple[int, int], bg: tuple[int, int, int] = (255, 255, 255)) -> Image.Image:
    tw, th = size
    src = img.convert("RGBA")
    sw, sh = src.size
    scale = max(tw / sw, th / sh)
    nw, nh = int(sw * scale), int(sh * scale)
    resized = src.resize((nw, nh), Image.Resampling.LANCZOS)
    x0 = (nw - tw) // 2
    y0 = (nh - th) // 2
    cropped = resized.crop((x0, y0, x0 + tw, y0 + th))
    out = Image.new("RGB", size, bg)
    out.paste(cropped, mask=cropped.split()[3])
    return out


def square_icon(img: Image.Image, size: int) -> Image.Image:
    src = img.convert("RGBA")
    sw, sh = src.size
    side = min(sw, sh)
    x0 = (sw - side) // 2
    y0 = (sh - side) // 2
    cropped = src.crop((x0, y0, x0 + side, y0 + side))
    return cropped.resize((size, size), Image.Resampling.LANCZOS)


def main() -> None:
    icon = Image.open(ICON_SRC)
    banner = Image.open(BANNER_SRC)

    readme_dir = ROOT / "assets" / "readme"
    readme_dir.mkdir(parents=True, exist_ok=True)

    cover = cover_resize(banner, (1280, 720))
    cover.save(readme_dir / "next-trainer-cover.png", optimize=True)
    cover.save(readme_dir / "anima-cover.png", optimize=True)

    social = cover_resize(banner, (1280, 640))
    social.save(readme_dir / "next-trainer-social.png", optimize=True)

    logo = square_icon(icon, 256)
    logo.save(ROOT / "assets" / "logo.png", optimize=True)

    webp_src = square_icon(icon, 512)
    webp_src.save(ROOT / "frontend" / "dist" / "assets" / "icon.65fd68ba.webp", format="WEBP", quality=90, method=6)

    ico_master = square_icon(icon, 256)
    ico_master.save(
        ROOT / "assets" / "favicon.ico",
        format="ICO",
        sizes=[(256, 256), (128, 128), (64, 64), (48, 48), (32, 32), (16, 16)],
    )

    print("Wrote:")
    for p in [
        readme_dir / "next-trainer-cover.png",
        readme_dir / "anima-cover.png",
        readme_dir / "next-trainer-social.png",
        ROOT / "assets" / "logo.png",
        ROOT / "frontend" / "dist" / "assets" / "icon.65fd68ba.webp",
        ROOT / "assets" / "favicon.ico",
    ]:
        print(f"  {p.relative_to(ROOT)} ({p.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
