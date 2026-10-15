"""SSD1306 128x64 1-bit framebuffer: eyes, icons, two text lines.

There is deliberately no mouth: the face shows system state, not emotion.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import config

W, H = config.OLED_W, config.OLED_H

# Blink: open -> half -> closed -> half -> open over 150 ms (4 transitions).
BLINK_FRAMES = (1.0, 0.5, 0.0, 0.5, 1.0)

_FONT = None


def font() -> ImageFont.ImageFont:
    """Pillow's built-in bitmap font: 6 px per character, glyphs 9 rows tall."""
    global _FONT
    if _FONT is None:
        try:
            _FONT = ImageFont.load_default_imagefont()
        except AttributeError:  # Pillow < 10.1
            _FONT = ImageFont.load_default()
    return _FONT


def _bitmap(rows: list[str]) -> np.ndarray:
    return np.array([[c == "#" for c in r] for r in rows], dtype=bool)


ICONS = {
    "lock": _bitmap([
        "....######....",
        "...##....##...",
        "..##......##..",
        "..##......##..",
        "..##......##..",
        "..##......##..",
        "##############",
        "##############",
        "######..######",
        "#####....#####",
        "#####....#####",
        "######..######",
        "######..######",
        "##############",
        "##############",
    ]),
    "question": _bitmap([
        ".####.",
        "##..##",
        "....##",
        "...##.",
        "..##..",
        "..##..",
        "......",
        "..##..",
    ]),
    "bang": _bitmap([
        "##", "##", "##", "##", "##", "##", "##", "##", "..", "##",
    ]),
    "check": _bitmap([
        "..........##",
        ".........##.",
        "........##..",
        "##.....##...",
        ".##...##....",
        "..##.##.....",
        "...###......",
        "....#.......",
    ]),
}


@dataclass(frozen=True)
class OledParams:
    eyes: str = "neutral"          # neutral | attentive | sleepy | none
    openness: float = 1.0          # 1 open .. 0 closed (blink)
    gaze_px: int = 0
    icon: Optional[str] = None     # lock | question | bang | check
    lines: tuple[str, ...] = ()
    dim: bool = False


class TextOverflow(ValueError):
    pass


def check_lines(lines: tuple[str, ...]) -> None:
    if len(lines) > 2:
        raise TextOverflow(f"at most two OLED lines, got {len(lines)}")
    for ln in lines:
        if len(ln) > config.OLED_MAX_CHARS:
            raise TextOverflow(f"OLED line too long ({len(ln)} > {config.OLED_MAX_CHARS}): {ln!r}")
        if not ln.isascii():
            raise TextOverflow(f"OLED font is ASCII only: {ln!r}")
        bbox = font().getbbox(ln) if ln else (0, 0, 0, 0)
        if bbox[2] > W:
            raise TextOverflow(f"OLED line wider than {W}px: {ln!r}")


def _blit(fb: np.ndarray, bm: np.ndarray, cx: int, cy: int, scale: int = 1) -> None:
    if scale > 1:
        bm = np.kron(bm, np.ones((scale, scale), dtype=bool))
    h, w = bm.shape
    x0, y0 = cx - w // 2, cy - h // 2
    fb[y0:y0 + h, x0:x0 + w] |= bm


def render(p: OledParams) -> np.ndarray:
    """Return a (64, 128) bool framebuffer. Raises TextOverflow on bad text."""
    check_lines(p.lines)
    img = Image.new("1", (W, H), 0)
    d = ImageDraw.Draw(img)

    if p.eyes != "none":
        ew, eh = config.OLED_EYE_W, config.OLED_EYE_H
        if p.eyes == "attentive":
            ew, eh = ew + 2, eh + 4
        openness = max(0.0, min(1.0, p.openness))
        if p.eyes == "sleepy":
            openness = min(openness, 0.3)
        h = max(2, int(round(eh * openness)))
        gaze = max(-config.OLED_GAZE_MAX_PX, min(config.OLED_GAZE_MAX_PX, int(p.gaze_px)))
        for cx, cy in config.OLED_EYE_CENTERS:
            x0, x1 = cx + gaze - ew // 2, cx + gaze + ew // 2 - 1
            y0, y1 = cy - h // 2, cy - h // 2 + h - 1
            d.rounded_rectangle((x0, y0, x1, y1), radius=min(6, max(1, h // 2 - 1)), fill=1)

    for i, ln in enumerate(p.lines):
        w = font().getbbox(ln)[2] if ln else 0
        d.text(((W - w) // 2, config.OLED_TEXT_Y[i]), ln, font=font(), fill=1)

    fb = np.array(img, dtype=bool)
    if p.icon:
        # centred in the eye area; "?" sits in the gap between the eyes
        _blit(fb, ICONS[p.icon], 64, 20, scale=2)
    return fb


def to_rgb(fb: np.ndarray, scale: int = 1, dim: bool = False) -> np.ndarray:
    on = np.array(config.OLED_ON_RGB, dtype=np.float32)
    off = np.array(config.OLED_OFF_RGB, dtype=np.float32)
    if dim:
        on = off + (on - off) * config.OLED_DIM_FACTOR
    img = np.where(fb[..., None], on, off).astype(np.uint8)
    if scale > 1:
        img = img.repeat(scale, axis=0).repeat(scale, axis=1)
    return img


def frame_hash(fb: np.ndarray, dim: bool = False) -> str:
    h = hashlib.sha1(np.packbits(fb).tobytes())
    h.update(b"d" if dim else b"b")
    return h.hexdigest()[:16]


class TextureCache:
    """Writes OLED textures as PNGs, only when the frame content changes."""

    def __init__(self, directory: Path = config.OLED_CACHE_DIR) -> None:
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.writes = 0
        self.current: Optional[str] = None

    def update(self, fb: np.ndarray, dim: bool = False) -> tuple[str, Path, bool]:
        """Return (hash, png_path, changed). ``changed`` is False for repeats."""
        h = frame_hash(fb, dim)
        path = self.dir / f"{h}.png"
        changed = h != self.current
        if changed and not path.exists():
            tw, th = config.OLED_TEXTURE_SIZE
            img = to_rgb(fb, scale=tw // W, dim=dim)
            Image.fromarray(img).save(path)
            self.writes += 1
        self.current = h
        return h, path, changed


def contact_sheet(frames: list[tuple[str, np.ndarray, bool]], cols: int = 3, scale: int = 3) -> Image.Image:
    """Grid of labelled OLED frames for documentation."""
    pad, label_h = 16, 22
    cw, ch = W * scale, H * scale
    rows = (len(frames) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * (cw + pad) + pad, rows * (ch + pad + label_h) + pad), (246, 245, 241))
    d = ImageDraw.Draw(sheet)
    try:
        lf = ImageFont.load_default(size=14)
    except TypeError:
        lf = ImageFont.load_default()
    for i, (name, fb, dim) in enumerate(frames):
        r, c = divmod(i, cols)
        x = pad + c * (cw + pad)
        y = pad + r * (ch + pad + label_h)
        d.text((x, y), f"{i + 1}  {name}", fill=(28, 42, 46), font=lf)
        sheet.paste(Image.fromarray(to_rgb(fb, scale, dim)), (x, y + label_h))
    return sheet
