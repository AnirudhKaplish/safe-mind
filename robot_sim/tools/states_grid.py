"""Write runs/robot/states_grid.png: every state from two camera presets."""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from robot_sim import config
from robot_sim.runtime import TwinRuntime
from robot_sim.sim import CAMERA_PRESETS
from robot_sim.states import SPECS, STATE_ORDER

VIEWS = ("front", "three_quarter")
W, H = 400, 300


def main(settle_s: float = 2.0) -> None:
    rt = TwinRuntime(live=False, manual=True)
    cams = {c.name: c for c in CAMERA_PRESETS}
    try:
        lf = ImageFont.load_default(size=15)
    except TypeError:
        lf = ImageFont.load_default()
    label_h = 26
    sheet = Image.new("RGB", (len(STATE_ORDER) * W, len(VIEWS) * (H + label_h)), config.PALETTE["paper"])
    d = ImageDraw.Draw(sheet)
    for i, st in enumerate(STATE_ORDER):
        rt.manual_key(i)
        for _ in range(int(settle_s / 0.02)):
            rt.step(0.02)
        for r, v in enumerate(VIEWS):
            c = cams[v]
            img = rt.sim.render(c.target, c.distance, c.yaw, c.pitch, 640, 480, True)
            tile = Image.fromarray(img).resize((W, H), Image.LANCZOS)
            y = r * (H + label_h)
            sheet.paste(tile, (i * W, y + label_h))
            d.text((i * W + 8, y + 5), f"{i + 1} {SPECS[st].label} · {v}", fill=config.PALETTE["ink"], font=lf)
    out = config.RUNS_DIR / "states_grid.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    # wrap 9 columns into 3x3 blocks of (front over three-quarter) for readability
    cols = 3
    blocks = []
    for i in range(len(STATE_ORDER)):
        blocks.append(sheet.crop((i * W, 0, (i + 1) * W, sheet.height)))
    rows = (len(blocks) + cols - 1) // cols
    bh = sheet.height
    final = Image.new("RGB", (cols * W, rows * bh), config.PALETTE["paper"])
    for k, b in enumerate(blocks):
        final.paste(b, ((k % cols) * W, (k // cols) * bh))
    final.save(out)
    rt.close()
    print(out)


if __name__ == "__main__":
    main()
