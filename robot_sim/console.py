"""OpenCV console: 3D view + telemetry cards, keys, mouse orbit.

Layout (1280x720, window resizable):
  banner (48 px, full width) - LIVE or striped MANUAL banner, every frame
  left  (58 %): 3D view with camera preset name + legend, "Last report" card
  right (42 %): State, Virtual OLED (exact 4x), Servos (+10 s trace),
                LED + Buzzer, Link
  footer: key hints

The "Last report" card sits under the 3D view (not in the right column)
because the 4x OLED (512 px wide) needs a column of at least 520 px.
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import config
from .fsm import Mode
from .oled import to_rgb
from .runtime import TwinRuntime
from .sim import CameraRig
from .states import SPECS, RobotState

P = config.PALETTE
W, H = config.CONSOLE_W, config.CONSOLE_H
BANNER = config.BANNER_H
FOOTER = 26
LEFT_W = 740
VIEW_H = 520
GAP = 8
RX = LEFT_W + GAP          # right column x
RW = W - RX - GAP          # right column width (= 524)

_FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
    "/System/Library/Fonts/SFNS.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "C:/Windows/Fonts/segoeui.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
]
_BOLD_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "C:/Windows/Fonts/segoeuib.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]
_SUBST = {"●": "*", "—": "-", "°": " deg", "·": "|", "✓": "v", "µ": "u", "↔": "<>", "×": "x", "±": "+/-"}


def _glyph(font, ch: str) -> bytes:
    im = Image.new("L", (40, 40), 0)
    ImageDraw.Draw(im).text((4, 4), ch, font=font, fill=255)
    return im.tobytes()


class Fonts:
    def __init__(self) -> None:
        self.path = next((p for p in _FONT_CANDIDATES if Path(p).exists()), None)
        self.bold_path = next((p for p in _BOLD_CANDIDATES if Path(p).exists()), self.path)
        self._cache: dict = {}
        self._missing: set[str] = set()
        if self.path:
            probe = self.get(14)
            tofu = _glyph(probe, "\ue000")
            for ch in _SUBST:
                try:
                    if _glyph(probe, ch) == tofu:
                        self._missing.add(ch)
                except Exception:  # pragma: no cover
                    self._missing.add(ch)
        else:
            self._missing = set(_SUBST)

    def get(self, size: int, bold: bool = False):
        key = (size, bold)
        if key not in self._cache:
            path = self.bold_path if bold else self.path
            try:
                self._cache[key] = ImageFont.truetype(path, size) if path else ImageFont.load_default(size=size)
            except (OSError, TypeError):
                self._cache[key] = ImageFont.load_default()
        return self._cache[key]

    def fix(self, text: str) -> str:
        for ch in self._missing:
            text = text.replace(ch, _SUBST[ch])
        return text


class Console:
    def __init__(self, rt: TwinRuntime, *, settings=None, use_worker: bool = False,
                 render_3d: bool = True) -> None:
        self.rt = rt
        self.fonts = Fonts()
        self.rig = CameraRig()
        self.settings = settings
        self.render_3d = render_3d and rt.sim is not None
        self.worker = None
        if self.render_3d and use_worker:
            from .sim import RenderWorker
            self.worker = RenderWorker()
        self.view: Optional[np.ndarray] = None
        self.view_time = 0.0
        self.hint: Optional[tuple[str, float]] = None
        self.texts: list[str] = []
        self.fps = 0.0
        self.view_fps = 0.0
        self._fps_t = time.monotonic()
        self._fps_n = 0
        self._view_n = 0
        self._drag: Optional[tuple[int, int]] = None
        self.view_rect = (0, BANNER, LEFT_W, VIEW_H)

    # -- input --------------------------------------------------------------
    def flash_hint(self, text: str, seconds: float = 1.5) -> None:
        self.hint = (text, self.rt.now() + seconds)

    def on_mouse(self, event, x, y, flags, param=None) -> None:  # pragma: no cover - GUI
        import cv2
        vx, vy, vw, vh = self.view_rect
        if event == cv2.EVENT_LBUTTONDOWN and vx <= x < vx + vw and vy <= y < vy + vh:
            self._drag = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self._drag is not None:
            dx, dy = x - self._drag[0], y - self._drag[1]
            self.rig.orbit += np.array([-dx * 0.4, -dy * 0.3])
            self._drag = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self._drag = None

    # -- 3D view ------------------------------------------------------------
    def update_view(self) -> None:
        if not self.render_3d:
            return
        rt = self.rt
        cam = self.rig.update(rt.now())
        st = self.settings
        if self.worker is not None and not self.worker.alive:
            print("[robot] render worker stopped; rendering in-process instead", file=sys.stderr)
            self.worker.close()
            self.worker = None
        if self.worker is not None:
            pan, _ = rt.actual("pan")
            tilt, _ = rt.actual("tilt")
            img = self.worker.poll()
            if img is not None and img is not self.view:
                self.view = img
                self._view_n += 1
            self.worker.submit(pan, tilt, rt.led, (rt.oled_hash, rt.oled_png), cam, st)
        else:
            self.view = rt.sim.render(*cam, st.w, st.h, st.shadows)
            self._view_n += 1

    # -- drawing helpers ----------------------------------------------------
    def _t(self, d: ImageDraw.ImageDraw, xy, text: str, size=14, fill=P["ink"], bold=False, anchor="la"):
        text = self.fonts.fix(text)
        self.texts.append(text)
        d.text(xy, text, font=self.fonts.get(size, bold), fill=fill, anchor=anchor)

    def _card(self, d, x, y, w, h, title: Optional[str] = None, muted=False):
        d.rounded_rectangle((x, y, x + w, y + h), radius=8, fill=P["card"], outline=P["line"])
        if title:
            self._t(d, (x + 12, y + 8), title.upper(), 11, P["muted"], bold=True)

    # -- compose ------------------------------------------------------------
    def compose(self) -> np.ndarray:
        """Return the full console as an RGB array (H, W, 3)."""
        rt, fsm = self.rt, self.rt.fsm
        now = rt.now()
        self.texts = []
        img = Image.new("RGB", (W, H), P["paper"])
        d = ImageDraw.Draw(img)
        self._banner(img, d, now)
        self._left(img, d, now)
        y = BANNER + GAP
        y = self._state_card(d, y, now) + GAP
        y = self._oled_card(img, d, y) + GAP
        y = self._servo_card(d, y) + GAP
        self._led_link_cards(d, y, now)
        self._footer(d, now)
        self._tick_fps()
        return np.asarray(img)

    def _tick_fps(self) -> None:
        self._fps_n += 1
        t = time.monotonic()
        if t - self._fps_t >= 1.0:
            self.fps = self._fps_n / (t - self._fps_t)
            self.view_fps = self._view_n / (t - self._fps_t)
            self._fps_t, self._fps_n, self._view_n = t, 0, 0

    def _banner(self, img, d, now) -> None:
        fsm = self.rt.fsm
        if fsm.mode is Mode.MANUAL:
            d.rectangle((0, 0, W, BANNER), fill=P["amber"])
            stripe = tuple(int(c * 0.86) for c in P["amber"])
            for x in range(-BANNER, W, 28):
                d.polygon([(x, BANNER), (x + 14, BANNER), (x + 14 + BANNER, 0), (x + BANNER, 0)], fill=stripe)
            d.rectangle((14, 9, 560, BANNER - 9), fill=P["ink"])
            self._t(d, (26, BANNER // 2), "MANUAL DEMO — NOT A MODEL PREDICTION", 17, (255, 214, 120), True, "lm")
            if fsm.live_waiting:
                d.rectangle((W - 230, 11, W - 14, BANNER - 11), fill=P["ink"])
                self._t(d, (W - 122, BANNER // 2), "live update waiting", 13, (255, 255, 255), anchor="mm")
        else:
            d.rectangle((0, 0, W, BANNER), fill=P["ink"])
            rid = (fsm.report or {}).get("report_id")
            if fsm.state is RobotState.OFFLINE:
                msg = "● LIVE AI — dashboard unreachable"
            elif fsm.link_ok is None:
                msg = "● LIVE AI — connecting to dashboard"
            elif rid is not None:
                msg = f"● LIVE AI — dashboard report #{rid}"
            else:
                msg = "● LIVE AI — dashboard"
            col = P["live"] if fsm.link_ok else P["blocked"]
            self._t(d, (20, BANNER // 2), msg, 17, col, True, "lm")
            lat = fsm.last_latency
            right = f"link {lat * 1000:.0f} ms" if (lat is not None and fsm.link_ok) else "link —"
            self._t(d, (W - 20, BANNER // 2), right, 13, (220, 226, 224), anchor="rm")
        flash = fsm.current_flash()
        hint = self.hint[0] if (self.hint and now < self.hint[1]) else None
        msg = hint or flash
        if msg:
            tw = int(d.textlength(self.fonts.fix(msg), font=self.fonts.get(14))) + 28
            x0 = (LEFT_W - tw) // 2
            d.rounded_rectangle((x0, BANNER + 10, x0 + tw, BANNER + 38), radius=14, fill=P["ink"])
            self._t(d, (x0 + tw // 2, BANNER + 24), msg, 14, (255, 255, 255), anchor="mm")

    def _left(self, img, d, now) -> None:
        vx, vy = 0, BANNER
        vw, vh = LEFT_W, VIEW_H              # 740 x 520 (4:3 render, centre-cropped)
        self.view_rect = (vx, vy, vw, vh)
        if self.view is not None:
            full_h = int(vw * self.view.shape[0] / self.view.shape[1])
            v = Image.fromarray(self.view).resize((vw, full_h), Image.BILINEAR)
            top = max(0, (full_h - vh) // 2)
            img.paste(v.crop((0, top, vw, top + vh)), (vx, vy))
        else:
            d.rectangle((vx, vy, vx + vw, vy + vh), fill=(232, 230, 224))
            self._t(d, (vw // 2, vy + vh // 2), "3D view off (--no-3d)" if not self.render_3d else "rendering…",
                    15, P["muted"], anchor="mm")
        if self.rt.fsm.mode is Mode.MANUAL:
            # keep the label inside the picture too, so crops never lose it
            d.rectangle((vx, vy + vh - 26, vx + vw, vy + vh), fill=P["amber"])
            self._t(d, (vx + vw // 2, vy + vh - 13), "MANUAL DEMO — NOT A MODEL PREDICTION", 12,
                    (255, 255, 255), True, "mm")
        d.rounded_rectangle((vx + 10, vy + 50, vx + 190, vy + 74), radius=6, fill=(255, 255, 255))
        self._t(d, (vx + 18, vy + 62), f"camera: {self.rig.name}", 12, P["ink"], anchor="lm")
        legend = "Digital twin of ESP32 device · 2× SG90 · SSD1306 128×64"
        ly = vy + vh - (54 if self.rt.fsm.mode is Mode.MANUAL else 28)
        lw = int(d.textlength(self.fonts.fix(legend), font=self.fonts.get(12))) + 16
        d.rounded_rectangle((vx + 10, ly - 4, vx + 10 + lw, ly + 18), radius=6, fill=(255, 255, 255))
        self._t(d, (vx + 18, ly), legend, 12, P["ink"])
        if self.settings is not None and self.render_3d:
            rate = "fixed-step render" if self.rt.clock == self.rt._virtual_clock else f"{self.view_fps:.0f} fps"
            self._t(d, (vw - 12, vy + 62), f"{self.settings.label} · {rate}"
                    f"{' · worker' if self.worker else ''}", 11, P["muted"], anchor="rm")
        self._report_card(d, vx + GAP, vy + vh + GAP, vw - 2 * GAP, H - FOOTER - (vy + vh + GAP) - GAP)

    def _report_card(self, d, x, y, w, h) -> None:
        fsm = self.rt.fsm
        dismissed = fsm.report_dismissed
        self._card(d, x, y, w, h, "Last report")
        rep = fsm.report if fsm.mode is Mode.LIVE else None
        ink = P["blocked"] if dismissed else P["ink"]
        if fsm.mode is Mode.MANUAL:
            self._t(d, (x + 12, y + 30), "Manual demo — no report is shown in this mode.", 14, P["muted"])
        elif not rep:
            self._t(d, (x + 12, y + 30), "No report yet.", 14, P["muted"])
        else:
            agree = {True: "yes", False: "no", None: "—"}[rep.get("models_agree")]
            conf = rep.get("confidence")
            conf_s = f"{conf:.2f}" if (conf is not None and rep.get("status") == "AGREEMENT") else "—"
            cols = [("status", str(rep.get("status"))), ("models agree", agree), ("mean confidence", conf_s),
                    ("consent", str(rep.get("consent"))), ("received", fsm.report_received_at or "—")]
            cx = x + 12
            for label, val in cols:
                self._t(d, (cx, y + 30), label, 11, P["muted"])
                self._t(d, (cx, y + 46), val, 16, ink, True)
                cx += max(110, int(d.textlength(val, font=self.fonts.get(15, True))) + 26)
            if dismissed:
                self._t(d, (x + w - 12, y + 8), "dismissed on device", 11, P["muted"], anchor="ra")
        self._t(d, (x + 12, y + h - 22),
                "Screening support only — not a diagnosis. Full report and explanations in the dashboard.",
                11, P["muted"])

    def _state_card(self, d, y, now) -> int:
        fsm = self.rt.fsm
        h = 56
        self._card(d, RX, y, RW, h, "State")
        spec = SPECS[fsm.state]
        col = {RobotState.SCREEN_FLAGGED: P["amber"], RobotState.REVIEW_NEEDED: P["amber"],
               RobotState.SCREEN_CLEAR: P["low"], RobotState.CONSENT_BLOCKED: P["blocked"],
               RobotState.ERROR: P["blocked"], RobotState.OFFLINE: P["blocked"]}.get(fsm.state, P["accent"])
        d.rounded_rectangle((RX + 12, y + 26, RX + 18, y + 48), radius=3, fill=col)
        self._t(d, (RX + 26, y + 37), spec.label, 21, P["ink"], True, "lm")
        src = {"live report": "live report", "manual": "manual", "link": "link", "button": "button",
               "startup": "startup"}.get(fsm.source, fsm.source)
        info = f"{fsm.time_in_state():5.1f} s · source: {src}"
        self._t(d, (RX + RW - 12, y + 9), info, 12, P["muted"], anchor="ra")
        if config.BORDERLINE_REVIEW:
            self._t(d, (RX + RW - 12, y + 30), "borderline rule active", 11, P["amber"], True, anchor="ra")
        return y + h

    def _oled_card(self, img, d, y) -> int:
        rt = self.rt
        h = 256 + 44
        self._card(d, RX, y, RW, h, "Virtual OLED · SSD1306 128×64 · 4×")
        bx = RX + (RW - 512) // 2
        by = y + 30
        d.rounded_rectangle((bx - 4, by - 4, bx + 516, by + 260), radius=6, fill=(18, 20, 22))
        if rt.oled_fb is not None:
            o = to_rgb(rt.oled_fb, config.OLED_CONSOLE_SCALE, rt.oled_params.dim)
            img.paste(Image.fromarray(o), (bx, by))
            self.texts.extend(rt.oled_params.lines)
        return y + h

    def _servo_card(self, d, y) -> int:
        rt = self.rt
        h = 150
        self._card(d, RX, y, RW, h, "Servos · SG90")
        tp, tt = rt.telemetry()
        heads = ["", "cmd°", "actual°", "err°", "PWM µs", "°/s"]
        xs = [RX + 12, RX + 70, RX + 140, RX + 220, RX + 290, RX + 370]
        for x, hd in zip(xs, heads):
            self._t(d, (x, y + 26), hd, 11, P["muted"])
        for row, (name, t) in enumerate((("Pan", tp), ("Tilt", tt))):
            yy = y + 42 + row * 18
            vals = [name, f"{t.cmd:6.1f}", f"{t.actual:6.1f}", f"{t.err:+5.2f}", f"{t.pulse_us:5.0f}", f"{t.vel:+6.1f}"]
            for x, v in zip(xs, vals):
                self._t(d, (x, yy), v, 13, P["ink"], bold=(x == xs[0]))
        # trace
        gx, gy, gw, gh = RX + 12, y + 80, RW - 24, h - 88
        d.rectangle((gx, gy, gx + gw, gy + gh), fill=(250, 249, 246), outline=P["line"])
        lo, hi = 30.0, 150.0
        for ref in (60, 90, 120):
            yy = gy + gh - (ref - lo) / (hi - lo) * gh
            d.line((gx, yy, gx + gw, yy), fill=(236, 233, 226))
            self._t(d, (gx + gw - 4, yy - 1), f"{ref}°", 9, P["muted"], anchor="rb")
        tr = list(rt.trace)
        if len(tr) > 2:
            t_end = tr[-1][0]

            def pt(t, v):
                return (gx + gw - (t_end - t) / config.TRACE_SECONDS * gw,
                        gy + gh - (min(max(v, lo), hi) - lo) / (hi - lo) * gh)
            step = max(1, len(tr) // 250)
            tr = tr[::step]
            for ci, ai, col in ((1, 2, P["accent"]), (3, 4, P["amber"])):
                act = [pt(r[0], r[ai]) for r in tr]
                d.line(act, fill=col, width=2)
                cmd = [pt(r[0], r[ci]) for r in tr]
                for k in range(0, len(cmd) - 1, 2):     # dashed commanded line
                    d.line((cmd[k], cmd[k + 1]), fill=(28, 42, 46), width=1)
        self._t(d, (gx + 6, gy + 4), "pan", 10, P["accent"], True)
        self._t(d, (gx + 34, gy + 4), "tilt", 10, P["amber"], True)
        self._t(d, (gx + 62, gy + 4), "dashed = commanded · solid = actual · 10 s", 10, P["muted"])
        return y + h

    def _led_link_cards(self, d, y, now) -> None:
        rt, fsm = self.rt, self.rt.fsm
        h = H - FOOTER - GAP - y
        half = (RW - GAP) // 2
        # LED + buzzer
        self._card(d, RX, y, half, h, "LED + Buzzer")
        spec = SPECS[fsm.state]
        d.ellipse((RX + 12, y + 28, RX + 36, y + 52), fill=tuple(rt.led) if any(rt.led) else (40, 40, 42),
                  outline=P["line"])
        pat = spec.led_pattern + (f" {spec.led_hz:g} Hz" if spec.led_pattern in ("pulse", "blink") else "")
        self._t(d, (RX + 44, y + 30), pat, 13, P["ink"], True)
        self._t(d, (RX + 44, y + 47), f"RGB {spec.led_rgb[0]},{spec.led_rgb[1]},{spec.led_rgb[2]}", 11, P["muted"])
        tile = rt.buzzer.tile_text(now)
        self._t(d, (RX + 12, y + h - 40), "BUZZER", 10, P["muted"], bold=True)
        self._t(d, (RX + 12, y + h - 24), tile or "—", 11, P["ink"] if tile else P["muted"], bold=bool(tile))
        # link
        lx = RX + half + GAP
        self._card(d, lx, y, RW - half - GAP, h, "Link")
        host = rt.url.split("//", 1)[-1].split("/", 1)[0]
        status = {True: "OK", False: "OFFLINE", None: "—"}[fsm.link_ok]
        if not rt.polling_enabled:
            status = "not polling" + (" (manual)" if fsm.mode is Mode.MANUAL else "")
        ago = f"{rt.now() - fsm.last_success:.1f} s ago" if fsm.last_success is not None else "never"
        boot = fsm.gate.boot_id[:8] if fsm.gate.boot_id else "—"
        self._t(d, (lx + 12, y + 28), f"{host} · {status}", 13,
                P["low"] if fsm.link_ok else P["blocked"], True)
        self._t(d, (lx + 12, y + 46), f"last poll {ago} · seq {fsm.gate.seq if fsm.gate.seq >= 0 else '—'}",
                11, P["muted"])
        self._t(d, (lx + 12, y + 61), f"boot {boot} · out-of-order {fsm.gate.out_of_order}", 11, P["muted"])

    def _footer(self, d, now) -> None:
        d.rectangle((0, H - FOOTER, W, H), fill=(234, 232, 225))
        keys = "m mode · 1–9 states (manual) · 0 auto-cycle · c confirm · x dismiss · v/b camera · r reset view · s screenshot · space pause · q quit"
        self._t(d, (12, H - FOOTER // 2), keys, 12, P["ink"], anchor="lm")
        right = "fixed-step (headless)" if self.rt.clock == self.rt._virtual_clock else f"{self.fps:4.1f} fps"
        if self.rt.paused:
            right = "PAUSED · " + right
        self._t(d, (W - 12, H - FOOTER // 2), right, 12, P["muted"], anchor="rm")

    def close(self) -> None:
        if self.worker is not None:
            self.worker.close()
            self.worker = None


def bgr(rgb: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(rgb[:, :, ::-1])
