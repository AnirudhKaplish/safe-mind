"""OpenCV console: 3D robot view + plain-language result cards, keys, mouse orbit.

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
_SUBST = {"●": "*", "—": "-", "°": " deg", "·": "|", "✓": "v", "µ": "u", "↔": "<>", "×": "x", "±": "+/-",
          "…": "..."}

RED = (192, 57, 43)
GREEN = (46, 125, 82)

# Plain-language text for each state: (headline, explanation, colour key, light)
FRIENDLY = {
    RobotState.IDLE: ("Ready", "Type a post in the dashboard to start a check.", "accent", "White"),
    RobotState.PROCESSING: ("Reading the post…", "Both AI models are checking the text.", "accent", "Blue (pulsing)"),
    RobotState.NOT_APPLICABLE: ("Not checked", "Too short, or not a personal post. Please write a bit more.",
                                "blocked", "Dim white"),
    RobotState.CONSENT_BLOCKED: ("No permission", "Consent was withdrawn, so nothing was analysed.",
                                 "blocked", "Grey"),
    RobotState.SCREEN_CLEAR: ("You seem okay", "No signs of depression were found in this post.", "green", "Green"),
    RobotState.SCREEN_FLAGGED: ("Signs of depression found", "Please meet a therapist.", "red", "Red"),
    RobotState.REVIEW_NEEDED: ("Not sure", "The two AI models disagree. A therapist should take a look.",
                               "amber", "Amber (pulsing)"),
    RobotState.ERROR: ("Something went wrong", "No result is available. Please try again.", "blocked",
                       "Grey (blinking)"),
    RobotState.OFFLINE: ("Not connected", "Start the dashboard so the robot can receive results.", "blocked", "Off"),
}


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
                 render_3d: bool = True, soft_3d: bool = False) -> None:
        self.rt = rt
        self.fonts = Fonts()
        self.rig = CameraRig()
        self.settings = settings
        self.render_3d = render_3d and rt.sim is not None
        self.soft = None
        if soft_3d and not self.render_3d:
            from .soft3d import SoftRenderer
            self.soft = SoftRenderer()
        self.details = False
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
        rt = self.rt
        if self.soft is not None:
            target, dist, yaw, pitch = self.rig.update(rt.now())
            pan, _ = rt.actual("pan")
            tilt, _ = rt.actual("tilt")
            w, h = (self.settings.w, self.settings.h) if self.settings is not None else (640, 480)
            self.view = self.soft.render(pan, tilt, rt.led, rt.oled_fb,
                                         rt.oled_params.dim if rt.oled_params else False,
                                         target, dist, yaw, pitch, w, h)
            self._view_n += 1
            return
        if not self.render_3d:
            return
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
        if self.details:
            self._details_card(d, y, now)
        else:
            self._status_card(d, y, now)
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

    def _colour(self, key: str):
        return {"red": RED, "green": GREEN, "amber": P["amber"], "blocked": P["blocked"]}.get(key, P["accent"])

    def _banner(self, img, d, now) -> None:
        fsm = self.rt.fsm
        if fsm.mode is Mode.MANUAL:
            d.rectangle((0, 0, W, BANNER), fill=P["amber"])
            stripe = tuple(int(c * 0.86) for c in P["amber"])
            for x in range(-BANNER, W, 28):
                d.polygon([(x, BANNER), (x + 14, BANNER), (x + 14 + BANNER, 0), (x + BANNER, 0)], fill=stripe)
            d.rectangle((14, 9, 600, BANNER - 9), fill=P["ink"])
            self._t(d, (26, BANNER // 2), "DEMO MODE — sample screens, not a real result", 17,
                    (255, 214, 120), True, "lm")
            if fsm.live_waiting:
                d.rectangle((W - 260, 11, W - 14, BANNER - 11), fill=P["ink"])
                self._t(d, (W - 137, BANNER // 2), "new result waiting (press m)", 13, (255, 255, 255),
                        anchor="mm")
        else:
            d.rectangle((0, 0, W, BANNER), fill=P["ink"])
            rid = (fsm.report or {}).get("report_id")
            if fsm.state is RobotState.OFFLINE:
                msg = "● Dashboard not connected"
            elif fsm.link_ok is None:
                msg = "● Connecting to the dashboard…"
            else:
                msg = "● Live — connected to the Meridian dashboard"
            col = P["live"] if fsm.link_ok else P["blocked"]
            self._t(d, (20, BANNER // 2), msg, 17, col, True, "lm")
            if rid is not None:
                self._t(d, (W - 20, BANNER // 2), f"Report #{rid}", 14, (220, 226, 224), True, anchor="rm")
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
            off = not self.render_3d and self.soft is None
            self._t(d, (vw // 2, vy + vh // 2), "3D view off (--no-3d)" if off else "rendering…",
                    15, P["muted"], anchor="mm")
        if self.rt.fsm.mode is Mode.MANUAL:
            d.rectangle((vx, vy + vh - 26, vx + vw, vy + vh), fill=P["amber"])
            self._t(d, (vx + vw // 2, vy + vh - 13), "DEMO MODE — NOT A REAL RESULT", 12,
                    (255, 255, 255), True, "mm")
        legend = "Your companion robot (virtual copy of the ESP32 device) · drag to rotate"
        ly = vy + vh - (54 if self.rt.fsm.mode is Mode.MANUAL else 28)
        lw = int(d.textlength(self.fonts.fix(legend), font=self.fonts.get(12))) + 16
        d.rounded_rectangle((vx + 10, ly - 4, vx + 10 + lw, ly + 18), radius=6, fill=(255, 255, 255))
        self._t(d, (vx + 18, ly), legend, 12, P["ink"])
        if self.details:
            d.rounded_rectangle((vx + 10, vy + 50, vx + 190, vy + 74), radius=6, fill=(255, 255, 255))
            self._t(d, (vx + 18, vy + 62), f"camera: {self.rig.name}", 12, P["ink"], anchor="lm")
            if self.settings is not None and (self.render_3d or self.soft is not None):
                kind = "lightweight 3D" if self.soft is not None else self.settings.label
                self._t(d, (vw - 12, vy + 62), f"{kind} · {self.view_fps:.0f} fps", 11, P["muted"], anchor="rm")
        self._report_card(d, vx + GAP, vy + vh + GAP, vw - 2 * GAP, H - FOOTER - (vy + vh + GAP) - GAP)

    def _report_card(self, d, x, y, w, h) -> None:
        fsm = self.rt.fsm
        dismissed = fsm.report_dismissed
        self._card(d, x, y, w, h, "Latest result")
        rep = fsm.report if fsm.mode is Mode.LIVE else None
        ink = P["blocked"] if dismissed else P["ink"]
        if fsm.mode is Mode.MANUAL:
            self._t(d, (x + 12, y + 32), "Demo mode — press 1–9 to preview each screen, or m to go live.", 14,
                    P["muted"])
        elif not rep:
            self._t(d, (x + 12, y + 32), "No result yet. Submit a post in the dashboard.", 14, P["muted"])
        else:
            st = rep.get("status")
            agree = {True: "Yes", False: "No", None: "—"}[rep.get("models_agree")]
            conf = rep.get("confidence")
            sure = f"{conf * 100:.0f}%" if (conf is not None and st == "AGREEMENT") else "—"
            consent = {"granted": "Given", "revoked": "Withdrawn"}.get(str(rep.get("consent")), "Unknown")
            head, _, ckey, _ = FRIENDLY[fsm.state]
            cols = [("Result", head, self._colour(ckey)), ("Both AIs agree?", agree, ink),
                    ("How sure", sure, ink), ("Consent", consent, ink),
                    ("Received", fsm.report_received_at or "—", ink)]
            cx = x + 12
            for label, val, col in cols:
                self._t(d, (cx, y + 30), label, 11, P["muted"])
                self._t(d, (cx, y + 46), val, 16, P["blocked"] if dismissed else col, True)
                cx += max(100, int(d.textlength(self.fonts.fix(val), font=self.fonts.get(16, True))) + 28)
            if dismissed:
                self._t(d, (x + w - 12, y + 8), "dismissed on robot", 11, P["muted"], anchor="ra")
        self._t(d, (x + 12, y + h - 22),
                "A screening aid, not a diagnosis. Open the dashboard to see why, and save the report as a PDF.",
                11, P["muted"])

    def _state_card(self, d, y, now) -> int:
        fsm = self.rt.fsm
        h = 110
        self._card(d, RX, y, RW, h)
        head, expl, ckey, _ = FRIENDLY[fsm.state]
        col = self._colour(ckey)
        d.rounded_rectangle((RX + 12, y + 14, RX + 20, y + h - 14), radius=4, fill=col)
        self._t(d, (RX + 34, y + 14), "WHAT THE ROBOT SAYS", 11, P["muted"], True)
        self._t(d, (RX + 34, y + 30), head, 25, col if ckey in ("red", "green") else P["ink"], True)
        if fsm.state is RobotState.SCREEN_FLAGGED:
            d.rounded_rectangle((RX + 34, y + 66, RX + RW - 14, y + 96), radius=6, fill=(253, 236, 234),
                                outline=RED)
            self._t(d, (RX + 46, y + 81), "Please meet a therapist.", 17, RED, True, "lm")
        else:
            self._t(d, (RX + 34, y + 70), expl, 14, P["ink"])
        if config.BORDERLINE_REVIEW:
            self._t(d, (RX + RW - 12, y + 14), "borderline rule active", 11, P["amber"], True, anchor="ra")
        return y + h

    def _oled_card(self, img, d, y) -> int:
        rt = self.rt
        h = 256 + 44
        self._card(d, RX, y, RW, h, "Robot face (screen on the device)")
        bx = RX + (RW - 512) // 2
        by = y + 30
        d.rounded_rectangle((bx - 4, by - 4, bx + 516, by + 260), radius=6, fill=(18, 20, 22))
        if rt.oled_fb is not None:
            o = to_rgb(rt.oled_fb, config.OLED_CONSOLE_SCALE, rt.oled_params.dim)
            img.paste(Image.fromarray(o), (bx, by))
            self.texts.extend(rt.oled_params.lines)
        return y + h

    def _status_card(self, d, y, now) -> None:
        rt, fsm = self.rt, self.rt.fsm
        h = H - FOOTER - GAP - y
        self._card(d, RX, y, RW, h, "Robot status")
        _, _, _, light = FRIENDLY[fsm.state]
        rows_y = y + 34
        d.ellipse((RX + 14, rows_y - 9, RX + 32, rows_y + 9),
                  fill=tuple(rt.led) if any(rt.led) else (40, 40, 42), outline=P["line"])
        self._t(d, (RX + 42, rows_y), "Light", 13, P["muted"], anchor="lm")
        self._t(d, (RX + 140, rows_y), light, 14, P["ink"], True, "lm")
        tile = rt.buzzer.tile_text(now)
        self._t(d, (RX + 42, rows_y + 28), "Sound", 13, P["muted"], anchor="lm")
        sound = rt.buzzer.last.label if (tile and rt.buzzer.last is not None) else "quiet"
        self._t(d, (RX + 140, rows_y + 28), sound, 14, P["ink"], bool(tile), "lm")
        if fsm.mode is Mode.MANUAL:
            conn, ccol = "Demo mode (not listening)", P["amber"]
        elif fsm.link_ok:
            conn, ccol = "Connected to dashboard", GREEN
        elif fsm.link_ok is None:
            conn, ccol = "Connecting…", P["muted"]
        else:
            conn, ccol = "Not connected", RED
        self._t(d, (RX + 42, rows_y + 56), "Dashboard", 13, P["muted"], anchor="lm")
        self._t(d, (RX + 140, rows_y + 56), conn, 14, ccol, True, "lm")
        self._t(d, (RX + 14, y + h - 22), "Happy face = no signs found · Sad face = please meet a therapist",
                11, P["muted"])

    def _details_card(self, d, y, now) -> None:
        rt, fsm = self.rt, self.rt.fsm
        h = H - FOOTER - GAP - y
        self._card(d, RX, y, RW, h, "Technical details (press d to hide)")
        tp, tt = rt.telemetry()
        spec = SPECS[fsm.state]
        host = rt.url.split("//", 1)[-1].split("/", 1)[0]
        status = {True: "OK", False: "OFFLINE", None: "—"}[fsm.link_ok]
        if not rt.polling_enabled:
            status = "not polling"
        ago = f"{rt.now() - fsm.last_success:.1f} s ago" if fsm.last_success is not None else "never"
        tile = rt.buzzer.tile_text(now)
        lines = [
            f"State {fsm.state.value} · {fsm.time_in_state():.1f} s · source {fsm.source}",
            f"Pan  servo  cmd {tp.cmd:6.1f}°  actual {tp.actual:6.1f}°  PWM {tp.pulse_us:4.0f} µs",
            f"Tilt servo  cmd {tt.cmd:6.1f}°  actual {tt.actual:6.1f}°  PWM {tt.pulse_us:4.0f} µs",
            f"LED {spec.led_pattern} {spec.led_hz:g} Hz · RGB {spec.led_rgb[0]},{spec.led_rgb[1]},{spec.led_rgb[2]}",
            f"Buzzer {tile or '—'}",
            f"Link {host} · {status} · last poll {ago} · seq {fsm.gate.seq if fsm.gate.seq >= 0 else '—'}",
            f"Window {self.fps:.0f} fps · 3D {'pybullet' if self.render_3d else 'lightweight' if self.soft else 'off'}",
        ]
        for i, ln in enumerate(lines):
            self._t(d, (RX + 12, y + 28 + i * 21), ln, 12, P["ink"])

    def _footer(self, d, now) -> None:
        d.rectangle((0, H - FOOTER, W, H), fill=(234, 232, 225))
        keys = ("m live/demo · 1–9 preview screens · 0 auto demo · drag to rotate · v/b camera · "
                "d details · s screenshot · q quit")
        self._t(d, (12, H - FOOTER // 2), keys, 12, P["ink"], anchor="lm")
        right = "PAUSED" if self.rt.paused else ""
        self._t(d, (W - 12, H - FOOTER // 2), right, 12, P["muted"], anchor="rm")

    def close(self) -> None:
        if self.worker is not None:
            self.worker.close()
            self.worker = None


def bgr(rgb: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(rgb[:, :, ::-1])
