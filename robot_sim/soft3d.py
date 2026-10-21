"""Lightweight 3D view of the companion robot, used when pybullet is missing.

Pure numpy + Pillow + OpenCV (already required by the console). It draws the
same robot as the PyBullet scene (dimensions from ``urdf_gen.DIMS``), moves
the head with the pan/tilt servo angles, maps the live OLED face onto the
screen and lights the LED ring. Camera presets and mouse orbit work as usual.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from . import config
from .oled import to_rgb
from .urdf_gen import BASE_TOP, DIMS, FACE_CZ, FOOT_H, HEAD_CZ, COLORS

FOV_DEG = 38.0
LIGHT = np.array([1.2, 0.8, 2.0]) / np.linalg.norm([1.2, 0.8, 2.0])
SS = 2  # supersampling factor for smooth edges

# unit cube faces: (normal, 4 corner indices) for corners in itertools order
_CORNERS = np.array([[x, y, z] for x in (-0.5, 0.5) for y in (-0.5, 0.5) for z in (-0.5, 0.5)])
_FACES = [
    (np.array([1, 0, 0]), (4, 6, 7, 5)), (np.array([-1, 0, 0]), (0, 1, 3, 2)),
    (np.array([0, 1, 0]), (2, 3, 7, 6)), (np.array([0, -1, 0]), (0, 4, 5, 1)),
    (np.array([0, 0, 1]), (1, 5, 7, 3)), (np.array([0, 0, -1]), (0, 2, 6, 4)),
]


def _rgb(name: str) -> tuple[int, int, int]:
    r, g, b, _ = COLORS[name]
    return int(r * 255), int(g * 255), int(b * 255)


def _rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


class SoftRenderer:
    """Painter's-algorithm renderer for the handful of boxes in the robot."""

    def __init__(self) -> None:
        self._bg: Optional[tuple[int, int, Image.Image]] = None

    # -- camera ---------------------------------------------------------------
    @staticmethod
    def _camera(target, distance, yaw, pitch):
        """PyBullet yaw/pitch convention (yaw 90 = camera in front, on +X)."""
        y, p = math.radians(yaw), math.radians(pitch)
        t = np.asarray(target, dtype=float)
        eye = t + distance * np.array([math.cos(p) * math.sin(y), -math.cos(p) * math.cos(y), -math.sin(p)])
        fwd = t - eye
        fwd /= np.linalg.norm(fwd)
        right = np.cross(fwd, [0, 0, 1.0])
        if np.linalg.norm(right) < 1e-6:
            right = np.array([0, 1.0, 0])
        right /= np.linalg.norm(right)
        up = np.cross(right, fwd)
        return eye, fwd, right, up

    # -- scene ----------------------------------------------------------------
    def _boxes(self, pan_deg: float, tilt_deg: float):
        """Yield (world corners (8,3), colour) for every box of the robot."""
        d = DIMS
        qp = math.radians(pan_deg - config.REST_DEG)
        qt = math.radians(tilt_deg - config.REST_DEG)
        out = []

        def box(size, centre, R=np.eye(3), origin=np.zeros(3), col=(200, 200, 200)):
            local = _CORNERS * np.asarray(size) + np.asarray(centre)
            out.append(((R @ local.T).T + origin, col))

        # fixed base
        box(d["foot"], (0, 0, FOOT_H / 2), col=_rgb("foot"))
        box(d["base"], (0, 0, FOOT_H + d["base"][2] / 2), col=_rgb("shell"))
        psx, psy, psz = d["pan_servo"]
        pan_origin = np.array([-0.012, 0, BASE_TOP])
        box(d["pan_servo"], (0, 0, psz / 2), origin=pan_origin, col=_rgb("servo"))
        # pan joint -> neck
        Rp = _rot_z(qp)
        neck_origin = pan_origin + np.array([0, 0, psz])
        nx, ny, nz = d["neck"]
        box(d["neck"], (0, 0, nz / 2), Rp, neck_origin, _rgb("neck"))
        tx, ty, tz = d["tilt_servo"]
        box(d["tilt_servo"], (0, ny / 2 + ty / 2, nz - tz / 2 + 0.004), Rp, neck_origin, _rgb("servo"))
        # tilt joint -> head
        Rh = Rp @ _rot_y(qt)
        head_origin = neck_origin + Rp @ np.array([0, 0, nz])
        hx, hy, hz = d["head"]
        box(d["head"], (0, 0, HEAD_CZ), Rh, head_origin, _rgb("shell"))
        zx, zy, zz = d["bezel"]
        box(d["bezel"], (hx / 2 + zx / 2, 0, FACE_CZ), Rh, head_origin, _rgb("bezel"))
        # face quad (YZ plane facing +X), returned separately
        fw, fh = d["face"]
        fxo = hx / 2 + zx + 0.0006
        quad = np.array([[fxo, -fw / 2, FACE_CZ - fh / 2], [fxo, fw / 2, FACE_CZ - fh / 2],
                         [fxo, fw / 2, FACE_CZ + fh / 2], [fxo, -fw / 2, FACE_CZ + fh / 2]])
        face = (Rh @ quad.T).T + head_origin
        face_normal = Rh @ np.array([1.0, 0, 0])
        return out, face, face_normal

    def _background(self, w: int, h: int) -> Image.Image:
        if self._bg and self._bg[0] == w and self._bg[1] == h:
            return self._bg[2].copy()
        top, bot = np.array([236, 238, 240]), np.array([214, 216, 220])
        k = np.linspace(0, 1, h)[:, None]
        col = (top * (1 - k) + bot * k).astype(np.uint8)
        img = Image.fromarray(np.repeat(col[:, None, :], w, axis=1))
        self._bg = (w, h, img)
        return img.copy()

    # -- render ---------------------------------------------------------------
    def render(self, pan_deg: float, tilt_deg: float, led_rgb, oled_fb, oled_dim: bool,
               target, distance, yaw, pitch, w: int, h: int) -> np.ndarray:
        W, H = w * SS, h * SS
        eye, fwd, right, up = self._camera(target, distance, yaw, pitch)
        f = (H / 2) / math.tan(math.radians(FOV_DEG) / 2)

        def proj(pts):
            v = np.atleast_2d(pts) - eye
            z = v @ fwd
            z = np.maximum(z, 1e-4)
            x = (v @ right) * f / z + W / 2
            y = -(v @ up) * f / z + H / 2
            return np.stack([x, y], 1), z

        img = self._background(W, H)
        d = ImageDraw.Draw(img)

        # desk: a large wooden square, clipped to points in front of the camera
        s = 0.6
        desk = np.array([[-s, -s, 0], [s, -s, 0], [s, s, 0], [-s, s, 0]], dtype=float)
        dp, dz = proj(desk)
        if (dz > 0.01).all():
            d.polygon([tuple(p) for p in dp], fill=(204, 173, 133))
        else:  # camera very low: draw the horizon band instead
            hy = proj(np.array([[eye[0] + fwd[0] * 5, eye[1] + fwd[1] * 5, 0]]))[0][0][1]
            d.rectangle((0, max(0, hy), W, H), fill=(204, 173, 133))

        # soft contact shadow
        sh = Image.new("L", (W, H), 0)
        sd = ImageDraw.Draw(sh)
        ring = np.array([[0.075 * math.cos(a), 0.065 * math.sin(a), 0.0005]
                         for a in np.linspace(0, 2 * math.pi, 32, endpoint=False)])
        rp, _ = proj(ring)
        sd.polygon([tuple(p) for p in rp], fill=110)
        sh = sh.filter(ImageFilter.GaussianBlur(10 * SS))
        img.paste((120, 95, 70), (0, 0), sh)

        boxes, face, face_n = self._boxes(pan_deg, tilt_deg)
        polys = []
        for corners, col in boxes:
            centre = corners.mean(0)
            for n_local, idx in _FACES:
                pts = corners[list(idx)]
                n = np.cross(pts[1] - pts[0], pts[3] - pts[0])
                nn = np.linalg.norm(n)
                if nn < 1e-12:
                    continue
                n /= nn
                if np.dot(n, pts.mean(0) - centre) < 0:
                    n = -n
                if np.dot(n, eye - pts.mean(0)) <= 0:
                    continue  # back face
                shade = 0.58 + 0.5 * max(0.0, float(np.dot(n, LIGHT)))
                c = tuple(int(min(255, v * shade)) for v in col)
                p2, z = proj(pts)
                polys.append((float(z.mean()), "poly", p2, c))

        # LED ring on top of the base (cylinder top as an ellipse)
        r, lh = DIMS["led_ring"]
        lc = np.array([DIMS["base"][0] / 2 - 0.022, 0, BASE_TOP + lh])
        ring = np.array([lc + [r * math.cos(a), r * math.sin(a), 0]
                         for a in np.linspace(0, 2 * math.pi, 24, endpoint=False)])
        rp, rz = proj(ring)
        on = led_rgb is not None and any(led_rgb)
        lcol = tuple(int(v) for v in led_rgb) if on else _rgb("led")
        polys.append((float(rz.mean()) - 1e-4, "led", rp, lcol))

        # face texture
        if np.dot(face_n, eye - face.mean(0)) > 0 and oled_fb is not None:
            fp, fz = proj(face)
            polys.append((float(fz.mean()) - 2e-4, "face", fp, None))

        polys.sort(key=lambda t: -t[0])   # far to near
        arr = None
        for _, kind, p2, c in polys:
            if kind == "poly":
                d.polygon([tuple(p) for p in p2], fill=c, outline=tuple(max(0, v - 25) for v in c))
            elif kind == "led":
                if on:   # glow
                    glow = Image.new("L", (W, H), 0)
                    ImageDraw.Draw(glow).polygon([tuple(p) for p in p2], fill=200)
                    glow = glow.filter(ImageFilter.GaussianBlur(8 * SS))
                    img.paste(c, (0, 0), glow)
                    d = ImageDraw.Draw(img)
                d.polygon([tuple(p) for p in p2], fill=c)
            elif kind == "face":
                import cv2
                tex = to_rgb(oled_fb, 4, oled_dim)
                th, tw = tex.shape[:2]
                src = np.float32([[0, th], [tw, th], [tw, 0], [0, 0]])
                M = cv2.getPerspectiveTransform(src, np.float32(p2))
                arr = np.asarray(img).copy()
                warped = cv2.warpPerspective(tex, M, (W, H), flags=cv2.INTER_LINEAR)
                mask = cv2.warpPerspective(np.full((th, tw), 255, np.uint8), M, (W, H))
                m = (mask > 127)[..., None]
                arr = np.where(m, warped, arr)
                img = Image.fromarray(arr)
                d = ImageDraw.Draw(img)

        return np.asarray(img.resize((w, h), Image.LANCZOS))
