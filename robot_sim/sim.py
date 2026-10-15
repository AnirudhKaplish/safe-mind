"""PyBullet scene: URDF, SG90 motors, OLED face texture, LED, cameras."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
try:
    import pybullet as p
except ImportError:
    p = None

from . import config, urdf_gen
from .led import rgba


def deg_to_q(deg: float) -> float:
    return math.radians(deg - config.REST_DEG)


def q_to_deg(q: float) -> float:
    return math.degrees(q) + config.REST_DEG


@dataclass(frozen=True)
class CameraPose:
    name: str
    target: tuple[float, float, float]
    distance: float
    yaw: float      # PyBullet convention, degrees
    pitch: float


# Presets: yaw 90 looks along -X, i.e. the camera sits in front (+X) of the face.
CAMERA_PRESETS: tuple[CameraPose, ...] = (
    CameraPose("front", (0.0, 0.0, 0.105), 0.38, 90.0, -8.0),
    CameraPose("three_quarter", (0.0, 0.0, 0.10), 0.38, 130.0, -22.0),
    CameraPose("side", (0.0, 0.0, 0.11), 0.44, 180.0, -8.0),
    CameraPose("top", (0.0, 0.0, 0.08), 0.42, 90.0, -85.0),
    CameraPose("servo_closeup", (-0.01, 0.0, 0.09), 0.17, 145.0, -18.0),
    CameraPose("face_closeup", (0.02, 0.0, 0.145), 0.17, 90.0, -4.0),
)


def view_matrix(target, distance, yaw, pitch):
    return p.computeViewMatrixFromYawPitchRoll(target, distance, yaw, pitch, 0, 2)


def min_jerk_s(s: float) -> float:
    s = min(max(s, 0.0), 1.0)
    return 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5


class CameraRig:
    """Preset cycling with 0.6 s min-jerk glides, plus optional orbit offsets."""

    def __init__(self, presets=CAMERA_PRESETS) -> None:
        self.presets = presets
        self.index = 0
        self.cur = np.array(self._vec(presets[0]))
        self.src = self.cur.copy()
        self.dst = self.cur.copy()
        self.t0 = -1e9
        self.orbit = np.zeros(2)     # yaw, pitch offsets from mouse drag

    @staticmethod
    def _vec(c: CameraPose):
        return [*c.target, c.distance, c.yaw, c.pitch]

    @property
    def name(self) -> str:
        return self.presets[self.index].name

    def go(self, index: int, now: float) -> None:
        self.index = index % len(self.presets)
        self.src = self.cur.copy()
        self.dst = np.array(self._vec(self.presets[self.index]))
        # take the short way round in yaw
        dy = (self.dst[4] - self.src[4] + 180) % 360 - 180
        self.dst[4] = self.src[4] + dy
        self.orbit[:] = 0
        self.t0 = now

    def next(self, now): self.go(self.index + 1, now)
    def prev(self, now): self.go(self.index - 1, now)
    def reset(self, now): self.go(self.index, now)

    def update(self, now: float):
        s = min_jerk_s((now - self.t0) / config.CAMERA_GLIDE_S)
        self.cur = self.src + (self.dst - self.src) * s
        tx, ty, tz, dist, yaw, pitch = self.cur
        return (tx, ty, tz), dist, yaw + self.orbit[0], max(-89, min(5, pitch + self.orbit[1]))


class TwinSim:
    def __init__(self, gui: bool = False, urdf_dir: Path = config.URDF_DIR) -> None:
        self.gui = gui
        self.cid = p.connect(p.GUI if gui else p.DIRECT)
        p.setTimeStep(1.0 / config.PHYSICS_HZ, physicsClientId=self.cid)
        p.setGravity(0, 0, -9.81, physicsClientId=self.cid)
        if gui:
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self.cid)
        self._desk()
        urdf = urdf_gen.write(urdf_dir)
        flags = p.URDF_USE_SELF_COLLISION | p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT | p.URDF_USE_INERTIA_FROM_FILE
        self.robot = p.loadURDF(str(urdf), [0, 0, 0], useFixedBase=True, flags=flags, physicsClientId=self.cid)
        self.joint = {}
        self.link = {"base": -1}
        for j in range(p.getNumJoints(self.robot, physicsClientId=self.cid)):
            info = p.getJointInfo(self.robot, j, physicsClientId=self.cid)
            self.joint[info[1].decode()] = j
            self.link[info[12].decode()] = j
        self.revolute = [n for n, j in self.joint.items()
                         if p.getJointInfo(self.robot, j, physicsClientId=self.cid)[2] == p.JOINT_REVOLUTE]
        for name in ("pan", "tilt"):
            p.resetJointState(self.robot, self.joint[name], 0.0, physicsClientId=self.cid)
        self.set_servo_targets(90.0, 90.0)
        self._textures: dict[str, int] = {}
        self._face_hash: Optional[str] = None
        self._led_last = (-1.0, None)
        self.texture_swaps = 0

    def _desk(self) -> None:
        half = [0.6, 0.6, 0.01]
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=[0.80, 0.68, 0.52, 1],
                                  specularColor=[0.05, 0.05, 0.05], physicsClientId=self.cid)
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half, physicsClientId=self.cid)
        self.desk = p.createMultiBody(0, col, vis, [0, 0, -0.01], physicsClientId=self.cid)
        # baked soft contact shadow under the robot (keeps it grounded when
        # real-time shadows are switched off for speed)
        for r, a in ((0.080, 0.05), (0.072, 0.08), (0.064, 0.11)):
            sh = p.createVisualShape(p.GEOM_CYLINDER, radius=r, length=0.0004,
                                     rgbaColor=[0.80 * (1 - a), 0.68 * (1 - a), 0.52 * (1 - a), 1],
                                     physicsClientId=self.cid)
            p.createMultiBody(0, -1, sh, [-0.002, 0.0, 0.0003 + a * 0.002], physicsClientId=self.cid)
        # a few darker plank stripes for a procedural wood look
        for i, x in enumerate(np.linspace(-0.5, 0.5, 6)):
            sv = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.004, 0.6, 0.0004],
                                     rgbaColor=[0.70, 0.57, 0.42, 1], physicsClientId=self.cid)
            p.createMultiBody(0, -1, sv, [x, 0, 0.0001], physicsClientId=self.cid)

    # -- actuation ----------------------------------------------------------
    def set_servo_targets(self, pan_deg: float, tilt_deg: float) -> None:
        for name, deg in (("pan", pan_deg), ("tilt", tilt_deg)):
            p.setJointMotorControl2(
                self.robot, self.joint[name], p.POSITION_CONTROL, targetPosition=deg_to_q(deg),
                force=config.SG90_STALL_TORQUE_NM, maxVelocity=config.SG90_MAX_RAD_S,
                positionGain=config.MOTOR_POSITION_GAIN, velocityGain=config.MOTOR_VELOCITY_GAIN,
                physicsClientId=self.cid)

    def step(self, n: int = 1) -> None:
        for _ in range(n):
            p.stepSimulation(physicsClientId=self.cid)

    def joint_deg(self, name: str) -> tuple[float, float]:
        q, qd, *_ = p.getJointState(self.robot, self.joint[name], physicsClientId=self.cid)
        return q_to_deg(q), math.degrees(qd)

    def reset_joints(self, pan_deg: float, tilt_deg: float) -> None:
        for name, deg in (("pan", pan_deg), ("tilt", tilt_deg)):
            p.resetJointState(self.robot, self.joint[name], deg_to_q(deg), physicsClientId=self.cid)
        self.set_servo_targets(pan_deg, tilt_deg)

    # -- appearance ---------------------------------------------------------
    def set_led(self, rgb, now: Optional[float] = None) -> None:
        now = time.monotonic() if now is None else now
        last_t, last_rgb = self._led_last
        if tuple(rgb) == last_rgb or now - last_t < 1.0 / 30:
            return
        p.changeVisualShape(self.robot, self.link["led_ring"], rgbaColor=rgba(tuple(rgb)), physicsClientId=self.cid)
        self._led_last = (now, tuple(rgb))

    def set_face(self, frame_hash: str, png_path: Path) -> None:
        if frame_hash == self._face_hash:
            return
        tid = self._textures.get(frame_hash)
        if tid is None:
            tid = p.loadTexture(str(png_path), physicsClientId=self.cid)
            self._textures[frame_hash] = tid
        p.changeVisualShape(self.robot, self.link["face"], textureUniqueId=tid, physicsClientId=self.cid)
        self._face_hash = frame_hash
        self.texture_swaps += 1

    # -- sensing ------------------------------------------------------------
    def clearance_head_base(self, max_dist: float = 0.2) -> float:
        pts = p.getClosestPoints(self.robot, self.robot, max_dist, linkIndexA=self.link["head"],
                                 linkIndexB=self.link["base"], physicsClientId=self.cid)
        return min((pt[8] for pt in pts), default=max_dist)

    def render(self, target, distance, yaw, pitch, w: int = config.RENDER_W,
               h: int = config.RENDER_H, shadows: bool = True) -> np.ndarray:
        """Return an RGB uint8 image (h, w, 3)."""
        vm = view_matrix(target, distance, yaw, pitch)
        pm = p.computeProjectionMatrixFOV(fov=38, aspect=w / h, nearVal=0.01, farVal=5.0)
        _, _, rgba_px, _, _ = p.getCameraImage(
            w, h, vm, pm, shadow=1 if shadows else 0, lightDirection=[1.2, 0.8, 2.0],
            lightColor=[1, 1, 1], lightDistance=2.0, lightAmbientCoeff=0.55,
            lightDiffuseCoeff=0.55, lightSpecularCoeff=0.08,
            renderer=p.ER_TINY_RENDERER, physicsClientId=self.cid)
        img = np.reshape(np.asarray(rgba_px, dtype=np.uint8), (h, w, 4))[:, :, :3]
        return img

    def close(self) -> None:
        try:
            p.disconnect(self.cid)
        except p.error:  # pragma: no cover
            pass


# ---------------------------------------------------------------------------
# Render settings and an optional render worker process
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RenderSettings:
    w: int = config.RENDER_W
    h: int = config.RENDER_H
    shadows: bool = True

    @property
    def label(self) -> str:
        return f"{self.w}x{self.h}{' shadows' if self.shadows else ''}"


QUALITY_LADDER = (RenderSettings(640, 480, True), RenderSettings(640, 480, False),
                  RenderSettings(480, 360, False))


def measure_fps(sim: "TwinSim", settings: RenderSettings, frames: int = 6) -> float:
    cam = CAMERA_PRESETS[0]
    sim.render(cam.target, cam.distance, cam.yaw, cam.pitch, settings.w, settings.h, settings.shadows)
    t0 = time.perf_counter()
    for _ in range(frames):
        sim.render(cam.target, cam.distance, cam.yaw, cam.pitch, settings.w, settings.h, settings.shadows)
    return frames / (time.perf_counter() - t0)


def choose_settings(sim: "TwinSim") -> tuple[RenderSettings, float, list]:
    """Measure, then pick the best quality that reaches RENDER_FPS_MIN."""
    tried = []
    for st in QUALITY_LADDER:
        fps = measure_fps(sim, st)
        tried.append((st.label, round(fps, 1)))
        if fps >= config.RENDER_FPS_MIN:
            return st, fps, tried
    return QUALITY_LADDER[-1], fps, tried


def _worker_main(conn, urdf_dir: str) -> None:  # pragma: no cover - runs in child
    sim = TwinSim(gui=False, urdf_dir=Path(urdf_dir))
    try:
        while True:
            msg = conn.recv()
            if msg is None:
                break
            pan, tilt, led, face, cam, st = msg
            sim.reset_joints(pan, tilt)
            if led is not None:
                sim.set_led(led, now=1e12)
                sim._led_last = (-1.0, None)
            if face is not None:
                sim.set_face(*face)
            img = sim.render(*cam, st.w, st.h, st.shadows)
            conn.send(img)
    finally:
        sim.close()


class RenderWorker:
    """Renders in a child process so the console loop never waits on the
    tiny renderer. Used only when the measured in-process fps is too low."""

    def __init__(self, urdf_dir: Path = config.URDF_DIR) -> None:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        self.conn, child = ctx.Pipe()
        self.proc = ctx.Process(target=_worker_main, args=(child, str(urdf_dir)), daemon=True)
        self.proc.start()
        self.busy = False
        self.dead = False
        self.latest: Optional[np.ndarray] = None
        self.frames = 0

    @property
    def alive(self) -> bool:
        return not self.dead and self.proc.is_alive()

    def submit(self, pan, tilt, led, face, cam, settings: RenderSettings) -> None:
        if self.busy or self.dead:
            return
        try:
            self.conn.send((pan, tilt, led, (face[0], str(face[1])) if face else None, cam, settings))
            self.busy = True
        except (OSError, EOFError, BrokenPipeError):
            self.dead = True

    def poll(self) -> Optional[np.ndarray]:
        try:
            if self.busy and self.conn.poll():
                self.latest = self.conn.recv()
                self.busy = False
                self.frames += 1
        except (OSError, EOFError, BrokenPipeError):
            self.dead = True
        return self.latest

    def wait(self, timeout: float = 5.0) -> Optional[np.ndarray]:
        if self.busy and self.conn.poll(timeout):
            self.latest = self.conn.recv()
            self.busy = False
            self.frames += 1
        return self.latest

    def close(self) -> None:
        try:
            if self.busy:
                self.conn.poll(2.0) and self.conn.recv()
            self.conn.send(None)
        except (OSError, EOFError, BrokenPipeError):
            pass
        self.proc.join(timeout=3)
        if self.proc.is_alive():
            self.proc.terminate()
