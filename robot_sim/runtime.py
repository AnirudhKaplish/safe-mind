"""The twin's main loop, without any window: FSM + behaviour + servos + sim.

``TwinRuntime.step(dt)`` advances everything by ``dt`` seconds of simulated
time. The console (``console.py``) and the self-test both drive it.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Optional

from . import config
from .behaviours import Behaviour
from .buzzer import Buzzer
from .fsm import Mode, RobotFSM
from .link import LinkPoller
from .oled import TextureCache, render
from .servo import Servo
from .states import SPECS, STATE_ORDER, RobotState


@dataclass
class ServoTelemetry:
    cmd: float
    actual: float
    err: float
    pulse_us: float
    vel: float


class TwinRuntime:
    def __init__(self, *, live: bool = True, manual: bool = False, url: str = config.DEFAULT_URL,
                 poll_hz: float = config.POLL_HZ, seed: int = 7, physics: bool = True,
                 gui: bool = False, sound: bool = False,
                 clock: Optional[Callable[[], float]] = None, verbose: bool = False) -> None:
        self._virtual_t = 0.0
        self.clock = clock or self._virtual_clock
        self.verbose = verbose
        self.fsm = RobotFSM(mode=Mode.MANUAL if manual else Mode.LIVE, clock=self.clock)
        self.behaviour = Behaviour(seed)
        self.pan = Servo("pan")
        self.tilt = Servo("tilt")
        self.buzzer = Buzzer(sound)
        self.textures = TextureCache()
        self.url = url
        self.poller: Optional[LinkPoller] = LinkPoller(url, poll_hz) if (live or manual) else None
        self.polling_enabled = live and not manual
        if self.poller and self.polling_enabled:
            self.poller.start()
        self.sim = None
        if physics:
            from .sim import TwinSim, p
            if p is not None:
                self.sim = TwinSim(gui=gui)
        self.paused = False
        self._gen_seen = -1
        self._servo_acc = 0.0
        self._phys_acc = 0.0
        self.oled_fb = None
        self.oled_params = None
        self.oled_hash = ""
        self.oled_png = None
        self.led = (0, 0, 0)
        self.trace: deque = deque(maxlen=int(config.TRACE_SECONDS * config.SERVO_UPDATE_HZ))
        self.demo_cycle = False
        self._demo_next = 0.0
        self._demo_idx = 0
        self.max_track_err = {"pan": 0.0, "tilt": 0.0}
        self.limit_violations = 0
        self.fsm.listeners.append(self._on_transition)
        self._sync_behaviour()

    # -- clock --------------------------------------------------------------
    def _virtual_clock(self) -> float:
        return self._virtual_t

    def now(self) -> float:
        return self.clock()

    def _on_transition(self, tr) -> None:
        if self.verbose:
            print(f"[robot] {tr.wall} {tr.frm} -> {tr.to} ({tr.cause})", flush=True)

    def _sync_behaviour(self) -> None:
        if self.fsm.generation != self._gen_seen:
            self._gen_seen = self.fsm.generation
            now = self.now()
            self.behaviour.enter(self.fsm.state, now)
            if self.fsm.cause == "dismiss button":
                self.behaviour.show_overlay("dismiss", now)
            self.buzzer.trigger(SPECS[self.fsm.state].buzzer, now)

    # -- controls -----------------------------------------------------------
    def toggle_mode(self) -> Mode:
        new = Mode.LIVE if self.fsm.mode is Mode.MANUAL else Mode.MANUAL
        if new is Mode.MANUAL:
            self.demo_cycle = False
        if self.poller and not self.polling_enabled:
            self.polling_enabled = True
            self.poller.start()
        self.fsm.set_mode(new)
        self._sync_behaviour()
        return new

    def manual_key(self, index: int) -> bool:
        if not 0 <= index < len(STATE_ORDER):
            return False
        ok = self.fsm.manual_select(STATE_ORDER[index])
        if ok:
            self.demo_cycle = False
        self._sync_behaviour()
        return ok

    def start_demo_cycle(self) -> bool:
        if self.fsm.mode is not Mode.MANUAL:
            return False
        self.demo_cycle = True
        self._demo_idx = 0
        self._demo_next = self.now()
        return True

    def confirm(self) -> None:
        self.fsm.confirm()
        self.behaviour.show_overlay("confirm", self.now())

    def dismiss(self) -> bool:
        ok = self.fsm.dismiss()
        self._sync_behaviour()
        return ok

    # -- main step ----------------------------------------------------------
    def step(self, dt: float) -> None:
        if self.clock == self._virtual_clock:
            self._virtual_t += dt
        if self.poller and self.polling_enabled:
            for r in self.poller.drain():
                if r.ok:
                    self.fsm.on_poll_success(r.payload, r.latency)
                else:
                    self.fsm.on_poll_failure(r.error)
        if self.paused:
            return
        now = self.now()
        if self.demo_cycle and now >= self._demo_next:
            self.fsm.manual_select(STATE_ORDER[self._demo_idx % len(STATE_ORDER)])
            self._demo_idx += 1
            self._demo_next = now + config.DEMO_CYCLE_S
        self.fsm.tick()
        self._sync_behaviour()

        servo_dt = 1.0 / config.SERVO_UPDATE_HZ
        self._servo_acc += dt
        while self._servo_acc >= servo_dt - 1e-9:
            self._servo_acc -= servo_dt
            self.behaviour.step(now, self.pan, self.tilt)
            self.pan.update(servo_dt)
            self.tilt.update(servo_dt)
            if self.sim is not None:
                self.sim.set_servo_targets(self.pan.pos, self.tilt.pos)
                n = int(round(config.PHYSICS_HZ * servo_dt + self._phys_acc))
                self._phys_acc += config.PHYSICS_HZ * servo_dt - n
                self.sim.step(n)
            tp, tt = self.telemetry()
            self.trace.append((now, tp.cmd, tp.actual, tt.cmd, tt.actual))
            for name, tel, srv in (("pan", tp, self.pan), ("tilt", tt, self.tilt)):
                self.max_track_err[name] = max(self.max_track_err[name], abs(tel.err))
                lo, hi = srv.limits
                if not (lo - 0.5 <= tel.actual <= hi + 0.5):
                    self.limit_violations += 1

        self.led = self.behaviour.led(now)
        self.oled_params = self.behaviour.oled(now, self.pan.pos)
        self.oled_fb = render(self.oled_params)
        h, path, changed = self.textures.update(self.oled_fb, self.oled_params.dim)
        self.oled_hash, self.oled_png = h, path
        if self.sim is not None:
            self.sim.set_led(self.led, now)
            if changed:
                self.sim.set_face(h, path)

    # -- telemetry ----------------------------------------------------------
    def actual(self, name: str) -> tuple[float, float]:
        srv = self.pan if name == "pan" else self.tilt
        if self.sim is None:
            return srv.pos, srv.vel
        return self.sim.joint_deg(name)

    def telemetry(self) -> tuple[ServoTelemetry, ServoTelemetry]:
        out = []
        for name, srv in (("pan", self.pan), ("tilt", self.tilt)):
            act, vel = self.actual(name)
            out.append(ServoTelemetry(srv.pos, act, act - srv.pos, srv.pulse_us(), vel))
        return out[0], out[1]

    def close(self) -> None:
        if self.poller:
            self.poller.stop()
        if self.sim is not None:
            self.sim.close()
            self.sim = None
