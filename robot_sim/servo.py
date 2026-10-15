"""SG90 servo model: limits, speed cap, dead band, PWM and min-jerk motion.

A rest-to-rest move of size D at speed cap v uses the classic minimum-jerk
profile

    theta(t) = theta0 + D * (10 s^3 - 15 s^4 + 6 s^5),  s = t / T
    T        = max(T_MIN, 1.875 * |D| / v)   (peak velocity = 1.875 D / T)

When a new target arrives mid-motion the segment is re-planned from the
current commanded position, velocity and acceleration with a quintic
polynomial that ends at rest (v = a = 0). The duration is the shortest one
for which the new segment respects the speed cap, the soft limits and an
acceleration ceiling, so there is never a position jump or velocity spike.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

from . import config

log = logging.getLogger("robot_sim.servo")

_S = [i / 64 for i in range(65)]


def pulse_us(deg: float) -> float:
    """Arduino Servo library mapping (544..2400 us over 0..180 deg)."""
    lo, hi = config.SERVO_HW_LIMITS_DEG
    d = min(max(deg, lo), hi)
    return config.SERVO_PULSE_MIN_US + d / 180.0 * (config.SERVO_PULSE_MAX_US - config.SERVO_PULSE_MIN_US)


def min_jerk_duration(delta: float, v_cap: float) -> float:
    return max(config.TRAJ_T_MIN_S, 1.875 * abs(delta) / max(v_cap, 1e-6))


@dataclass
class Quintic:
    c: tuple[float, float, float, float, float, float]
    T: float

    @classmethod
    def fit(cls, p0: float, v0: float, a0: float, pf: float, T: float) -> "Quintic":
        d = pf - p0
        c3 = (20 * d - 12 * v0 * T - 3 * a0 * T * T) / (2 * T ** 3)
        c4 = (-30 * d + 16 * v0 * T + 3 * a0 * T * T) / (2 * T ** 4)
        c5 = (12 * d - 6 * v0 * T - a0 * T * T) / (2 * T ** 5)
        return cls((p0, v0, a0 / 2.0, c3, c4, c5), T)

    def eval(self, t: float) -> tuple[float, float, float]:
        t = min(max(t, 0.0), self.T)
        c0, c1, c2, c3, c4, c5 = self.c
        p = c0 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5))))
        v = c1 + t * (2 * c2 + t * (3 * c3 + t * (4 * c4 + t * 5 * c5)))
        a = 2 * c2 + t * (6 * c3 + t * (12 * c4 + t * 20 * c5))
        return p, v, a

    def extremes(self) -> tuple[float, float, float, float]:
        """(min pos, max pos, max |vel|, max |acc|) over the segment."""
        ps, vs, accs = [], [], []
        for s in _S:
            p, v, a = self.eval(s * self.T)
            ps.append(p); vs.append(abs(v)); accs.append(abs(a))
        return min(ps), max(ps), max(vs), max(accs)


class Servo:
    """One SG90 channel. Angles in degrees in the firmware's servo.write() convention."""

    def __init__(self, name: str, soft: Optional[tuple[float, float]] = None,
                 hw: tuple[float, float] = config.SERVO_HW_LIMITS_DEG,
                 init_deg: float = config.REST_DEG,
                 vmax_hw: float = config.SG90_MAX_DEG_S,
                 deadband: float = config.SERVO_DEADBAND_DEG,
                 enforce_soft: bool = True) -> None:
        self.name = name
        self.soft = soft or config.SOFT_LIMITS_DEG.get(name, hw)
        self.hw = hw
        self.vmax_hw = vmax_hw
        self.deadband = deadband
        self.enforce_soft = enforce_soft
        self.pos = float(init_deg)
        self.vel = 0.0
        self.acc = 0.0
        self.goal = float(init_deg)
        self.t = 0.0
        self._traj: Optional[Quintic] = None
        self._t0 = 0.0
        self.cap = vmax_hw
        self.clamp_events = 0
        self.last_plan_T = 0.0
        self._pending: Optional[tuple] = None
        self._seg_end = float(init_deg)

    # -- limits -------------------------------------------------------------
    @property
    def limits(self) -> tuple[float, float]:
        if self.enforce_soft:
            return max(self.soft[0], self.hw[0]), min(self.soft[1], self.hw[1])
        return self.hw

    def clamp(self, deg: float) -> float:
        lo, hi = self.limits
        out = min(max(deg, lo), hi)
        out = min(max(out, self.hw[0]), self.hw[1])
        if abs(out - deg) > 1e-9:
            self.clamp_events += 1
            log.warning("WARN %s target %.1f deg clamped to %.1f deg", self.name, deg, out)
        return out

    @property
    def moving(self) -> bool:
        return self._traj is not None

    def pulse_us(self) -> float:
        return pulse_us(self.pos)

    # -- planning -----------------------------------------------------------
    def set_target(self, deg: float, cap_deg_s: float, duration: Optional[float] = None) -> bool:
        """Request a move. Returns False if ignored by the dead band."""
        if not math.isfinite(deg):
            return False
        target = self.clamp(float(deg))
        cap = min(max(cap_deg_s, 1e-3), self.vmax_hw)
        if abs(target - self.goal) < self.deadband and (abs(cap - self.cap) < 1e-9 or not self.moving):
            return False
        if not self.moving and abs(target - self.pos) < self.deadband:
            self.goal = self.pos
            return False
        self.cap = cap
        self._plan(target, cap, duration)
        return True

    def _search(self, p0: float, v0: float, a0: float, pf: float, start: float, t_max: float,
                v_allow: float, a_allow: float, lo_ok: float, hi_ok: float) -> Optional[tuple[float, float]]:
        """Shortest duration whose quintic respects all caps, matching a0 if possible."""
        for a_init in (a0, 0.0):
            cand = start
            while cand <= t_max:
                pmin, pmax, vpk, apk = Quintic.fit(p0, v0, a_init, pf, cand).extremes()
                if vpk <= v_allow and apk <= a_allow and pmin >= lo_ok and pmax <= hi_ok:
                    return cand, a_init
                cand *= 1.06
        return None

    def _plan(self, target: float, cap: float, duration: Optional[float]) -> None:
        p0, v0, a0 = self.pos, self.vel, self.acc
        delta = target - p0
        t_nom = min_jerk_duration(delta, cap)
        if duration is not None:
            t_nom = max(t_nom, duration)
        self.goal = target
        self._pending = None
        if abs(v0) < 1e-6 and abs(a0) < 1e-6:          # rest-to-rest: classic min-jerk
            self._start(Quintic.fit(p0, 0.0, 0.0, target, t_nom), target)
            return
        v_allow = min(max(cap, abs(v0)), self.vmax_hw) * (1 + 1e-6)
        a_allow = max(5.7735 * abs(delta) / (t_nom ** 2), config.TRAJ_ACC_FLOOR_DEG_S2, abs(a0) * 1.05)
        lo, hi = self.limits
        lo_ok, hi_ok = min(lo, p0, target) - 1e-6, max(hi, p0, target) + 1e-6
        start = t_nom if duration is not None else max(config.TRAJ_T_MIN_S, 0.3 * t_nom)
        found = self._search(p0, v0, a0, target, start, max(20.0, 4 * t_nom),
                             v_allow, a_allow, lo_ok, hi_ok)
        if found is not None:
            T, a_init = found
            self._start(Quintic.fit(p0, v0, a_init, target, T), target)
            return
        # No single smooth segment fits (typically a reversal close to a
        # limit): brake smoothly to rest first, then do a rest-to-rest move.
        a_brake = max(a_allow, abs(v0) ** 2 / max(2 * 0.25, 1e-6))
        for k in range(1, 200):
            Tb = 0.02 * k
            stop = min(max(p0 + 0.5 * v0 * Tb, lo), hi)
            q = Quintic.fit(p0, v0, 0.0, stop, Tb)
            pmin, pmax, vpk, apk = q.extremes()
            if vpk <= v_allow and pmin >= lo - 1e-6 and pmax <= hi + 1e-6 and apk <= max(a_brake, 4 * a_allow):
                break
        self._start(q, stop)
        self._pending = (target, cap, duration)

    def _start(self, q: Quintic, end: float) -> None:
        self._traj = q
        self._t0 = self.t
        self._seg_end = end
        self.last_plan_T = q.T

    # -- stepping -----------------------------------------------------------
    def update(self, dt: float) -> float:
        self.t += dt
        if self._traj is None:
            self.vel = self.acc = 0.0
            return self.pos
        tau = self.t - self._t0
        if tau >= self._traj.T:
            self.pos, self.vel, self.acc = self._seg_end, 0.0, 0.0
            self._traj = None
            if self._pending is not None:
                target, cap, duration = self._pending
                self._pending = None
                self._plan(target, cap, duration)
        else:
            p, v, a = self._traj.eval(tau)
            lo, hi = self.limits      # last-resort guard; planning keeps us inside
            self.pos, self.vel, self.acc = min(max(p, lo), hi), v, a
        return self.pos

    def snap(self, deg: float) -> None:
        """Teleport (tests/self-test set-up only; never used for live motion)."""
        self.pos = self.goal = self._seg_end = float(deg)
        self.vel = self.acc = 0.0
        self._traj = None
        self._pending = None
