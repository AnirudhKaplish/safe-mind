"""Per-state behaviour: joint targets, LED colour, OLED frame parameters.

Motion is issued as min-jerk segments to the two ``Servo`` objects; the
servos themselves enforce speed caps, limits and smooth re-planning.
"""
from __future__ import annotations

import random
from typing import Optional

from . import config
from .led import led_for
from .oled import BLINK_FRAMES, OledParams
from .servo import Servo
from .states import OVERLAY_LINES, RobotState, SPECS


class Behaviour:
    def __init__(self, seed: int = 7) -> None:
        self.rng = random.Random(seed)
        self.state = RobotState.IDLE
        self.t_enter = 0.0
        self.next_blink = 0.0
        self.blink_start = -1.0
        self._phase = 0
        self._dwell_until: Optional[float] = None
        self._sweep_next = 105.0
        self._idle_next = 92.0
        self.overlay: Optional[str] = None
        self.overlay_until = 0.0

    # -- state entry --------------------------------------------------------
    def enter(self, state: RobotState, now: float) -> None:
        self.state = RobotState(state)
        self.t_enter = now
        self._phase = 0
        self._dwell_until = None
        self._sweep_next = 105.0
        self._idle_next = 92.0
        delay = config.FLAGGED_NO_BLINK_S if self.state is RobotState.SCREEN_FLAGGED else 0.0
        self.next_blink = now + delay + self.rng.uniform(*config.BLINK_INTERVAL_S)
        if self.state is RobotState.SCREEN_FLAGGED:
            self.next_blink = now + delay + self.rng.uniform(0.5, 2.0)
        self.blink_start = -1.0
        if self.overlay == "dismiss" and self.state is not RobotState.IDLE:
            self.overlay = None

    def show_overlay(self, kind: str, now: float) -> None:
        self.overlay = kind
        self.overlay_until = now + (config.CONFIRM_OVERLAY_S if kind == "confirm" else config.DISMISS_OVERLAY_S)

    # -- motion -------------------------------------------------------------
    def step(self, now: float, pan: Servo, tilt: Servo) -> None:
        s = SPECS[self.state]
        cap = s.speed_cap_deg_s
        st = self.state
        if st is RobotState.IDLE:
            pan.set_target(90, cap)
            if not tilt.moving:
                if abs(tilt.pos - 90) > 3 and self._phase == 0:
                    tilt.set_target(90, cap)
                else:
                    self._phase = 1
                    tilt.set_target(self._idle_next, cap, duration=3.0)
                    self._idle_next = 88.0 if self._idle_next > 90 else 92.0
        elif st is RobotState.PROCESSING:
            tilt.set_target(90, cap)
            if not pan.moving:
                if self._dwell_until is None:
                    self._dwell_until = now + 0.3
                elif now >= self._dwell_until:
                    pan.set_target(self._sweep_next, cap)
                    self._sweep_next = 75.0 if self._sweep_next > 90 else 105.0
                    self._dwell_until = None
        elif st is RobotState.SCREEN_CLEAR:
            pan.set_target(90, cap)
            if not tilt.moving:
                if self._phase == 0:
                    if abs(tilt.pos - 90) < 0.5:
                        tilt.set_target(84, cap)
                        self._phase = 1
                    else:
                        tilt.set_target(90, cap)
                elif self._phase == 1:
                    tilt.set_target(90, cap)
                    self._phase = 2
        elif st is RobotState.REVIEW_NEEDED:
            tilt.set_target(90, cap)
            cyc = (now - self.t_enter) % 6.0
            pan.set_target(98 if cyc < 2.5 else 90, cap)
        else:
            pan.set_target(s.pan_deg, cap)
            tilt.set_target(s.tilt_deg, cap)

    # -- LED / OLED ---------------------------------------------------------
    def led(self, now: float) -> tuple[int, int, int]:
        return led_for(SPECS[self.state], now - self.t_enter)

    def _openness(self, now: float) -> float:
        if self.blink_start < 0 and now >= self.next_blink:
            self.blink_start = now
        if self.blink_start >= 0:
            k = (now - self.blink_start) / config.OLED_BLINK_S
            if k >= 1.0:
                self.blink_start = -1.0
                self.next_blink = now + self.rng.uniform(*config.BLINK_INTERVAL_S)
                return 1.0
            return BLINK_FRAMES[min(int(k * (len(BLINK_FRAMES) - 1) + 0.5), len(BLINK_FRAMES) - 1)]
        return 1.0

    def oled(self, now: float, pan_deg: float = 90.0) -> OledParams:
        s = SPECS[self.state]
        if self.overlay and now >= self.overlay_until:
            self.overlay = None
        lines = s.oled_lines
        icon = s.icon
        eyes = s.eyes
        openness = self._openness(now) if eyes in ("neutral", "attentive") else 1.0
        gaze = 0
        if self.state is RobotState.PROCESSING:
            dots = int((now - self.t_enter) * 2) % 4
            lines = ("Analysing" + "." * dots,)
            gaze = int(round((pan_deg - 90.0) / 15.0 * config.OLED_GAZE_MAX_PX))
        if self.overlay == "confirm":
            icon, eyes = "check", "none"
            lines = OVERLAY_LINES["confirm"]
        elif self.overlay == "dismiss":
            lines = OVERLAY_LINES["dismiss"]
        return OledParams(eyes=eyes, openness=openness, gaze_px=gaze, icon=icon,
                          lines=tuple(lines), dim=s.dim)
