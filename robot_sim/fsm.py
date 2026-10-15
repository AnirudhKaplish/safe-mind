"""Robot state machine: live/manual arbitration, link health, timeouts.

Every transition is logged (timestamp, from, to, cause) in a ring buffer.
Causes are closed-vocabulary strings: never report content.
"""
from __future__ import annotations

import csv
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

from . import config
from .controller import validate_payload
from .link import BAD_JSON, PayloadGate
from .states import RESULT_STATES, RobotState

S = RobotState


class Mode(str, Enum):
    LIVE = "LIVE"
    MANUAL = "MANUAL"


_ALL = frozenset(RobotState)
# Explicit transition table. Live payloads may move to any server state;
# ERROR/OFFLINE are reachable from everywhere; Dismiss -> IDLE only from
# result states (it must not hide PROCESSING or OFFLINE).
ALLOWED: dict[str, dict[RobotState, frozenset]] = {
    "live": {s: _ALL - {S.OFFLINE} for s in RobotState},
    "link": {s: frozenset({S.OFFLINE}) for s in RobotState},
    "error": {s: frozenset({S.ERROR}) for s in RobotState},
    "manual": {s: _ALL for s in RobotState},
    "dismiss": {s: (frozenset({S.IDLE}) if s in RESULT_STATES else frozenset()) for s in RobotState},
    "startup": {s: frozenset({S.IDLE, S.OFFLINE}) for s in RobotState},
}


@dataclass
class Transition:
    wall: str
    mono: float
    frm: str
    to: str
    cause: str
    mode: str
    seq: Optional[int]


class RobotFSM:
    def __init__(self, mode: Mode = Mode.LIVE, clock: Callable[[], float] = time.monotonic,
                 processing_timeout: float = config.PROCESSING_TIMEOUT_S,
                 offline_after: int = config.OFFLINE_AFTER_FAILURES) -> None:
        self.clock = clock
        self.mode = Mode(mode)
        self.processing_timeout = processing_timeout
        self.offline_after = offline_after
        now = clock()
        self.state = S.IDLE
        self.source = "startup"
        self.cause = "startup"
        self.entered_at = now
        self.generation = 0                 # bumps on every (re-)entry
        self.report: Optional[dict] = None  # sanitised summary from payload
        self.report_received_at: Optional[str] = None
        self.report_dismissed = False
        self.seq: Optional[int] = None
        self.gate = PayloadGate()
        # link telemetry
        self.link_ok: Optional[bool] = None
        self.failures = 0
        self.last_success: Optional[float] = None
        self.last_latency: Optional[float] = None
        self.last_error = ""
        self.invalid_payloads = 0
        self.last_server: Optional[dict] = None
        self.live_waiting = False
        self.flash: Optional[tuple[str, float]] = None
        self.log: deque[Transition] = deque(maxlen=1000)
        self.events: deque[tuple[str, str]] = deque(maxlen=200)
        self.listeners: list[Callable[[Transition], None]] = []

    # -- core ---------------------------------------------------------------
    def _go(self, to: RobotState, kind: str, cause: str, *, source: str,
            report: Any = "keep", seq: Optional[int] = None) -> bool:
        to = RobotState(to)
        if to not in ALLOWED[kind][self.state]:
            self.events.append((datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                f"refused {kind}:{self.state.value}->{to.value}"))
            return False
        now = self.clock()
        tr = Transition(datetime.now(timezone.utc).isoformat(timespec="milliseconds"), now,
                        self.state.value, to.value, cause, self.mode.value, seq)
        self.log.append(tr)
        self.state, self.entered_at, self.source, self.cause = to, now, source, cause
        self.generation += 1
        if report != "keep":
            self.report = report
        if seq is not None:
            self.seq = seq
        for fn in self.listeners:
            fn(tr)
        return True

    def time_in_state(self) -> float:
        return self.clock() - self.entered_at

    def set_flash(self, text: str, seconds: float = 2.0) -> None:
        self.flash = (text, self.clock() + seconds)

    def current_flash(self) -> Optional[str]:
        if self.flash and self.clock() < self.flash[1]:
            return self.flash[0]
        return None

    # -- link events --------------------------------------------------------
    def on_poll_success(self, payload: Any, latency: float = 0.0) -> None:
        now = self.clock()
        was_offline = self.link_ok is False
        self.failures = 0
        self.link_ok = True
        self.last_success = now
        self.last_latency = latency
        err = "bad_json" if payload is BAD_JSON else validate_payload(payload)
        if err is not None:
            self.invalid_payloads += 1
            if self.mode is Mode.LIVE:
                self._go(S.ERROR, "error", "invalid payload", source="link",
                         report=None)
            return
        if was_offline:
            self.gate.force(payload)
        elif not self.gate.accept(payload):
            return
        self.last_server = payload
        if self.mode is Mode.LIVE:
            self._apply(payload, "live seq" if not was_offline else "link restored")
        else:
            self.live_waiting = True

    def on_poll_failure(self, error: str = "unreachable") -> None:
        self.failures += 1
        self.last_error = error
        if self.failures >= self.offline_after and self.link_ok is not False:
            self.link_ok = False
            if self.mode is Mode.LIVE:
                self._go(S.OFFLINE, "link", f"link lost ({error})", source="link")

    def _apply(self, payload: dict, cause: str) -> None:
        rep = payload.get("report")
        self.report_dismissed = False
        if rep is not None:
            self.report_received_at = datetime.now().strftime("%H:%M:%S")
        self._go(RobotState(payload["state"]), "live", cause, source="live report",
                 report=rep if rep is not None else self.report, seq=payload["seq"])

    # -- mode / manual ------------------------------------------------------
    def set_mode(self, mode: Mode) -> None:
        mode = Mode(mode)
        if mode is self.mode:
            return
        self.mode = mode
        if mode is Mode.MANUAL:
            self.live_waiting = False
            self.events.append((datetime.now(timezone.utc).isoformat(timespec="seconds"), "manual mode"))
            return
        # back to LIVE
        self.live_waiting = False
        if self.link_ok is False:
            self._go(S.OFFLINE, "link", "returned to live: link down", source="link")
        elif self.last_server is not None:
            self._apply(self.last_server, "returned to live")
            rid = (self.last_server.get("report") or {}).get("report_id")
            self.set_flash(f"Returned to LIVE — showing report #{rid}" if rid is not None
                           else "Returned to LIVE — showing dashboard state")
        else:
            self._go(S.IDLE, "manual", "returned to live: awaiting link", source="link", report=None)
            self.set_flash("Returned to LIVE — waiting for dashboard")

    def manual_select(self, state: RobotState) -> bool:
        if self.mode is not Mode.MANUAL:
            return False
        return self._go(state, "manual", "manual key", source="manual", report=None)

    def dismiss(self) -> bool:
        ok = self._go(S.IDLE, "dismiss", "dismiss button", source="button")
        if ok:
            self.report_dismissed = True
        return ok

    def confirm(self) -> None:
        self.events.append((datetime.now(timezone.utc).isoformat(timespec="seconds"), "acknowledged"))

    def tick(self) -> None:
        if (self.state is S.PROCESSING and self.mode is Mode.LIVE
                and self.time_in_state() > self.processing_timeout):
            self._go(S.ERROR, "error", "timeout", source="link", report=None)

    # -- export -------------------------------------------------------------
    def write_csv(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["utc_time", "monotonic_s", "from", "to", "cause", "mode", "seq"])
            for t in self.log:
                w.writerow([t.wall, f"{t.mono:.3f}", t.frm, t.to, t.cause, t.mode,
                            "" if t.seq is None else t.seq])
        return path
