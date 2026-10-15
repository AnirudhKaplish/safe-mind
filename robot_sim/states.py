"""Robot states and their per-state specification (single source of truth).

Standard library only. The dashboard builds its payload ``device`` block from
this table, and the simulator renders from the same table, so the two can
never disagree.

The robot shows *system states*, never a person's emotions: no mouth, no
clinical words (see ``FORBIDDEN_WORDS`` and the unit tests).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class RobotState(str, Enum):
    IDLE = "IDLE"
    PROCESSING = "PROCESSING"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    CONSENT_BLOCKED = "CONSENT_BLOCKED"
    SCREEN_CLEAR = "SCREEN_CLEAR"
    SCREEN_FLAGGED = "SCREEN_FLAGGED"
    REVIEW_NEEDED = "REVIEW_NEEDED"
    ERROR = "ERROR"
    OFFLINE = "OFFLINE"

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return self.value


# Order of spec table 5.1 -> keys 1..9 in manual mode.
STATE_ORDER: tuple[RobotState, ...] = (
    RobotState.IDLE,
    RobotState.PROCESSING,
    RobotState.NOT_APPLICABLE,
    RobotState.CONSENT_BLOCKED,
    RobotState.SCREEN_CLEAR,
    RobotState.SCREEN_FLAGGED,
    RobotState.REVIEW_NEEDED,
    RobotState.ERROR,
    RobotState.OFFLINE,
)

# States that only carry a report summary when produced from a report.
RESULT_STATES = frozenset({
    RobotState.NOT_APPLICABLE, RobotState.CONSENT_BLOCKED, RobotState.SCREEN_CLEAR,
    RobotState.SCREEN_FLAGGED, RobotState.REVIEW_NEEDED, RobotState.ERROR,
})


@dataclass(frozen=True)
class BuzzerEvent:
    event: str            # machine name, e.g. "soft_chime"
    label: str            # human label, e.g. "soft chime"
    freq_hz: tuple[int, ...]
    ms: int

    def as_dict(self) -> dict:
        return {"event": self.event, "freq_hz": list(self.freq_hz), "ms": self.ms}

    def describe(self) -> str:
        freqs = "/".join(str(f) for f in self.freq_hz)
        return f"{self.label} {freqs} Hz · {self.ms} ms"


@dataclass(frozen=True)
class StateSpec:
    state: RobotState
    label: str                      # human label for the console
    pan_deg: float                  # rest pose of the state
    tilt_deg: float
    speed_cap_deg_s: float
    led_rgb: tuple[int, int, int]
    led_pattern: str                # steady | pulse | blink | off
    led_hz: float
    oled_lines: tuple[str, ...]     # at most two lines, <= 21 chars each
    icon: Optional[str]             # None | lock | question | bang
    eyes: str                       # neutral | attentive | none | sleepy
    motion: str                     # human description of the motion
    buzzer: Optional[BuzzerEvent] = None
    dim: bool = False

    def device_dict(self) -> dict:
        """The payload ``device`` block (spec 10.2)."""
        return {
            "pan_deg": self.pan_deg,
            "tilt_deg": self.tilt_deg,
            "led": {"rgb": list(self.led_rgb), "pattern": self.led_pattern, "hz": self.led_hz},
            "oled": list(self.oled_lines),
            "icon": _ICON_NAMES.get(self.icon),
            "buzzer": self.buzzer.as_dict() if self.buzzer else None,
        }


_ICON_NAMES = {None: None, "lock": "lock", "question": "question", "bang": "exclamation"}

# Buzzer tones. The firmware sketch (hardware/mhd_hardware.ino) is the
# reference; align these numbers with the tone() calls there if they differ.
TICK = BuzzerEvent("tick", "short tick", (2000,), 30)
TWO_LOW_TICKS = BuzzerEvent("two_low_ticks", "two low ticks", (400, 400), 40)
SOFT_CHIME = BuzzerEvent("soft_chime", "soft chime", (880,), 200)
SOFT_TONE = BuzzerEvent("soft_tone", "single soft tone", (660,), 250)
TWO_TONE = BuzzerEvent("two_tone", "two-tone cue", (600, 800), 150)

SPECS: dict[RobotState, StateSpec] = {
    RobotState.IDLE: StateSpec(
        RobotState.IDLE, "Idle", 90, 90, 40, (180, 180, 170), "steady", 0.0,
        ("Ready",), None, "neutral", "rest, slow tilt micro-motion ±2°"),
    RobotState.PROCESSING: StateSpec(
        RobotState.PROCESSING, "Analysing", 90, 90, 25, (40, 110, 220), "pulse", 1.0,
        ("Analysing",), None, "neutral", "pan sweep 75↔105°"),
    RobotState.NOT_APPLICABLE: StateSpec(
        RobotState.NOT_APPLICABLE, "Not screened", 90, 90, 40, (90, 90, 85), "steady", 0.0,
        ("Not screened", "Not a personal post"), None, "neutral", "still",
        buzzer=TICK),
    RobotState.CONSENT_BLOCKED: StateSpec(
        RobotState.CONSENT_BLOCKED, "Consent blocked", 90, 100, 30, (120, 125, 122), "steady", 0.0,
        ("Consent not valid", "Nothing processed"), "lock", "none", "still, slightly lowered",
        buzzer=TWO_LOW_TICKS),
    RobotState.SCREEN_CLEAR: StateSpec(
        RobotState.SCREEN_CLEAR, "Screen complete", 90, 90, 40, (60, 170, 90), "steady", 0.0,
        ("Screen complete", "No indicators flagged"), None, "neutral", "one gentle nod",
        buzzer=SOFT_CHIME),
    RobotState.SCREEN_FLAGGED: StateSpec(
        RobotState.SCREEN_FLAGGED, "Indicators flagged", 90, 86, 30, (230, 160, 40), "steady", 0.0,
        # Spec text "Clinician review advised" is 24 chars and does not fit
        # the 21-char OLED line; shortened, meaning kept.
        ("Indicators flagged", "Clinician to review"), None, "attentive",
        "still, attentive, facing clinician", buzzer=SOFT_TONE),
    RobotState.REVIEW_NEEDED: StateSpec(
        RobotState.REVIEW_NEEDED, "Review needed", 90, 90, 30, (230, 160, 40), "pulse", 0.5,
        ("Models disagree", "Review needed"), "question", "neutral",
        "small look-aside every 6 s", buzzer=TWO_TONE),
    RobotState.ERROR: StateSpec(
        RobotState.ERROR, "Error", 90, 90, 40, (120, 125, 122), "blink", 1.0,
        ("Result unavailable",), "bang", "none", "still"),
    RobotState.OFFLINE: StateSpec(
        RobotState.OFFLINE, "Offline", 90, 110, 15, (0, 0, 0), "off", 0.0,
        ("No connection",), None, "sleepy", "slow droop", dim=True),
}

# Local overlay lines (buttons). Also covered by the wording tests.
OVERLAY_LINES = {
    "confirm": ("Acknowledged",),
    "dismiss": ("Dismissed",),
}

# Words that must never appear on the device (OLED / LED legend), spec 5.4.
FORBIDDEN_WORDS = (
    "depression", "depressed", "risk", "high", "low risk", "diagnosis", "diagnosed",
    "patient", "emotion", "sad", "happy", "anxiety", "anxious", "suicid", "disorder",
    "symptom", "mental", "illness", "clinical", "severe", "mood",
)


def spec(state: RobotState | str) -> StateSpec:
    return SPECS[RobotState(state)]


def all_device_strings() -> list[str]:
    """Every string the device can ever show (for tests)."""
    out: list[str] = []
    for s in SPECS.values():
        out.extend(s.oled_lines)
        out.append(s.led_pattern)
    for lines in OVERLAY_LINES.values():
        out.extend(lines)
    out.append("Analysing...")
    return out
