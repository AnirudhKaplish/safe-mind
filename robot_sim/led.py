"""RGB LED patterns: steady / pulse / blink / off -> colour per frame."""
from __future__ import annotations

import math

from .states import StateSpec


def led_rgb(rgb: tuple[int, int, int], pattern: str, hz: float, t: float) -> tuple[int, int, int]:
    if pattern == "off":
        return (0, 0, 0)
    if pattern == "steady" or hz <= 0:
        k = 1.0
    elif pattern == "pulse":
        # smooth breathing between 25 % and 100 %
        k = 0.25 + 0.75 * (0.5 - 0.5 * math.cos(2 * math.pi * hz * t))
    elif pattern == "blink":
        k = 1.0 if (t * hz) % 1.0 < 0.5 else 0.0
    else:
        raise ValueError(f"unknown LED pattern {pattern!r}")
    return tuple(int(round(c * k)) for c in rgb)  # type: ignore[return-value]


def led_for(spec: StateSpec, t_in_state: float) -> tuple[int, int, int]:
    return led_rgb(spec.led_rgb, spec.led_pattern, spec.led_hz, t_in_state)


def rgba(rgb: tuple[int, int, int], glow_floor: float = 0.12) -> list[float]:
    """Colour for PyBullet. An 'off' LED is shown as dark smoked plastic."""
    if rgb == (0, 0, 0):
        return [0.10, 0.10, 0.11, 1.0]
    return [max(glow_floor, c / 255.0) for c in rgb] + [1.0]
