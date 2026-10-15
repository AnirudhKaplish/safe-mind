"""Buzzer tone events.

Shown visually in the console's BUZZER tile. Real audio is off by default;
``--sound`` plays tones only where the standard library can (``winsound``
on Windows). On macOS/Linux there is no stdlib tone generator, so the flag
just prints a notice: no new dependency is added for sound.
"""
from __future__ import annotations

import sys
import threading
from typing import Optional

from . import config
from .states import BuzzerEvent


class Buzzer:
    def __init__(self, sound: bool = False) -> None:
        self.sound = sound and sys.platform.startswith("win")
        self.requested_sound = sound
        self.last: Optional[BuzzerEvent] = None
        self.last_t = -1e9

    @property
    def sound_note(self) -> str:
        if not self.requested_sound:
            return "audio off"
        return "audio on (winsound)" if self.sound else "audio unavailable on this OS (visual only)"

    def trigger(self, event: Optional[BuzzerEvent], now: float) -> None:
        if event is None:
            return
        self.last, self.last_t = event, now
        if self.sound:
            threading.Thread(target=self._play, args=(event,), daemon=True).start()

    @staticmethod
    def _play(event: BuzzerEvent) -> None:  # pragma: no cover - Windows only
        import winsound
        for f in event.freq_hz:
            winsound.Beep(int(f), int(event.ms))

    def tile_text(self, now: float) -> Optional[str]:
        if self.last is not None and now - self.last_t <= config.BUZZER_TILE_S:
            return self.last.describe()
        return None
