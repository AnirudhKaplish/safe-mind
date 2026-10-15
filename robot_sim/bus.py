"""Thread-safe holder of the latest robot command (used by the dashboard).

Standard library only. One module-level singleton, ``robot_bus``, keeps the
latest payload, a monotonically increasing ``seq`` and a random ``boot_id``
created once per process start.
"""
from __future__ import annotations

import copy
import secrets
import threading

from .controller import RobotCommand, idle_command, to_payload


class RobotBus:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.boot_id = secrets.token_hex(4)
        self._seq = 0
        self._payload = to_payload(idle_command(), self.boot_id, 0)

    def publish(self, cmd: RobotCommand) -> int:
        if not isinstance(cmd, RobotCommand):
            raise TypeError("robot_bus.publish expects a RobotCommand")
        with self._lock:
            self._seq += 1
            self._payload = to_payload(cmd, self.boot_id, self._seq)
            return self._seq

    def snapshot(self) -> dict:
        with self._lock:
            return copy.deepcopy(self._payload)

    @property
    def seq(self) -> int:
        with self._lock:
            return self._seq


robot_bus = RobotBus()
