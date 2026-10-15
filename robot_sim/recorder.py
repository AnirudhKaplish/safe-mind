"""MP4 recording and PNG screenshots of the console (into runs/robot/).

Frames are whatever the console composed, so the LIVE/MANUAL banner is
always burned in.
"""
from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from . import config


def stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def screenshot(frame_bgr: np.ndarray, directory: Path = config.RUNS_DIR) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"shot_{stamp()}.png"
    cv2.imwrite(str(path), frame_bgr)
    return path


class Recorder:
    """Writes a constant-frame-rate MP4. ``add(frame, t)`` repeats or skips
    frames so the video timeline follows ``t`` (seconds since start)."""

    def __init__(self, path: Path, fps: int = 30, size=(config.CONSOLE_W, config.CONSOLE_H)) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.size = size
        self.writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        if not self.writer.isOpened():
            raise RuntimeError(f"could not open video writer for {self.path}")
        self.frames = 0
        self.t0: Optional[float] = None

    def add(self, frame_bgr: np.ndarray, t: Optional[float] = None) -> None:
        if frame_bgr.shape[1::-1] != self.size:
            frame_bgr = cv2.resize(frame_bgr, self.size)
        if t is None:
            self.writer.write(frame_bgr)
            self.frames += 1
            return
        if self.t0 is None:
            self.t0 = t
        target = int((t - self.t0) * self.fps) + 1
        while self.frames < target:
            self.writer.write(frame_bgr)
            self.frames += 1

    def close(self) -> None:
        if self.writer is not None:
            self.writer.release()
            self.writer = None
