"""Every tunable constant of the digital twin, in one place.

Standard library only: this module is imported (indirectly) by the Flask
dashboard through ``controller.py`` / ``bus.py``.
"""
from __future__ import annotations

import math
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent
RUNS_DIR = REPO_ROOT / "runs" / "robot"
URDF_DIR = PACKAGE_DIR / "urdf"
MESH_DIR = URDF_DIR / "meshes"
ASSET_DIR = PACKAGE_DIR / "assets"
OLED_CACHE_DIR = ASSET_DIR / "oled_cache"

# --------------------------------------------------------------------------
# SG90 micro-servo (TowerPro SG90 datasheet values)
# --------------------------------------------------------------------------
SERVO_HW_LIMITS_DEG = (0.0, 180.0)          # datasheet: ~180 deg travel
SG90_SECONDS_PER_60DEG = 0.1                 # datasheet: 0.1 s / 60 deg @ 4.8 V
SG90_MAX_DEG_S = 60.0 / SG90_SECONDS_PER_60DEG   # = 600 deg/s
SG90_MAX_RAD_S = math.radians(SG90_MAX_DEG_S)    # = 10.47 rad/s
SG90_STALL_TORQUE_KGCM = 1.8                 # datasheet: 1.8 kg*cm @ 4.8 V
SG90_STALL_TORQUE_NM = SG90_STALL_TORQUE_KGCM * 9.80665 / 100.0   # = 0.1765 N*m
SERVO_PWM_HZ = 50                            # datasheet: 20 ms PWM period
# Arduino Servo library defaults (MIN_PULSE_WIDTH / MAX_PULSE_WIDTH)
SERVO_PULSE_MIN_US = 544.0
SERVO_PULSE_MAX_US = 2400.0
SERVO_DEADBAND_DEG = 1.0                     # emulated dead band
TRAJ_T_MIN_S = 0.15                          # shortest min-jerk segment
# Acceleration ceiling used only when re-planning from a moving start
# (rest-to-rest moves use the nominal min-jerk profile).
TRAJ_ACC_FLOOR_DEG_S2 = 400.0

SOFT_LIMITS_DEG = {"pan": (30.0, 150.0), "tilt": (70.0, 115.0)}
REST_DEG = 90.0

# --------------------------------------------------------------------------
# Loop timing
# --------------------------------------------------------------------------
PHYSICS_HZ = 240
SERVO_UPDATE_HZ = 50
RENDER_FPS_TARGET = 30
RENDER_FPS_MIN = 20
RENDER_W, RENDER_H = 640, 480

# PyBullet POSITION_CONTROL gains (tuned in Stage 2/5, see docs).
MOTOR_POSITION_GAIN = 0.3
MOTOR_VELOCITY_GAIN = 1.0
JOINT_DAMPING = 0.0005

# --------------------------------------------------------------------------
# State logic
# --------------------------------------------------------------------------
PROCESSING_TIMEOUT_S = 120.0
# Optional borderline rule, OFF by default (spec 5.1).
BORDERLINE_REVIEW = False
BORDERLINE_BAND = 0.05
# Decision thresholds on dep_prob used by src/dashboard/app.py:
# BASELINE_DEPRESSION_THRESHOLD = 0.40, BERT_DEPRESSION_THRESHOLD = 0.50.
DASHBOARD_THRESHOLDS = (0.40, 0.50)

# --------------------------------------------------------------------------
# Link
# --------------------------------------------------------------------------
DEFAULT_URL = "http://127.0.0.1:5000/api/robot/state"
ALLOWED_HOSTS = ("127.0.0.1", "localhost")
POLL_HZ = 5.0
POLL_TIMEOUT_S = 0.5
OFFLINE_AFTER_FAILURES = 3
SCHEMA_VERSION = 1

# --------------------------------------------------------------------------
# OLED (SSD1306 128x64)
# --------------------------------------------------------------------------
OLED_W, OLED_H = 128, 64
OLED_MAX_CHARS = 21
OLED_EYE_W, OLED_EYE_H = 22, 26
OLED_EYE_CENTERS = ((40, 20), (88, 20))
OLED_GAZE_MAX_PX = 6
OLED_TEXT_Y = (42, 53)            # glyph rows land in 44..63
OLED_BLINK_S = 0.150
OLED_ON_RGB = (200, 235, 255)
OLED_OFF_RGB = (8, 10, 12)
OLED_DIM_FACTOR = 0.35
OLED_TEXTURE_SIZE = (256, 128)
OLED_CONSOLE_SCALE = 4

BLINK_INTERVAL_S = (4.0, 7.0)
FLAGGED_NO_BLINK_S = 3.0
CONFIRM_OVERLAY_S = 1.0
DISMISS_OVERLAY_S = 1.5
BUZZER_TILE_S = 1.5

# --------------------------------------------------------------------------
# Console palette (dashboard CSS tokens), hex -> RGB
# --------------------------------------------------------------------------
def hex_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


PALETTE = {
    "paper": hex_rgb("#F6F5F1"),
    "ink": hex_rgb("#1C2A2E"),
    "accent": hex_rgb("#2F6F63"),
    "amber": hex_rgb("#B08328"),
    "blocked": hex_rgb("#8B958F"),
    "low": hex_rgb("#4C7A5D"),
    "card": hex_rgb("#FFFFFF"),
    "line": hex_rgb("#DEDBD2"),
    "muted": hex_rgb("#6B7672"),
    "live": hex_rgb("#5FD3A8"),
}

CONSOLE_W, CONSOLE_H = 1280, 720
BANNER_H = 48
TRACE_SECONDS = 10.0
CAMERA_GLIDE_S = 0.6
DEMO_CYCLE_S = 5.0
