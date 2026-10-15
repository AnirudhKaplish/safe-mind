import math
import random

import pytest

from robot_sim import config
from robot_sim.servo import Quintic, Servo, min_jerk_duration, pulse_us

DT = 1.0 / config.SERVO_UPDATE_HZ


def run(servo, seconds, dt=DT):
    out = []
    for _ in range(int(round(seconds / dt))):
        prev = servo.pos
        servo.update(dt)
        out.append((servo.pos, servo.vel, servo.pos - prev))
    return out


def test_pwm_mapping():
    assert pulse_us(0) == pytest.approx(544)
    assert pulse_us(90) == pytest.approx(1472)
    assert pulse_us(180) == pytest.approx(2400)


def test_min_jerk_profile_matches_closed_form():
    q = Quintic.fit(0, 0, 0, 30, 2.0)
    for s in (0.1, 0.3, 0.5, 0.8):
        expected = 30 * (10 * s**3 - 15 * s**4 + 6 * s**5)
        assert q.eval(s * 2.0)[0] == pytest.approx(expected)
    assert q.extremes()[2] == pytest.approx(1.875 * 30 / 2.0, rel=1e-3)


@pytest.mark.parametrize("cap", [15, 25, 40, 600])
def test_peak_velocity_and_settling(cap):
    s = Servo("pan")
    s.set_target(140, cap)
    T = s.last_plan_T
    assert T == pytest.approx(min_jerk_duration(50, cap))
    trace = run(s, T + 0.2)
    peak = max(abs(v) for _, v, _ in trace)
    assert peak <= cap * 1.01
    assert abs(s.pos - 140) <= 1.0


def test_random_targets_limits_and_caps():
    rng = random.Random(1234)
    for name in ("pan", "tilt"):
        s = Servo(name)
        lo, hi = config.SOFT_LIMITS_DEG[name]
        cap = 40.0
        for i in range(10_000):
            if rng.random() < 0.05:
                s.set_target(rng.uniform(-30, 210), cap)
            prev_pos = s.pos
            s.update(DT)
            assert lo - 1e-6 <= s.pos <= hi + 1e-6
            assert abs(s.vel) <= cap * 1.01
            assert abs(s.pos - prev_pos) <= cap * DT * 1.01 + 1e-9


def test_replanning_no_jump_no_spike():
    rng = random.Random(7)
    s = Servo("pan")
    cap = 30.0
    s.set_target(140, cap)
    for k in range(400):
        if k % 17 == 5:
            before = s.pos
            s.set_target(rng.uniform(35, 145), cap)
            assert abs(s.pos - before) <= 0.1      # re-plan never moves the joint
        s.update(DT)
        assert abs(s.vel) <= cap * 1.01


def test_replan_continuity_of_velocity():
    s = Servo("pan")
    s.set_target(140, 40)
    run(s, 0.6)
    v_before = s.vel
    s.set_target(60, 40)                 # reverse mid-move
    s.update(1e-4)
    assert abs(s.vel - v_before) < 1.0   # velocity continuous through re-plan


def test_cap_reduction_while_moving():
    s = Servo("tilt")
    s.set_target(112, 40)
    run(s, 0.3)
    v0 = abs(s.vel)
    s.set_target(110, 15)
    trace = run(s, 5)
    assert max(abs(v) for _, v, _ in trace) <= max(15, v0) * 1.01


def test_deadband():
    s = Servo("pan")
    assert s.set_target(90.6, 40) is False
    assert not s.moving
    assert s.set_target(91.5, 40) is True


def test_clamp_soft_then_hw_logs_warning(caplog):
    s = Servo("tilt")
    with caplog.at_level("WARNING"):
        s.set_target(170, 40)
    assert s.goal == 115
    assert s.clamp_events == 1
    assert "clamped" in caplog.text
    hw = Servo("tilt", enforce_soft=False)
    hw.set_target(250, 40)
    assert hw.goal == 180


def test_explicit_duration_used():
    s = Servo("tilt")
    s.set_target(92, 40, duration=3.0)
    assert s.last_plan_T == pytest.approx(3.0)
