import threading

from robot_sim import controller as C
from robot_sim.bus import RobotBus


def test_initial_state():
    b = RobotBus()
    s = b.snapshot()
    assert s["state"] == "IDLE" and s["seq"] == 0 and C.validate_payload(s) is None


def test_snapshot_is_deep_copy():
    b = RobotBus()
    b.publish(C.from_report({"status": "DISAGREEMENT"}))
    s = b.snapshot()
    s["report"]["status"] = "HACKED"
    s["device"]["oled"].append("x")
    assert b.snapshot()["report"]["status"] == "DISAGREEMENT"
    assert b.snapshot()["device"]["oled"] == ["Models disagree", "Review needed"]


def test_thread_safety_and_monotonic_seq():
    b = RobotBus()
    seen, errors = [], []

    def pub():
        for _ in range(500):
            b.publish(C.processing_command())

    def read():
        last = -1
        for _ in range(2000):
            s = b.snapshot()
            if C.validate_payload(s) is not None:
                errors.append("invalid")
            if s["seq"] < last:
                errors.append("non-monotonic")
            last = s["seq"]
        seen.append(last)

    ts = [threading.Thread(target=pub) for _ in range(8)] + [threading.Thread(target=read) for _ in range(4)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert not errors
    assert b.snapshot()["seq"] == 4000


def test_publish_rejects_non_command():
    import pytest
    with pytest.raises(TypeError):
        RobotBus().publish({"state": "SCREEN_CLEAR"})
