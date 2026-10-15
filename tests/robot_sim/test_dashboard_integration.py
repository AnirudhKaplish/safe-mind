"""Dashboard -> robot wiring, with torch/BERT stubbed out.

Uses the real src/dashboard/app.py when present, otherwise a reference
stand-in that reproduces the documented flow (see reference_dashboard.py).
"""
import threading
import time

import pytest

from robot_sim import controller as C
from robot_sim.bus import robot_bus


def state(client):
    r = client.get("/api/robot/state")
    assert r.status_code == 200
    assert "no-store" in r.headers.get("Cache-Control", "")
    data = r.get_json()
    assert C.validate_payload(data) is None, data
    return data


def submit(client, text="I have felt tired and alone for weeks", scenario="granted", pid="P-1"):
    return client.post("/analyze", data={"text": text, "patient_id": pid, "consent_scenario": scenario})


REPORTS = {
    "clear": ({"status": "AGREEMENT", "risk_class": "not_depression", "risk_level": "Low",
               "confidence": 0.81, "meter_pct": 19, "consent_status": "granted",
               "model_breakdown": [{"prediction": "not_depression", "confidence": .8, "dep_prob": .2, "lime_words": ["w"]}] * 2},
              "SCREEN_CLEAR"),
    "flagged": ({"status": "AGREEMENT", "risk_class": "depression", "risk_level": "High",
                 "confidence": 0.77, "meter_pct": 77, "consent_status": "granted",
                 "model_breakdown": [{"prediction": "depression", "confidence": .8, "dep_prob": .8, "lime_words": ["w"]}] * 2},
                "SCREEN_FLAGGED"),
    "disagree": ({"status": "DISAGREEMENT", "consent_status": "granted",
                  "model_breakdown": [{"prediction": "depression", "confidence": .7, "dep_prob": .7, "lime_words": []},
                                      {"prediction": "not_depression", "confidence": .6, "dep_prob": .4, "lime_words": []}]},
                 "REVIEW_NEEDED"),
    "not_applicable": ({"status": "NOT_APPLICABLE", "reason": "question", "category": "question",
                        "model_breakdown": [], "consent_status": "granted"}, "NOT_APPLICABLE"),
}


@pytest.mark.parametrize("case", sorted(REPORTS))
def test_report_status_maps_to_robot_state(dashboard, monkeypatch, case):
    mod, _ = dashboard
    report, expected = REPORTS[case]
    monkeypatch.setattr(mod, "build_report", lambda text, pid, path: dict(report, id=len(mod.REPORTS) + 1))
    c = mod.app.test_client()
    seq0 = state(c)["seq"]
    r = submit(c)
    assert r.status_code in (302, 303)
    s = state(c)
    assert s["state"] == expected
    assert s["seq"] == seq0 + 2            # PROCESSING, then the result


def test_exception_publishes_error_and_still_raises(dashboard, monkeypatch):
    mod, _ = dashboard

    def boom(*a):
        raise ValueError("analysis failed")

    monkeypatch.setattr(mod, "build_report", boom)
    c = mod.app.test_client()
    with pytest.raises(ValueError):
        submit(c)
    s = state(c)
    assert s["state"] == "ERROR"


def test_revoked_consent_blocks_and_models_never_run(dashboard, monkeypatch):
    mod, kind = dashboard
    calls = {"baseline": 0, "bert": 0}
    for name, key in (("baseline_predict", "baseline"), ("bert_predict", "bert")):
        if not hasattr(mod, name):
            pytest.skip(f"{name} not found in dashboard module")

        def counter(*a, _k=key, **k):
            calls[_k] += 1
            raise AssertionError("model called despite revoked consent")

        monkeypatch.setattr(mod, name, counter)
    c = mod.app.test_client()
    try:
        submit(c, text="I feel exhausted and empty every single day lately", scenario="revoked")
    except FileNotFoundError as e:  # pragma: no cover - real repo without consent logs
        pytest.skip(f"consent log missing locally: {e}")
    s = state(c)
    assert s["state"] == "CONSENT_BLOCKED", s["state"]
    assert calls == {"baseline": 0, "bert": 0}


def test_state_is_processing_while_build_report_runs(dashboard, monkeypatch):
    mod, _ = dashboard
    gate = threading.Event()

    def slow(*a):
        gate.wait(3)
        return dict(REPORTS["clear"][0], id=len(mod.REPORTS) + 1)

    monkeypatch.setattr(mod, "build_report", slow)
    c_post, c_get = mod.app.test_client(), mod.app.test_client()
    t = threading.Thread(target=lambda: submit(c_post))
    t.start()
    seen = None
    t0 = time.monotonic()
    while time.monotonic() - t0 < 2:
        seen = state(c_get)["state"]
        if seen == "PROCESSING":
            break
        time.sleep(0.01)
    assert seen == "PROCESSING"
    assert time.monotonic() - t0 < 0.5
    gate.set()
    t.join(5)
    assert state(c_get)["state"] == "SCREEN_CLEAR"


def test_robot_endpoint_is_read_only(dashboard):
    mod, _ = dashboard
    c = mod.app.test_client()
    seq = state(c)["seq"]
    for verb in (c.post, c.put, c.delete, c.patch):
        assert verb("/api/robot/state", json={"state": "SCREEN_CLEAR"}).status_code == 405
    assert state(c)["seq"] == seq


def test_bus_is_the_dashboard_singleton(dashboard):
    mod, _ = dashboard
    assert mod.robot_bus is robot_bus
