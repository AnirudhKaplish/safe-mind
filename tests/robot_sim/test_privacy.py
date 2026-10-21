"""No input text, patient ID or explanation words may reach the robot."""
import json

import pytest

from robot_sim import config

SENTINEL_TEXT = "ZZ_SENTINEL_TEXT_ZZ"
SENTINEL_PID = "ZZ_SENTINEL_PID_ZZ"
SENTINEL_WORD = "ZZ_SENTINEL_LIME_ZZ"
SENTINELS = (SENTINEL_TEXT, SENTINEL_PID, SENTINEL_WORD)


_IDS = iter(range(1000, 2000))


def leaky_report(text, pid, path):
    # a report that (wrongly) carries the text everywhere it could
    return {"id": next(_IDS), "status": "AGREEMENT", "risk_class": "depression", "risk_level": "High",
            "confidence": 0.9, "meter_pct": 90,
            "consent_status": {"status": "granted", "contract_address": SENTINEL_PID},
            "text": text, "patient_id": pid, "reason": text, "category": text,
            "model_breakdown": [{"prediction": "depression", "confidence": .9, "dep_prob": .9,
                                 "lime_words": [SENTINEL_WORD, text]}] * 2}


def assert_clean(blob: str):
    for s in SENTINELS:
        assert s not in blob, s


def test_sentinels_never_reach_robot(dashboard, monkeypatch, tmp_path):
    mod, _ = dashboard
    monkeypatch.setattr(mod, "build_report", leaky_report)
    c = mod.app.test_client()
    c.post("/analyze", data={"text": f"I feel {SENTINEL_TEXT}", "patient_id": SENTINEL_PID,
                             "consent_scenario": "granted"})
    payload = c.get("/api/robot/state").get_json()
    assert payload["state"] == "SCREEN_FLAGGED"
    assert_clean(json.dumps(payload))

    # feed the payload through the robot side: FSM log, CSV and console text
    from robot_sim.console import Console
    from robot_sim.runtime import TwinRuntime
    rt = TwinRuntime(live=False, physics=False)
    rt.fsm.on_poll_success(payload, 0.01)
    for _ in range(10):
        rt.step(0.05)
    con = Console(rt, render_3d=False)
    con.compose()
    assert_clean(" ".join(con.texts))
    assert any("Please meet a therapist" in t for t in con.texts)
    csv = rt.fsm.write_csv(tmp_path / "transitions.csv").read_text()
    assert_clean(csv)
    assert_clean(json.dumps(list(rt.fsm.events)))
    rt.close()


def test_runs_dir_artifacts_clean():
    """Anything already written under runs/robot (CSVs) must be clean too."""
    if not config.RUNS_DIR.exists():
        pytest.skip("no runs yet")
    for p in config.RUNS_DIR.glob("*.csv"):
        assert_clean(p.read_text(errors="ignore"))
