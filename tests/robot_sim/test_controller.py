import json
import math

import pytest

from robot_sim import controller as C
from robot_sim.oled import TextOverflow, check_lines
from robot_sim.states import FORBIDDEN_WORDS, SPECS, RobotState, all_device_strings

S = RobotState


def mb(p1=0.8, p2=0.7):
    return [
        {"prediction": "x", "confidence": 0.8, "dep_prob": p1, "lime_words": ["secretword"]},
        {"prediction": "x", "confidence": 0.7, "dep_prob": p2, "lime_words": ["otherword"]},
    ]


def agreement(rc, conf=0.82, **kw):
    r = {"status": "AGREEMENT", "risk_class": rc, "risk_level": "High", "confidence": conf,
         "meter_pct": 82, "model_breakdown": mb(), "consent_status": "granted"}
    r.update(kw)
    return r


VALID_CASES = [
    ({"status": "NOT_APPLICABLE", "reason": "quoted input text", "category": "question",
      "model_breakdown": [], "consent_status": "granted"}, S.NOT_APPLICABLE),
    ({"status": "BLOCKED", "consent_status": "revoked", "model_breakdown": []}, S.CONSENT_BLOCKED),
    (agreement("not_depression"), S.SCREEN_CLEAR),
    (agreement("depression"), S.SCREEN_FLAGGED),
    ({"status": "DISAGREEMENT", "model_breakdown": mb(0.8, 0.2), "consent_status": "granted"}, S.REVIEW_NEEDED),
    ({"status": "DISAGREEMENT_NEEDS_REVIEW", "model_breakdown": mb(0.8, 0.2)}, S.REVIEW_NEEDED),
]


@pytest.mark.parametrize("report,expected", VALID_CASES)
def test_mapping_table(report, expected):
    cmd = C.from_report(report, report_id=3)
    assert cmd.state is expected
    assert cmd.report["report_id"] == 3


MALFORMED = [
    None, [], [1, 2], "AGREEMENT", 42, {}, {"status": "WHATEVER"}, {"status": None},
    {"status": "AGREEMENT", "confidence": 0.9},                         # no risk_class
    agreement("anxiety"), agreement(None), agreement("depression", conf=1.3),
    agreement("depression", conf=-0.1), agreement("depression", conf=float("nan")),
    agreement("depression", conf="0.9"), agreement("not_depression", conf=None),
    agreement("not_depression", conf=True), agreement("not_depression", conf=float("inf")),
    {"status": "DISAGREEMENT", "confidence": 7},
]


@pytest.mark.parametrize("report", MALFORMED)
def test_malformed_is_error_never_clear(report):
    cmd = C.from_report(report)
    assert cmd.state is S.ERROR
    assert cmd.state not in (S.SCREEN_CLEAR, S.IDLE)


def test_confidence_never_selects_state():
    for conf in (0.0, 0.01, 0.5, 0.51, 0.99, 1.0):
        assert C.from_report(agreement("not_depression", conf=conf)).state is S.SCREEN_CLEAR
        assert C.from_report(agreement("depression", conf=conf)).state is S.SCREEN_FLAGGED


def test_borderline_rule_off_by_default():
    r = agreement("depression", model_breakdown=mb(0.52, 0.50))
    assert C.from_report(r).state is S.SCREEN_FLAGGED


def test_borderline_rule_on():
    r = agreement("depression", model_breakdown=mb(0.52, 0.50))
    cmd = C.from_report(r, borderline_review=True, band=0.05, thresholds=(0.5,))
    assert cmd.state is S.REVIEW_NEEDED and cmd.report["borderline_rule"] is True
    far = agreement("depression", model_breakdown=mb(0.9, 0.8))
    assert C.from_report(far, borderline_review=True).state is S.SCREEN_FLAGGED


def test_borderline_rule_ignores_bad_dep_prob():
    r = agreement("not_depression", model_breakdown=[{"dep_prob": "x"}])
    assert C.from_report(r, borderline_review=True).state is S.SCREEN_CLEAR


def test_forbidden_words_never_on_device():
    for s in all_device_strings():
        low = s.lower()
        for w in FORBIDDEN_WORDS:
            assert w not in low, (w, s)


def test_oled_lines_fit():
    for spec in SPECS.values():
        assert 1 <= len(spec.oled_lines) <= 2
        for ln in spec.oled_lines:
            assert len(ln) <= 21, ln
        check_lines(spec.oled_lines)
    with pytest.raises(TextOverflow):
        check_lines(("Clinician review advised",))


def test_payload_never_leaks_report_content():
    r = agreement("depression", patient_id="P-777", text="my secret words",
                  reason="my secret words", lime_words=["secretword"])
    cmd = C.from_report(r, report_id=1)
    p = C.to_payload(cmd, "abcd1234", 5)
    blob = json.dumps(p)
    for leak in ("P-777", "secret", "lime", "model_breakdown", "meter_pct", "risk_level", "High"):
        assert leak not in blob
    assert C.validate_payload(p) is None
    assert p["report"]["confidence"] == 0.82


def test_unsafe_consent_string_is_sanitised():
    r = {"status": "BLOCKED", "consent_status": "revoked because John Smith said so"}
    p = C.to_payload(C.from_report(r), "abcd1234", 1)
    assert p["report"]["consent"] == "unknown"


def test_payload_report_null_for_idle_processing():
    assert C.to_payload(C.idle_command(), "abcd1234", 0)["report"] is None
    assert C.to_payload(C.processing_command(), "abcd1234", 1)["report"] is None


def test_error_command_reason_vocabulary():
    assert C.error_command("analysis_exception").reason == "analysis_exception"
    assert C.error_command("Traceback: user text here").reason == "error"


@pytest.mark.parametrize("mut", [
    lambda p: p.update(schema_version=2),
    lambda p: p.update(seq=-1),
    lambda p: p.update(seq=True),
    lambda p: p.update(state="HAPPY"),
    lambda p: p.update(state="OFFLINE"),
    lambda p: p.update(boot_id="ZZ"),
    lambda p: p.update(text="leak"),
    lambda p: p["report"].update(lime_words=["x"]),
    lambda p: p["report"].update(confidence=3),
    lambda p: p.update(device=None),
])
def test_validate_payload_rejects(mut):
    p = C.to_payload(C.from_report(agreement("depression")), "abcd1234", 3)
    mut(p)
    assert C.validate_payload(p) is not None


def test_firmware_line():
    line = C.to_firmware_line(C.from_report({"status": "DISAGREEMENT"}))
    d = json.loads(line)
    assert d == {"state": "REVIEW_NEEDED", "pan": 90, "tilt": 90, "rgb": [230, 160, 40],
                 "oled1": "Models disagree", "oled2": "Review needed"}
    assert "\n" not in line


def test_controller_is_stdlib_only():
    import ast, pathlib, sys
    root = pathlib.Path(C.__file__).parent
    for name in ("controller.py", "states.py", "bus.py", "config.py", "__init__.py"):
        tree = ast.parse((root / name).read_text())
        for node in ast.walk(tree):
            mods = []
            if isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                mods = [node.module]
            for m in mods:
                top = m.split(".")[0]
                assert top in sys.stdlib_module_names or top == "__future__", (name, m)


def test_dashboard_consent_dict_is_reduced_to_status_word():
    # src/dashboard/app.py stores consent as a dict with the contract address
    r = {"status": "BLOCKED", "id": 3,
         "consent_status": {"status": "revoked", "contract_address": "0xABCDEF"}}
    cmd = C.from_report(r, report_id=r["id"])
    p = C.to_payload(cmd, "abcd1234", 2)
    assert cmd.state is S.CONSENT_BLOCKED and p["report"]["consent"] == "revoked"
    assert "0xABCDEF" not in json.dumps(p)
