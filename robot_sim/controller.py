"""Map a dashboard report dict to a validated ``RobotCommand``.

Pure and standard-library only: the Flask dashboard imports this module.

Privacy: nothing from the report except a closed set of enumerated values
and one number (AGREEMENT confidence) ever leaves this module. The input
text, patient ID, ``reason``/``category`` strings, ``lime_words``,
``model_breakdown`` and ``meter_pct`` are never copied.

Fail safe: anything malformed, missing or unknown becomes ``ERROR``. Nothing
ever falls back to IDLE or SCREEN_CLEAR.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from . import config
from .states import RobotState, SPECS

DISAGREEMENT_STATUSES = frozenset({"DISAGREEMENT", "DISAGREEMENT_NEEDS_REVIEW"})
KNOWN_STATUSES = frozenset({"NOT_APPLICABLE", "BLOCKED", "AGREEMENT"}) | DISAGREEMENT_STATUSES

# Reason codes are a closed vocabulary; free text from a report never passes.
REASON_CODES = frozenset({
    "report", "analysis_exception", "invalid_report", "unknown_status",
    "missing_risk_class", "unknown_risk_class", "invalid_confidence", "borderline",
    "timeout", "invalid_payload", "processing", "startup", "error",
})

_SAFE_TOKEN = re.compile(r"^[a-z][a-z_]{0,23}$")
_BOOT_ID = re.compile(r"^[0-9a-f]{4,32}$")
REPORT_KEYS = frozenset({"report_id", "status", "models_agree", "confidence", "consent", "borderline_rule"})
PAYLOAD_KEYS = frozenset({"schema_version", "boot_id", "seq", "state", "source", "updated_at", "report", "device"})


@dataclass
class RobotCommand:
    state: RobotState
    source: str = "live"
    reason: str = "report"
    report: Optional[dict] = None          # sanitised summary only (REPORT_KEYS)

    def __post_init__(self) -> None:
        self.state = RobotState(self.state)
        if self.reason not in REASON_CODES:
            self.reason = "error"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _valid_unit(x: Any) -> bool:
    return _is_number(x) and math.isfinite(x) and 0.0 <= float(x) <= 1.0


def _safe_token(x: Any) -> str:
    if isinstance(x, str) and _SAFE_TOKEN.match(x.strip().lower() or "-"):
        return x.strip().lower()
    return "unknown"


def _consent_token(report: Any) -> str:
    """The dashboard stores consent as {"status": "granted"|"revoked"|"unknown",
    "contract_address": ...}; only the status word is ever passed on."""
    if not isinstance(report, dict):
        return "unknown"
    c = report.get("consent_status")
    if isinstance(c, dict):
        c = c.get("status")
    return _safe_token(c)


def _safe_report_id(x: Any) -> Optional[int]:
    if isinstance(x, int) and not isinstance(x, bool) and 0 <= x < 10**9:
        return x
    return None


def _summary(report: dict, report_id: Optional[int], status: str, *,
             agree: Optional[bool], confidence: Optional[float], borderline: bool) -> dict:
    return {
        "report_id": _safe_report_id(report_id),
        "status": status if status in KNOWN_STATUSES else "INVALID",
        "models_agree": agree,
        "confidence": None if confidence is None else round(float(confidence), 4),
        "consent": _consent_token(report),
        "borderline_rule": bool(borderline),
    }


def _mean_dep_prob(report: dict) -> Optional[float]:
    mb = report.get("model_breakdown")
    if not isinstance(mb, list) or not mb:
        return None
    vals = []
    for m in mb:
        if not isinstance(m, dict) or not _valid_unit(m.get("dep_prob")):
            return None
        vals.append(float(m["dep_prob"]))
    return sum(vals) / len(vals)


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------
def idle_command(reason: str = "startup") -> RobotCommand:
    return RobotCommand(RobotState.IDLE, reason=reason)


def processing_command() -> RobotCommand:
    return RobotCommand(RobotState.PROCESSING, reason="processing")


def error_command(reason: str = "error", report_id: Optional[int] = None) -> RobotCommand:
    rep = {"report_id": _safe_report_id(report_id), "status": "INVALID", "models_agree": None,
           "confidence": None, "consent": "unknown", "borderline_rule": False}
    return RobotCommand(RobotState.ERROR, reason=reason if reason in REASON_CODES else "error", report=rep)


def from_report(report: Any, report_id: Optional[int] = None, *,
                borderline_review: Optional[bool] = None,
                band: Optional[float] = None,
                thresholds: Optional[tuple] = None) -> RobotCommand:
    """Translate a ``build_report`` dict into a robot command (spec 5.1)."""
    borderline_review = config.BORDERLINE_REVIEW if borderline_review is None else borderline_review
    band = config.BORDERLINE_BAND if band is None else band
    thresholds = config.DASHBOARD_THRESHOLDS if thresholds is None else thresholds

    if not isinstance(report, dict):
        return error_command("invalid_report", report_id)
    status = report.get("status")
    if not isinstance(status, str) or status not in KNOWN_STATUSES:
        return error_command("unknown_status", report_id)

    def cmd(state: RobotState, reason: str = "report", *, agree=None, conf=None, borderline=False):
        return RobotCommand(state, reason=reason, report=_summary(
            report, report_id, status, agree=agree, confidence=conf, borderline=borderline))

    if status == "NOT_APPLICABLE":
        return cmd(RobotState.NOT_APPLICABLE)
    if status == "BLOCKED":
        return cmd(RobotState.CONSENT_BLOCKED)
    if status in DISAGREEMENT_STATUSES:
        c = report.get("confidence")
        if c is not None and not _valid_unit(c):
            return error_command("invalid_confidence", report_id)
        return cmd(RobotState.REVIEW_NEEDED, agree=False)

    # AGREEMENT
    if "risk_class" not in report:
        return error_command("missing_risk_class", report_id)
    risk_class = report.get("risk_class")
    if risk_class not in ("depression", "not_depression"):
        return error_command("unknown_risk_class", report_id)
    conf = report.get("confidence")
    if not _valid_unit(conf):
        return error_command("invalid_confidence", report_id)

    if borderline_review:
        mean_p = _mean_dep_prob(report)
        if mean_p is not None and any(abs(mean_p - t) <= band for t in thresholds):
            return cmd(RobotState.REVIEW_NEEDED, "borderline", agree=True, conf=conf, borderline=True)

    state = RobotState.SCREEN_FLAGGED if risk_class == "depression" else RobotState.SCREEN_CLEAR
    return cmd(state, agree=True, conf=conf)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def to_payload(cmd: RobotCommand, boot_id: str, seq: int, updated_at: Optional[str] = None) -> dict:
    """Build the schema_version 1 payload served at /api/robot/state."""
    report = None
    if cmd.state not in (RobotState.IDLE, RobotState.PROCESSING) and cmd.report is not None:
        report = {k: cmd.report.get(k) for k in sorted(REPORT_KEYS)}
    return {
        "schema_version": config.SCHEMA_VERSION,
        "boot_id": boot_id,
        "seq": int(seq),
        "state": cmd.state.value,
        "source": "live",
        "updated_at": updated_at or utc_now_iso(),
        "report": report,
        "device": SPECS[cmd.state].device_dict(),
    }


def validate_payload(obj: Any) -> Optional[str]:
    """Return None if ``obj`` is a valid payload, else a short error code."""
    if not isinstance(obj, dict):
        return "not_object"
    if set(obj) - PAYLOAD_KEYS:
        return "unexpected_keys"
    if obj.get("schema_version") != config.SCHEMA_VERSION or isinstance(obj.get("schema_version"), bool):
        return "schema_version"
    if not isinstance(obj.get("boot_id"), str) or not _BOOT_ID.match(obj["boot_id"]):
        return "boot_id"
    seq = obj.get("seq")
    if not isinstance(seq, int) or isinstance(seq, bool) or seq < 0:
        return "seq"
    try:
        state = RobotState(obj.get("state"))
    except ValueError:
        return "state"
    if state is RobotState.OFFLINE:          # only the robot decides OFFLINE
        return "state"
    if obj.get("source") != "live":
        return "source"
    if not isinstance(obj.get("updated_at"), str):
        return "updated_at"
    rep = obj.get("report")
    if state in (RobotState.IDLE, RobotState.PROCESSING):
        if rep is not None:
            return "report"
    elif rep is not None:
        if not isinstance(rep, dict) or set(rep) - REPORT_KEYS:
            return "report"
        if rep.get("confidence") is not None and not _valid_unit(rep.get("confidence")):
            return "report_confidence"
        if rep.get("report_id") is not None and _safe_report_id(rep.get("report_id")) is None:
            return "report_id"
    if not isinstance(obj.get("device"), dict):
        return "device"
    return None


def to_firmware_line(cmd: RobotCommand) -> str:
    """Compact one-line JSON for a future serial bridge to the real ESP32."""
    s = SPECS[cmd.state]
    lines = list(s.oled_lines) + ["", ""]
    return json.dumps({
        "state": cmd.state.value, "pan": int(s.pan_deg), "tilt": int(s.tilt_deg),
        "rgb": list(s.led_rgb), "oled1": lines[0], "oled2": lines[1],
    }, separators=(",", ":"))
