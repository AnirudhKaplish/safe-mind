"""
virtual_device.py
==================
A text-only "virtual ESP32": prints what the companion device shows
(LED / OLED / servo pose / buzzer) for a report, without a board.

It reads the real ensemble report (src/report/generate_report_ensemble.py)
and maps it with robot_sim.controller.from_report() - the SAME mapping and
state table (robot_sim/states.py) the 3D digital twin uses - so the text
demo, the twin (python -m robot_sim) and a future serial bridge to the
ESP32 can never disagree.

Safety rules carried over from the twin:
  - anything missing, malformed or unknown becomes ERROR, never "normal";
  - no explanation words (LIME/SHAP) and no input text are shown on the
    device - those stay in the dashboard / report.

Usage (from the repo root; needs results/bert_model/ and the baseline
*.joblib artefacts, like the dashboard):
    python hardware/virtual_device.py
"""

import os
import sys
import time

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from robot_sim.controller import from_report, to_firmware_line  # noqa: E402
from robot_sim.states import SPECS  # noqa: E402

RESULTS_DIR = "results"


def render_virtual_device(report: dict) -> str:
    """Return the console rendering of the device for one report dict."""
    cmd = from_report(report)
    spec = SPECS[cmd.state]
    led = f"RGB{spec.led_rgb} {spec.led_pattern}" + (f" {spec.led_hz:g} Hz" if spec.led_hz else "")
    oled = " / ".join(spec.oled_lines)
    buzzer = spec.buzzer.describe() if spec.buzzer else "none"
    width = 76
    rows = [
        ("State", f"{cmd.state.value} ({spec.label})"),
        ("LED", led),
        ("OLED", oled),
        ("Servos", f"pan {spec.pan_deg:g} deg, tilt {spec.tilt_deg:g} deg - {spec.motion}"),
        ("Buzzer", buzzer),
    ]
    out = ["+" + "-" * width + "+", "|  VIRTUAL ESP32 DEVICE".ljust(width + 1) + "|", "+" + "-" * width + "+"]
    for k, v in rows:
        out.append(f"|  {k:<7}: {v}"[: width + 1].ljust(width + 1) + "|")
    out.append("+" + "-" * width + "+")
    out.append("serial line for the board: " + to_firmware_line(cmd))
    return "\n".join(out)


def feed_from_report(report: dict) -> None:
    print(render_virtual_device(report))


if __name__ == "__main__":
    from src.report.generate_report_ensemble import generate_assessment_report_ensemble

    granted = f"{RESULTS_DIR}/blockchain_consent_state_granted.json"
    revoked = f"{RESULTS_DIR}/blockchain_consent_demo_log.json"
    demo_posts = [
        ("Had a really nice weekend hiking with some friends, the weather was perfect for it.", granted),
        ("I don't even know why I'm posting this. Nothing feels worth doing anymore, "
         "I just lie in bed most of the day.", granted),
        ("Does anyone else find it hard to sleep before exams?", granted),
        ("I haven't left my room in three days and I don't see the point in getting up anymore.", revoked),
    ]
    for text, consent_path in demo_posts:
        report = generate_assessment_report_ensemble(text, consent_log_path=consent_path)
        print("\nINPUT POST:", text[:70] + ("..." if len(text) > 70 else ""))
        feed_from_report(report)
        time.sleep(0.3)
