"""
virtual_device.py
==================
A "virtual ESP32" - mimics exactly what mhd_hardware.ino would show
on the OLED / LED / servo / buzzer, printed to the console instead.

Use this to demo the hardware RESPONSE LOGIC before the physical
board is fully wired up, or as a fallback if you don't have the
board in the room during the review. It consumes the SAME JSON
payload shape the real firmware expects over Serial/MQTT, so
swapping this script for the real ESP32 later is a drop-in change -
the payload format doesn't change, only where it's sent.

Usage:
    python virtual_device.py
    (then paste report JSON, or run generate_report.py and pipe its
    risk_class/confidence/top word into this automatically - see
    `feed_from_report()` below)
"""

import json
import time

RESPONSE_TABLE = {
    "normal":     {"led": "GREEN (steady)",        "oled": "Normal / Low - all clear",       "servo": "idle position",      "audio": "soft chime"},
    "stress":     {"led": "AMBER (slow pulse)",     "oled": "Stress - guarded state",         "servo": "turns toward user",  "audio": "two-tone cue"},
    "anxiety":    {"led": "ORANGE (breathing pulse)", "oled": "Anxiety - elevated",            "servo": "slow paced movement","audio": "paced-breathing tone"},
    "depression": {"led": "DEEP BLUE (slow fade)",  "oled": "High risk + key indicators",     "servo": "minimal / settles",  "audio": "supportive tone"},
}


def render_virtual_device(risk_level: str, confidence: float, top_word: str = "-"):
    profile = RESPONSE_TABLE.get(risk_level, RESPONSE_TABLE["normal"])
    print("\n" + "┌" + "─" * 40 + "┐")
    print("│  VIRTUAL ESP32 DEVICE" + " " * 17 + "│")
    print("├" + "─" * 40 + "┤")
    print(f"│  LED    : {profile['led']:<28}│")
    print(f"│  OLED   : {profile['oled']:<28}│")
    print(f"│  Servo  : {profile['servo']:<28}│")
    print(f"│  Audio  : {profile['audio']:<28}│")
    print("├" + "─" * 40 + "┤")
    print(f"│  risk_level = {risk_level:<10} conf={confidence:.2f}     │")
    print(f"│  top contributing word: {top_word:<14}│")
    print("└" + "─" * 40 + "┘")


def feed_from_report(report: dict):
    """Takes the dict produced by generate_report.generate_assessment_report()
    and drives the virtual device exactly as the real firmware would."""
    if report.get("status") == "BLOCKED":
        print("\n[VIRTUAL DEVICE] No display update - processing was BLOCKED (consent not valid).")
        print(f"  Reason: {report.get('reason')}")
        return
    risk_class = report["risk_class"]
    # The real (binary) dataset uses not_depression/depression - map onto
    # the existing 4-state hardware response table (normal/depression
    # states cover this; stress/anxiety states remain defined for when
    # a multi-class dataset is used instead).
    risk_class = {"not_depression": "normal"}.get(risk_class, risk_class)
    confidence = report["confidence"]
    top_word = report["evidence"]["lime_top_words"][0][0] if report["evidence"]["lime_top_words"] else "-"
    render_virtual_device(risk_class, confidence, top_word)


if __name__ == "__main__":
    import sys, os
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from src.report.generate_report import generate_assessment_report

    demo_posts = [
        ("Had a really nice weekend hiking with some friends, the weather was perfect for it.",
         "results/blockchain_consent_state_granted.json"),
        ("My heart has been racing all day and I can't shake this feeling that something bad is about to happen.",
         "results/blockchain_consent_state_granted.json"),
        ("I don't even know why I'm posting this. Nothing feels worth doing anymore, I just lie in bed most of the day.",
         "results/blockchain_consent_state_granted.json"),
    ]

    for text, consent_path in demo_posts:
        report = generate_assessment_report(text, consent_log_path=consent_path)
        print("\nINPUT POST:", text[:70], "...")
        feed_from_report(report)
        time.sleep(0.3)
