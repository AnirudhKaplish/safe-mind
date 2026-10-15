# Robot digital twin — Virtual ESP32 companion

A software-only digital twin of the project's ESP32 companion device
(`hardware/mhd_hardware.ino`): a desk-sized pan–tilt robot with a 128×64 OLED
face, an RGB LED, a buzzer and two buttons. It is driven automatically by the
reports that the Meridian dashboard (`src/dashboard/app.py`) produces.

> The robot shows **system states**, never a person's emotions. It has no
> mouth, no clinical words, and its manual demo mode is always labelled
> "MANUAL DEMO — NOT A MODEL PREDICTION". The dashboard's report and
> explanations remain the authoritative output. Screening support only — not
> a diagnosis.

Simulation approach inspired by a peer's PyBullet project; no code reused.

---

## 1. Run it

| Step | Where | Command |
|---|---|---|
| Install | type in terminal | `.venv/bin/pip install -r requirements-robot.txt` (Windows: `.venv\Scripts\pip install -r requirements-robot.txt`) |
| Manual demo (no models needed) | type in terminal | `.venv/bin/python -m robot_sim --manual` |
| Auto-cycle all states | type in terminal | `.venv/bin/python -m robot_sim --demo-cycle` |
| Dashboard + robot together | type in terminal | `./run_robot_demo.sh` (Windows: `run_robot_demo.bat`) |
| Robot only, dashboard already running | type in terminal | `.venv/bin/python -m robot_sim` |
| Tests (no torch/BERT needed) | type in terminal | `.venv/bin/python -m pytest tests/robot_sim -q` |
| Self-test | type in terminal | `.venv/bin/python -m robot_sim.selftest` |
| Record a 30 s demo video | type in terminal | `.venv/bin/python -m robot_sim --headless --demo-cycle --seconds 30 --record runs/robot/demo_cycle.mp4` |

Keys in the console: `m` mode · `1`–`9` states (manual) · `0` auto-cycle ·
`c` confirm · `x` dismiss · `v`/`b` camera · `r` reset view · `s` screenshot ·
`space` pause · `q`/`Esc` quit. Drag with the mouse over the 3D view to orbit.

CLI flags: `--live` (default) · `--manual` · `--demo-cycle` · `--url` (must be
127.0.0.1/localhost) · `--poll-hz 5` · `--gui` (PyBullet's own window) ·
`--headless` · `--seconds N` · `--record PATH.mp4` · `--seed 7` · `--verbose` ·
`--sound` · `--no-3d` · `--render-worker auto|on|off`.

---

## 2. Architecture

```
Clinician ─text─▶ Flask dashboard (src/dashboard/app.py)
                    │ robot_bus.publish(PROCESSING)
                    ▼
                  build_report()            ← existing, unchanged
                    │ report dict
                    ▼
                  robot_sim.controller.from_report(report)   ← stdlib only
                    │ RobotCommand (validated; no text, IDs or XAI words)
                    ▼
                  robot_sim.bus.robot_bus (thread-safe latest state, seq, boot_id)
                    ▼
                  GET /api/robot/state   (JSON, read-only, Cache-Control: no-store)
                    ▲ HTTP poll, 5 Hz, localhost only, GET only
┌───────────────────┴──────────────────────────────────────────────┐
│ python -m robot_sim  (separate process)                          │
│  link.py → fsm.py → behaviours.py → servo.py (min-jerk + SG90)   │
│                                   → oled.py (128×64 framebuffer) │
│                                   → led.py, buzzer.py            │
│  sim.py    : PyBullet DIRECT, URDF, motors, offscreen render     │
│  runtime.py: the loop (50 Hz servo, 240 Hz physics)              │
│  console.py: OpenCV window (3D view + cards), keys, recording    │
└──────────────────────────────────────────────────────────────────┘
```

Polling (not push) means the dashboard behaves the same with or without the
robot, the robot notices when the dashboard disappears (OFFLINE), and the
robot has **no write path**, so it cannot bypass consent.

| File | Role |
|---|---|
| `config.py` | every constant (servo, timings, colours, layout) |
| `states.py` | `RobotState` + per-state spec table — single source of truth |
| `controller.py` | report → `RobotCommand`; payload build/validate; firmware line (stdlib only) |
| `bus.py` | `robot_bus` singleton for the dashboard (stdlib only) |
| `servo.py` | SG90 model, min-jerk planner, PWM mapping, dead band |
| `fsm.py` | state machine, live/manual arbitration, link health, timeout, transition log |
| `behaviours.py` | per-state motion, LED pattern, OLED parameters, blink/gaze |
| `oled.py` | 1-bit framebuffer, icons, text, texture cache |
| `led.py`, `buzzer.py` | LED patterns; tone events (visual) |
| `link.py` | GET-only localhost poller, seq/boot_id gate |
| `urdf_gen.py`, `sim.py` | procedural URDF + PyBullet scene, cameras, render worker |
| `runtime.py`, `console.py`, `recorder.py`, `__main__.py` | loop, UI, recording, CLI |
| `selftest.py` | PASS/WARN/FAIL self-test (sections A–I) |
| `tools/` | `oled_sheet.py`, `states_grid.py`, `find_demo_texts.py` |

---

## 3. States

| # | State | Trigger | Pose (pan/tilt) | LED | OLED | Buzzer (visual) |
|---|---|---|---|---|---|---|
| 1 | IDLE | start-up, Dismiss | 90/90, tilt ±2° over 6 s | soft white steady | eyes, blink 4–7 s · `Ready` | – |
| 2 | PROCESSING | published just before `build_report` | pan sweep 75↔105°, 0.3 s dwell | blue pulse 1 Hz | eyes follow pan · `Analysing…` | – |
| 3 | NOT_APPLICABLE | `status == NOT_APPLICABLE` | 90/90 | dim white | `Not screened` / `Not a personal post` | short tick |
| 4 | CONSENT_BLOCKED | `status == BLOCKED` | 90/100 | grey | lock · `Consent not valid` / `Nothing processed` | two low ticks |
| 5 | SCREEN_CLEAR | AGREEMENT + `not_depression` | 90/90 + one nod 90→84→90 | green | `Screen complete` / `No indicators flagged` | soft chime |
| 6 | SCREEN_FLAGGED | AGREEMENT + `depression` | 90/86, still | amber **steady** | open eyes, no blink 3 s · `Indicators flagged` / `Clinician to review` | soft tone |
| 7 | REVIEW_NEEDED | `DISAGREEMENT` or `DISAGREEMENT_NEEDS_REVIEW` | pan 90→98→90 every 6 s | amber pulse 0.5 Hz | eyes + `?` · `Models disagree` / `Review needed` | two-tone |
| 8 | ERROR | exception, bad/unknown status, bad risk_class, bad confidence, non-dict, PROCESSING > 120 s, invalid payload | 90/90 | grey blink 1 Hz | `!` · `Result unavailable` | – |
| 9 | OFFLINE | 3 failed polls (robot side only) | droop to 90/110 at 15°/s | off | dimmed sleepy eyes · `No connection` | – |

Rules: confidence never selects a state, pose or expression (it only appears
as a number in the console's report card, for AGREEMENT). Malformed input is
always ERROR, never CLEAR/IDLE. Result states persist until the next event or
Dismiss. Speed caps per state: 40/25/40/30/40/30/30/40/15 °/s.

**Deviation from the build prompt:** the flagged OLED line "Clinician review
advised" is 24 characters and does not fit the 21-character line, so it reads
"Clinician to review". All OLED strings are checked by a unit test.

Optional **borderline rule** (`config.BORDERLINE_REVIEW`, off): AGREEMENT
reports whose mean `dep_prob` lies within `BORDERLINE_BAND` (±0.05) of
`DASHBOARD_THRESHOLDS` map to REVIEW_NEEDED; the console then says
"borderline rule active". Check `DASHBOARD_THRESHOLDS` against the dashboard
before enabling it.

Buttons: **Confirm** (`c`) shows a ✓ and "Acknowledged" for 1 s, logs locally.
**Dismiss** (`x`) returns to IDLE ("Dismissed" for 1.5 s), greys the report
card; it cannot hide PROCESSING or OFFLINE. Neither button touches the network.

Buzzer frequencies are placeholders shown visually (tick 2000 Hz/30 ms, low
ticks 400 Hz/40 ms, chime 880 Hz/200 ms, tone 660 Hz/250 ms, two-tone
600/800 Hz/150 ms). Align them with the `tone()` calls in `mhd_hardware.ino`
(edit `states.py`). `--sound` plays them only on Windows (stdlib `winsound`);
on macOS/Linux there is no stdlib tone generator, so no dependency was added
and the flag prints a notice.

---

## 4. Servo model (SG90)

Datasheet values (`config.py`): travel 0–180°, 0.1 s/60° at 4.8 V →
ω_max = 600 °/s = 10.47 rad/s, stall torque 1.8 kg·cm = 0.1765 N·m, PWM 50 Hz,
pulse = 544 + deg/180 × (2400 − 544) µs (Arduino Servo library), dead band 1°.
Soft limits: pan 30–150°, tilt 70–115°. Joint mapping: q = radians(deg − 90).

Rest-to-rest motion is minimum-jerk:

```
θ(t) = θ0 + Δ·(10τ³ − 15τ⁴ + 6τ⁵),   τ = t/T
T    = max(0.15 s, 1.875·|Δ| / v_cap)        (peak velocity = 1.875·Δ/T)
```

A new target mid-motion is re-planned from the current commanded position,
velocity and acceleration with a quintic that ends at rest; the planner picks
the shortest duration that keeps |v| ≤ cap, a bounded acceleration and the
soft limits. If no single quintic can (a reversal right next to a limit), it
brakes smoothly to rest and then does a rest-to-rest move. Targets are
clamped to the soft limits, then the hardware limits, with a WARN log.

Timing: physics 240 Hz, servo targets 50 Hz, render target 30 fps.
PyBullet actuation: `POSITION_CONTROL`, force 0.1765 N·m, maxVelocity
10.47 rad/s, **positionGain 0.3, velocityGain 1.0**, joint damping 0.0005.
Measured tracking error with these gains: ≤ 0.23° (self-test C).

---

## 5. 3D model (final dimensions, metres)

Generated by `urdf_gen.py` into `robot_sim/urdf/companion.urdf` (+X = front):

| Link | Geometry | Mass |
|---|---|---|
| base | box 0.120 × 0.100 × 0.045 on a 0.114 × 0.094 × 0.006 darker foot | 250 g |
| led_ring | cylinder r 0.012, h 0.004, front-top of base | 2 g |
| pan_servo | box 0.023 × 0.012 × 0.029 (SG90 blue) | 9 g |
| neck | box 0.030 × 0.030 × 0.050, child of **pan** (axis +Z) | 12 g |
| tilt_servo | SG90 box on the neck's side | 9 g |
| head | box 0.060 (X) × 0.090 (Y) × 0.070 (Z), child of **tilt** (axis +Y) | 45 g |
| bezel | box 0.004 × 0.060 × 0.036 | 4 g |
| face | OBJ quad 0.050 × 0.025 with UVs, 0.5 mm in front of the bezel | 1 g |

Joint limits (hardware): ±1.5708 rad, effort 0.1765, velocity 10.47.
Self-collision on (parent–child pairs excluded). Head–base clearance:
≥ 70 mm at the soft limits, 49 mm at the worst hardware corner.

The OLED texture is the framebuffer upscaled ×2 (256×128, pale cyan on
near-black), written to `robot_sim/assets/oled_cache/<hash>.png` only when the
frame changes and swapped onto the face with `changeVisualShape`.

Rendering: PyBullet `DIRECT` + `ER_TINY_RENDERER`. At start the console
measures fps and picks the best of 640×480 with shadows → 640×480 → 480×360
that reaches 20 fps; if it is below 30 fps the 3D view is rendered in a child
process (`--render-worker auto`) so the UI keeps its frame rate. A baked
contact shadow keeps the robot grounded when real-time shadows are off.
Headless recording uses virtual time and always renders with shadows.

---

## 6. Payload contract (`GET /api/robot/state`, schema_version 1)

```json
{
  "schema_version": 1, "boot_id": "a1b2c3d4", "seq": 17,
  "state": "REVIEW_NEEDED", "source": "live", "updated_at": "2026-10-15T17:02:11Z",
  "report": {"report_id": 4, "status": "DISAGREEMENT", "models_agree": false,
             "confidence": null, "consent": "granted", "borderline_rule": false},
  "device": {"pan_deg": 90, "tilt_deg": 90,
             "led": {"rgb": [230,160,40], "pattern": "pulse", "hz": 0.5},
             "oled": ["Models disagree", "Review needed"], "icon": "question",
             "buzzer": {"event": "two_tone", "freq_hz": [600,800], "ms": 150}}
}
```

- `report` is `null` for IDLE and PROCESSING. It never contains `text`,
  `patient_id`, `reason`/`category` strings, `lime_words`, `model_breakdown`,
  `meter_pct` or `risk_level`; `consent` is reduced to a short safe token.
- `device` comes from `states.py`, so robot and future firmware agree.
- The robot validates every field; anything invalid → ERROR ("invalid payload").
- Ordering: a payload is applied if its `boot_id` is new (dashboard restart)
  or `seq` is higher; duplicates and stale payloads are ignored and counted.
- `controller.to_firmware_line(cmd)` gives a one-line JSON for a future serial
  bridge, e.g.
  `{"state":"REVIEW_NEEDED","pan":90,"tilt":90,"rgb":[230,160,40],"oled1":"Models disagree","oled2":"Review needed"}`.

Link: GET every 200 ms, 0.5 s timeout, `urllib` with environment proxies
disabled, host must be 127.0.0.1 or localhost (checked at start-up). Three
consecutive failures → OFFLINE; first success afterwards applies the server's
current state. In MANUAL mode polling continues, live updates are buffered
(latest only) and the banner says "live update waiting"; returning to LIVE
applies it and flashes "Returned to LIVE — showing report #id".

---

## 7. Dashboard integration (the only `app.py` changes)

```python
from robot_sim.bus import robot_bus
from robot_sim.controller import from_report, processing_command, error_command

# in analyze(), after the empty-text redirect, before build_report(...)
robot_bus.publish(processing_command())
try:
    report = build_report(text, patient_id, consent_log_path)
except Exception:
    robot_bus.publish(error_command(reason="analysis_exception"))
    raise                      # existing behaviour (Flask error page) unchanged
# ... existing REPORTS.append(report) ...
robot_bus.publish(from_report(report, report_id=report.get("id")))

@app.route("/api/robot/state")          # GET only
def robot_state():
    resp = jsonify(robot_bus.snapshot())
    resp.headers["Cache-Control"] = "no-store"
    return resp
```

Arguments, consent path selection, the consent check and the redirect are
unchanged. Flask's dev server is threaded and the debug reloader serves from a
single child process, so the in-memory bus works (verified by the launcher run).

---

### Mapping to the firmware's original response table

`hardware/mhd_hardware.ino` was written for a 4-class model
(`normal` / `stress` / `anxiety` / `depression`). The trained models are
binary (`depression` / `not_depression`, Kaggle data), so:

| Model outcome (dashboard) | Firmware class | Twin state |
|---|---|---|
| both models: `not_depression` | `normal` | SCREEN_CLEAR |
| both models: `depression` | `depression` | SCREEN_FLAGGED |
| models disagree | – | REVIEW_NEEDED |
| not a personal disclosure | – | NOT_APPLICABLE |
| consent not granted | – | CONSENT_BLOCKED |
| never produced by the binary models | `stress`, `anxiety` | – |

The twin adds the states the firmware lacks (review, consent, not
applicable, error, offline) and drops the firmware's "Risk: High" and
"Key word: …" lines, which would put clinical wording and explanation
words on a device others can see. The firmware itself is unchanged.

## 8. What is real and what is simulated

| Real | Simulated |
|---|---|
| Model outputs and the dashboard's consent-gated reports | Servo physics (SG90 model in PyBullet) |
| The state mapping, payload contract and privacy filtering | OLED, LED, buzzer, buttons |
| Localhost link, timing, offline detection | The robot's body (procedural URDF) |

Consent is **log-based**: `get_consent_status()` reads
`results/blockchain_consent_state_granted.json` or
`results/blockchain_consent_demo_log.json` (`blockedConfirmed` → revoked). It
does not query the chain live.

---

## 9. Limitations and next steps

- The tiny renderer is CPU-bound; slow machines get a lower 3D frame rate
  (the console and the robot state stay real-time; see self-test H).
- Buzzer frequencies are placeholders until aligned with the `.ino`.
- `DASHBOARD_THRESHOLDS` (borderline rule) must be checked against the
  dashboard before the rule is enabled.
- **Future serial bridge to the real ESP32:** a small process that polls
  `/api/robot/state` like the twin does and writes
  `controller.to_firmware_line(cmd)` + `\n` to the board at 115200 baud; the
  sketch would parse `state`/`pan`/`tilt`/`rgb`/`oled1`/`oled2`. That needs a
  matching change in `hardware/mhd_hardware.ino`, which is out of scope here.
- `--sound` works on Windows only (stdlib), by design.

## 10. Platform notes

- **macOS (Apple Silicon):** `pip install pybullet` builds from source if no
  wheel matches; install Xcode Command Line Tools first (`xcode-select --install`).
  Use `opencv-python` (not `-headless`) for the window. The render worker uses
  `spawn`, which is the macOS default.
- **Windows:** use `.venv\Scripts\python`. If `pip install pybullet` needs a
  compiler, install "Microsoft C++ Build Tools", or use a Python 3.11 wheel.
- **Linux/CI:** `--headless` needs no display.
