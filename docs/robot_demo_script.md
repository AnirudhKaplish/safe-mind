# Robot digital twin — review demo script (about 8 minutes)

Before the review, once (type in terminal, from the repo root):

```bash
.venv/bin/python -m robot_sim.tools.find_demo_texts
```

It prints only line numbers and statuses for `edge_case_tests.txt`. Note one
line number that gives `DISAGREEMENT` for step 6. If none does, use MANUAL key
`7` in step 8 and say so honestly.

| # | You do | Robot shows | What to say |
|---|---|---|---|
| 1 | type in terminal: `./run_robot_demo.sh` (Windows: `run_robot_demo.bat`) | IDLE, soft white LED, slow tilt, `Ready`; Link card `OK` | Two processes: the dashboard, and a separate read-only twin of the ESP32 device. |
| 2 | Submit a neutral first-person post, consent **granted** (e.g. "I went for a long walk today and cooked dinner with my flatmates.") | PROCESSING (blue pulse, scanning) → SCREEN_CLEAR (green, one nod) | It switches the moment analysis starts and follows the report status, not the confidence. |
| 3 | Submit the first-person distress sample from `hardware/virtual_device.py` | SCREEN_FLAGGED: steady amber, attentive eyes, `Indicators flagged / Clinician to review` | Calm and steady, never flashing; no clinical words on the device. The dashboard holds the full report and explanations. |
| 4 | Submit "Does anyone else find it hard to sleep before exams?" | NOT_APPLICABLE: `Not screened / Not a personal post` | The input-type filter says this isn't a personal disclosure, so nothing is screened. |
| 5 | Switch consent to **revoked**, submit a personal post | CONSENT_BLOCKED: lock icon, `Consent not valid / Nothing processed` | The models never ran (a test asserts zero model calls). Consent is read from the consent log, not queried live on chain. |
| 6 | Consent back to granted; submit the DISAGREEMENT line from the helper | REVIEW_NEEDED: `?` between the eyes, amber pulse, small look-aside | The two models disagree, so a human reviews. |
| 7 | Stop the dashboard (Ctrl-C in its terminal, or close the launcher's dashboard), then start it again | OFFLINE within 1.5 s (head droops, LED off, `No connection`) → back to IDLE within 1 s | The robot notices the link is down; it never pretends a result. |
| 8 | Press `m`, then `1`…`9`, then `m` again | Striped MANUAL banner on every frame, each state in turn; back to LIVE with "Returned to LIVE" | Manual states are labelled "MANUAL DEMO — NOT A MODEL PREDICTION" and never look like model output. |
| 9 | type in terminal: `.venv/bin/python -m robot_sim.selftest` | PASS/WARN/FAIL table | Servo tracking, limits, clearance, link timing and privacy are all checked automatically. |

Tips: `v`/`b` cycle cameras (`face_closeup` shows the OLED on the head,
`servo_closeup` the pan servo); `s` saves a screenshot to `runs/robot/`;
`c` / `x` are the Confirm / Dismiss buttons.
