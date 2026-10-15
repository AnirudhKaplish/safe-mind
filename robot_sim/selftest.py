"""Headless self-test: python -m robot_sim.selftest

Sections A-I print PASS / WARN / FAIL rows, a summary table, and write
runs/robot/selftest_<timestamp>.csv. Exit code 1 if any FAIL.
"""
from __future__ import annotations

import csv
import json
import math
import platform
import sys
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import config


@dataclass
class Row:
    section: str
    check: str
    result: str
    detail: str


ROWS: list[Row] = []


def rec(section, check, ok, detail="", warn=False):
    res = "PASS" if ok else ("WARN" if warn else "FAIL")
    ROWS.append(Row(section, check, res, detail))
    print(f"  [{res}] {section} {check}: {detail}", flush=True)


def guard(section):
    def deco(fn):
        def run(*a, **k):
            print(f"\n{section}", flush=True)
            try:
                return fn(*a, **k)
            except Exception as e:  # report, don't hide
                rec(section.split()[0], "section crashed", False, repr(e))
        return run
    return deco


@guard("A environment")
def sec_a():
    import cv2
    import numpy
    import PIL
    import pybullet
    v = sys.version_info
    rec("A", "python", v[:2] == (3, 11), f"{platform.python_version()} ({platform.system()} {platform.machine()})",
        warn=v[:2] >= (3, 10))
    rec("A", "pybullet", True, f"API {pybullet.getAPIVersion()}")
    rec("A", "opencv", True, cv2.__version__)
    rec("A", "numpy", True, numpy.__version__)
    rec("A", "pillow", True, PIL.__version__)
    has_gui = hasattr(cv2, "imshow") and "highgui" in cv2.getBuildInformation().lower()
    rec("A", "opencv window support", True, "yes" if has_gui else "headless build (use opencv-python for windows)")


@guard("B URDF")
def sec_b(sim):
    import pybullet as p
    rec("B", "loads", sim.robot >= 0, f"body id {sim.robot}")
    rev = sim.revolute
    rec("B", "2 revolute joints", sorted(rev) == ["pan", "tilt"], ", ".join(rev))
    for name in ("pan", "tilt"):
        info = p.getJointInfo(sim.robot, sim.joint[name], physicsClientId=sim.cid)
        lo, hi, eff, vel = info[8], info[9], info[10], info[11]
        ok = abs(lo + 1.5708) < 1e-3 and abs(hi - 1.5708) < 1e-3 and abs(eff - config.SG90_STALL_TORQUE_NM) < 1e-3 \
            and abs(vel - config.SG90_MAX_RAD_S) < 0.01
        rec("B", f"{name} limits", ok, f"[{lo:.4f}, {hi:.4f}] rad, effort {eff:.4f} N·m, vel {vel:.2f} rad/s")
    # PyBullet reports mass 0 for a fixed base, so check the URDF's <inertial> too.
    import xml.etree.ElementTree as ET
    root = ET.parse(config.URDF_DIR / "companion.urdf").getroot()
    urdf_m = {l.get("name"): float(l.find("inertial/mass").get("value")) for l in root.findall("link")}
    moving = [p.getDynamicsInfo(sim.robot, j, physicsClientId=sim.cid)[0]
              for j in range(p.getNumJoints(sim.robot, physicsClientId=sim.cid))]
    ok = all(m > 0 for m in urdf_m.values()) and all(m > 0 for m in moving)
    rec("B", "masses present", ok, f"{len(urdf_m)} links with <inertial>, total {sum(urdf_m.values()) * 1000:.0f} g "
        f"(base fixed to world)")


@guard("C servo dynamics")
def sec_c():
    from .runtime import TwinRuntime
    rt = TwinRuntime(live=False, manual=True)
    moves = [(150, 90), (30, 90), (90, 115), (90, 70), (120, 100), (90, 90)]
    reached = True
    for pan, tilt in moves:
        rt.pan.set_target(pan, 40)
        rt.tilt.set_target(tilt, 40)
        for _ in range(int(7.0 / 0.02)):
            rt.pan.update(0.02)
            rt.tilt.update(0.02)
            rt.sim.set_servo_targets(rt.pan.pos, rt.tilt.pos)
            rt.sim.step(5 if _ % 5 else 4)
            for name, srv in (("pan", rt.pan), ("tilt", rt.tilt)):
                act, _v = rt.sim.joint_deg(name)
                rt.max_track_err[name] = max(rt.max_track_err[name], abs(act - srv.pos))
                lo, hi = srv.limits
                if not (lo - 0.5 <= act <= hi + 0.5):
                    rt.limit_violations += 1
        for name, goal in (("pan", pan), ("tilt", tilt)):
            act, _v = rt.sim.joint_deg(name)
            reached &= abs(act - goal) < 1.0
    err = max(rt.max_track_err.values())
    rec("C", "tracking error < 2° (calm moves, 40°/s)", err < 2.0,
        f"max |actual-cmd| pan {rt.max_track_err['pan']:.3f}°, tilt {rt.max_track_err['tilt']:.3f}° "
        f"(Kp {config.MOTOR_POSITION_GAIN}, Kd {config.MOTOR_VELOCITY_GAIN})")
    rec("C", "reaches targets (±1°)", reached, f"{len(moves)} moves")
    rec("C", "no soft-limit violation", rt.limit_violations == 0, f"{rt.limit_violations} samples outside")
    # state tour through behaviours
    from .states import STATE_ORDER
    rt.max_track_err = {"pan": 0.0, "tilt": 0.0}
    for i, _st in enumerate(STATE_ORDER):
        rt.manual_key(i)
        for _ in range(int(4 / 0.02)):
            rt.step(0.02)
    err = max(rt.max_track_err.values())
    rec("C", "tracking during full state tour", err < 2.0, f"max error {err:.3f}°, violations {rt.limit_violations}")
    rt.close()


@guard("D self-collision clearance")
def sec_d(sim):
    worst = 1.0
    for pan in config.SOFT_LIMITS_DEG["pan"] + (90.0,):
        for tilt in config.SOFT_LIMITS_DEG["tilt"] + (90.0,):
            sim.reset_joints(pan, tilt)
            c = sim.clearance_head_base()
            worst = min(worst, c)
    rec("D", "head–base clearance at soft limits ≥ 5 mm", worst >= 0.005, f"min {worst * 1000:.1f} mm")
    worst_hw, where = 1.0, None
    for pan in (0.0, 90.0, 180.0):
        for tilt in (0.0, 180.0):
            sim.reset_joints(pan, tilt)
            c = sim.clearance_head_base()
            if c < worst_hw:
                worst_hw, where = c, (pan, tilt)
    rec("D", "clearance at hardware limits (info)", worst_hw >= 0.005,
        f"min {worst_hw * 1000:.1f} mm at pan/tilt {where}", warn=True)
    sim.reset_joints(90, 90)


@guard("E rendering per state")
def sec_e():
    import numpy as np
    from .oled import render
    from .runtime import TwinRuntime
    from .sim import CAMERA_PRESETS
    from .states import STATE_ORDER
    rt = TwinRuntime(live=False, manual=True)
    cam = CAMERA_PRESETS[0]
    empty = None
    hashes = set()
    for i, st in enumerate(STATE_ORDER):
        rt.manual_key(i)
        for _ in range(60):
            rt.step(0.02)
        fb = render(rt.oled_params)
        img = rt.sim.render(cam.target, cam.distance, cam.yaw, cam.pitch, 320, 240, False)
        ok3d = img.std() > 10
        hashes.add(hash(img.tobytes()))
        rec("E", f"{st.value}", fb.shape == (64, 128) and fb.any() and ok3d,
            f"OLED {int(fb.sum())} px lit · 3D std {img.std():.1f}")
    rec("E", "3D frames differ between states", len(hashes) >= 6, f"{len(hashes)} distinct of {len(STATE_ORDER)}")
    rt.close()


@guard("F controller mapping")
def sec_f():
    from .controller import from_report
    from .states import RobotState as S
    table = [
        ({"status": "NOT_APPLICABLE"}, S.NOT_APPLICABLE),
        ({"status": "BLOCKED"}, S.CONSENT_BLOCKED),
        ({"status": "AGREEMENT", "risk_class": "not_depression", "confidence": .8}, S.SCREEN_CLEAR),
        ({"status": "AGREEMENT", "risk_class": "depression", "confidence": .8}, S.SCREEN_FLAGGED),
        ({"status": "DISAGREEMENT"}, S.REVIEW_NEEDED),
        ({"status": "DISAGREEMENT_NEEDS_REVIEW"}, S.REVIEW_NEEDED),
        ({}, S.ERROR), (None, S.ERROR), ({"status": "AGREEMENT", "confidence": .8}, S.ERROR),
        ({"status": "AGREEMENT", "risk_class": "depression", "confidence": float("nan")}, S.ERROR),
    ]
    bad = [(r, e, from_report(r).state) for r, e in table if from_report(r).state is not e]
    rec("F", "mapping table", not bad, f"{len(table)} rows" + (f", mismatches {bad}" if bad else ""))


class _Fake:
    def __init__(self):
        from .controller import idle_command, to_payload
        self.payload = to_payload(idle_command(), "cafe0001", 0)
        self.fail = False
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                if outer.fail:
                    self.send_response(500); self.end_headers(); return
                b = json.dumps(outer.payload).encode()
                self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers()
                self.wfile.write(b)

        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.srv.daemon_threads = True
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.srv.server_address[1]}/api/robot/state"


def _pump(fsm, poller, cond, limit):
    t0 = time.monotonic()
    while time.monotonic() - t0 < limit:
        for r in poller.drain():
            fsm.on_poll_success(r.payload, r.latency) if r.ok else fsm.on_poll_failure(r.error)
        if cond():
            return time.monotonic() - t0
        time.sleep(0.005)
    return None


@guard("G link vs local fake server")
def sec_g():
    from .controller import from_report, processing_command, to_payload
    from .fsm import RobotFSM
    from .link import LinkPoller, NonLocalURL, check_local_url
    from .states import RobotState as S
    fake = _Fake()
    fsm = RobotFSM()
    poller = LinkPoller(fake.url, 5)
    poller.start()
    try:
        t = _pump(fsm, poller, lambda: fsm.link_ok, 2)
        rec("G", "connects", t is not None, f"{t:.2f} s" if t is not None else "no answer")
        fake.payload = to_payload(processing_command(), "cafe0001", 1)
        t = _pump(fsm, poller, lambda: fsm.state is S.PROCESSING, 2)
        rec("G", "PROCESSING applied ≤ 0.5 s", t is not None and t <= 0.5, f"{t:.3f} s" if t is not None else "-")
        fake.payload = to_payload(from_report({"status": "DISAGREEMENT"}, 3), "cafe0001", 2)
        t = _pump(fsm, poller, lambda: fsm.state is S.REVIEW_NEEDED, 2)
        rec("G", "result applied ≤ 0.5 s", t is not None and t <= 0.5, f"{t:.3f} s" if t is not None else "-")
        fake.payload = to_payload(processing_command(), "cafe0001", 1)
        _pump(fsm, poller, lambda: False, 0.5)
        rec("G", "stale seq ignored", fsm.state is S.REVIEW_NEEDED, f"out-of-order {fsm.gate.out_of_order}")
        fake.srv.shutdown(); fake.srv.server_close()
        t = _pump(fsm, poller, lambda: fsm.state is S.OFFLINE, 3)
        rec("G", "dashboard stopped -> OFFLINE ≤ 1.5 s", t is not None and t <= 1.5, f"{t:.3f} s" if t else "-")
    finally:
        poller.stop()
    # restart on the same port with a new boot id
    port = int(fake.url.split(":")[2].split("/")[0])
    fake2 = _Fake.__new__(_Fake)
    from .controller import idle_command
    fake2.payload = to_payload(idle_command(), "cafe0002", 0)
    fake2.fail = False

    class H2(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            b = json.dumps(fake2.payload).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers()
            self.wfile.write(b)

    poller = LinkPoller(fake.url, 5)
    poller.start()
    _pump(fsm, poller, lambda: False, 0.7)
    srv = ThreadingHTTPServer(("127.0.0.1", port), H2)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        t = _pump(fsm, poller, lambda: fsm.state is S.IDLE, 3)
        rec("G", "restart -> server state (IDLE) ≤ 1 s", t is not None and t <= 1.0, f"{t:.3f} s" if t else "-")
    finally:
        poller.stop(); srv.shutdown(); srv.server_close()
    try:
        check_local_url("http://example.com/api/robot/state")
        rec("G", "non-local URL refused", False, "accepted!")
    except NonLocalURL:
        rec("G", "non-local URL refused", True, "example.com rejected")


@guard("H render performance")
def sec_h(sim):
    from .sim import RenderSettings, measure_fps
    best = 0.0
    details = []
    for st in (RenderSettings(640, 480, True), RenderSettings(640, 480, False)):
        fps = measure_fps(sim, st, frames=8)
        details.append(f"{st.label}: {fps:.1f} fps")
        best = max(best, fps)
    rec("H", "3D view fps at 640×480", best >= 20, " · ".join(details), warn=best >= 10)
    if best < 20:
        lo = measure_fps(sim, RenderSettings(480, 360, False), frames=8)
        print(f"         fallback 480x360: {lo:.1f} fps; the console also renders in a worker process "
              f"(--render-worker auto) so the UI keeps its frame rate.")


@guard("I privacy")
def sec_i():
    import tempfile
    from pathlib import Path
    from .console import Console
    from .controller import from_report, to_payload
    from .runtime import TwinRuntime
    sentinels = ("ZZ_SENTINEL_TEXT_ZZ", "ZZ_SENTINEL_PID_ZZ", "ZZ_SENTINEL_LIME_ZZ")
    report = {"status": "AGREEMENT", "risk_class": "depression", "confidence": .9, "text": sentinels[0],
              "patient_id": sentinels[1], "reason": sentinels[0], "consent_status": "granted",
              "model_breakdown": [{"dep_prob": .9, "lime_words": [sentinels[2]]}] * 2}
    payload = to_payload(from_report(report, 1), "beef0001", 1)
    blob = json.dumps(payload)
    rec("I", "payload free of text/ID/XAI words", not any(s in blob for s in sentinels), f"{len(blob)} bytes")
    rt = TwinRuntime(live=False, physics=False)
    rt.fsm.on_poll_success(payload, 0.0)
    rt.step(0.1)
    con = Console(rt, render_3d=False)
    con.compose()
    txt = " ".join(con.texts)
    rec("I", "console text clean", not any(s in txt for s in sentinels), f"{len(con.texts)} strings drawn")
    with tempfile.TemporaryDirectory() as d:
        csvtext = rt.fsm.write_csv(Path(d) / "t.csv").read_text()
    rec("I", "transition log clean", not any(s in csvtext for s in sentinels), f"{len(rt.fsm.log)} transitions")
    from .link import check_local_url  # noqa: F401  (localhost-only, see G)
    rt.close()


def main() -> int:
    print("robot_sim self-test (headless)")
    t0 = time.time()
    sec_a()
    from .sim import TwinSim
    sim = TwinSim()
    try:
        sec_b(sim)
        sec_c()
        sec_d(sim)
        sec_e()
        sec_f()
        sec_g()
        sec_h(sim)
        sec_i()
    finally:
        sim.close()
    counts = {k: sum(r.result == k for r in ROWS) for k in ("PASS", "WARN", "FAIL")}
    print("\nSummary")
    print(f"  {'section':<8}{'PASS':>6}{'WARN':>6}{'FAIL':>6}")
    for sec in "ABCDEFGHI":
        rs = [r for r in ROWS if r.section == sec]
        print(f"  {sec:<8}{sum(r.result == 'PASS' for r in rs):>6}{sum(r.result == 'WARN' for r in rs):>6}"
              f"{sum(r.result == 'FAIL' for r in rs):>6}")
    print(f"  {'total':<8}{counts['PASS']:>6}{counts['WARN']:>6}{counts['FAIL']:>6}   ({time.time() - t0:.1f} s)")
    from .recorder import stamp
    config.RUNS_DIR.mkdir(parents=True, exist_ok=True)
    out = config.RUNS_DIR / f"selftest_{stamp()}.csv"
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["section", "check", "result", "detail"])
        for r in ROWS:
            w.writerow([r.section, r.check, r.result, r.detail])
    print(f"  written {out}")
    return 1 if counts["FAIL"] else 0


if __name__ == "__main__":
    sys.exit(main())
