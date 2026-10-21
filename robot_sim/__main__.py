"""python -m robot_sim  — run the digital twin console.

Examples:
  python -m robot_sim                     # live: poll the local dashboard
  python -m robot_sim --manual            # manual demo, keys 1-9
  python -m robot_sim --demo-cycle        # manual + auto-cycle every 5 s
  python -m robot_sim --headless --seconds 30 --demo-cycle --record runs/robot/demo_cycle.mp4
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from . import config
from .link import NonLocalURL, check_local_url


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="python -m robot_sim", description="Virtual ESP32 companion (digital twin)")
    ap.add_argument("--live", action="store_true", default=True, help="poll the dashboard (default)")
    ap.add_argument("--manual", action="store_true", help="start in MANUAL mode (no polling until 'm')")
    ap.add_argument("--demo-cycle", action="store_true", help="MANUAL + auto-cycle states every 5 s")
    ap.add_argument("--url", default=config.DEFAULT_URL)
    ap.add_argument("--poll-hz", type=float, default=config.POLL_HZ)
    ap.add_argument("--gui", action="store_true", help="also open PyBullet's own OpenGL window")
    ap.add_argument("--headless", action="store_true", help="no windows; prints state transitions")
    ap.add_argument("--realtime", action="store_true",
                    help="with --headless: follow the wall clock (for recording a live session)")
    ap.add_argument("--seconds", type=float, default=None, help="quit after N seconds")
    ap.add_argument("--record", type=Path, default=None, help="record the console to PATH.mp4")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--sound", action="store_true", help="play buzzer tones (Windows only; stdlib winsound)")
    ap.add_argument("--no-3d", action="store_true", help="skip 3D rendering (very slow machines)")
    ap.add_argument("--render-worker", choices=("auto", "on", "off"), default="auto",
                    help="render the 3D view in a child process (auto: only if measured fps is low)")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    try:
        check_local_url(args.url)
    except NonLocalURL as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    manual = args.manual or args.demo_cycle

    from .console import Console, bgr
    from .fsm import Mode
    from .recorder import Recorder, screenshot, stamp
    from .runtime import TwinRuntime
    from .sim import QUALITY_LADDER, choose_settings, p

    headless = args.headless
    fixed_step = headless and not args.realtime   # deterministic virtual time when headless
    render_3d = not args.no_3d and p is not None
    soft_3d = not args.no_3d and p is None
    if soft_3d:
        print("[robot] pybullet not found; using the built-in lightweight 3D view.", file=sys.stderr)
    rt = TwinRuntime(live=True, manual=manual, url=args.url, poll_hz=args.poll_hz, seed=args.seed,
                     gui=args.gui, sound=args.sound, verbose=args.verbose or headless,
                     clock=None if fixed_step else time.monotonic, physics=render_3d)
    if args.sound and not rt.buzzer.sound:
        print("note: --sound needs winsound (Windows); tones are shown visually only.", file=sys.stderr)

    if fixed_step and render_3d:
        settings, use_worker = QUALITY_LADDER[0], False      # quality over speed when not real time
        print(f"[robot] headless: rendering {settings.label}")
    elif render_3d:
        settings, fps, tried = choose_settings(rt.sim)
        use_worker = args.render_worker == "on" or (args.render_worker == "auto" and fps < config.RENDER_FPS_TARGET)
        print(f"[robot] render probe {tried} -> {settings.label}{' (worker process)' if use_worker else ''}")
    else:
        settings, use_worker = QUALITY_LADDER[1], False
    console = Console(rt, settings=settings, use_worker=use_worker, render_3d=render_3d,
                      soft_3d=soft_3d)
    if args.demo_cycle:
        rt.start_demo_cycle()

    recorder = Recorder(args.record) if args.record else None
    window = "MHD companion - digital twin"
    cv2 = None
    if not headless:
        import cv2  # noqa: F811
        try:
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window, config.CONSOLE_W, config.CONSOLE_H)
            cv2.setMouseCallback(window, console.on_mouse)
        except cv2.error as e:
            print(f"error: cannot open a window ({e}). Use --headless, or install opencv-python "
                  f"(not -headless).", file=sys.stderr)
            return 3

    frame_dt = 1.0 / config.RENDER_FPS_TARGET
    t_start = rt.now()
    last = time.monotonic()
    rc = 0
    try:
        while True:
            if fixed_step:
                dt = frame_dt
            else:
                t = time.monotonic()
                dt, last = min(t - last, 0.1), t
            rt.step(dt)
            console.update_view()
            frame = bgr(console.compose())
            if recorder:
                recorder.add(frame, rt.now() - t_start if not fixed_step else None)
            if headless:
                if args.seconds is None:
                    args.seconds = 10.0
                if not fixed_step:
                    sleep = frame_dt - (time.monotonic() - last)
                    if sleep > 0:
                        time.sleep(sleep)
            else:
                cv2.imshow(window, frame)
                key = cv2.waitKeyEx(1)
                if key != -1 and handle_key(key & 0xFFFF, rt, console, frame):
                    break
                if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    break
                sleep = frame_dt - (time.monotonic() - last)
                if sleep > 0:
                    time.sleep(sleep)
            if args.seconds is not None and rt.now() - t_start >= args.seconds:
                break
        if headless:
            shot = screenshot(frame)
            print(f"[robot] screenshot {shot}")
    except KeyboardInterrupt:
        pass
    finally:
        csv = rt.fsm.write_csv(config.RUNS_DIR / f"transitions_{stamp()}.csv")
        if args.verbose or headless:
            print(f"[robot] transitions -> {csv}")
        if recorder:
            recorder.close()
            print(f"[robot] recording -> {recorder.path} ({recorder.frames} frames)")
        console.close()
        rt.close()
        if cv2 is not None:
            cv2.destroyAllWindows()
    return rc


def handle_key(key: int, rt, console, frame) -> bool:
    """Return True to quit."""
    from .fsm import Mode
    from .recorder import screenshot
    ch = chr(key) if 0 <= key < 256 else ""
    now = rt.now()
    if ch in ("q", "\x1b"):
        return True
    if ch == "m":
        mode = rt.toggle_mode()
        console.flash_hint("MANUAL mode" if mode is Mode.MANUAL else "LIVE mode")
    elif ch.isdigit() and ch != "0":
        if not rt.manual_key(int(ch) - 1):
            console.flash_hint("switch to manual (m) first")
    elif ch == "0":
        if not rt.start_demo_cycle():
            console.flash_hint("switch to manual (m) first")
        else:
            console.flash_hint("auto-cycling all states every 5 s")
    elif ch == "c":
        rt.confirm()
        console.flash_hint("acknowledged")
    elif ch == "x":
        if not rt.dismiss():
            console.flash_hint("nothing to dismiss")
    elif ch == "v":
        console.rig.next(now)
    elif ch == "b":
        console.rig.prev(now)
    elif ch == "r":
        console.rig.reset(now)
    elif ch == "s":
        path = screenshot(frame)
        console.flash_hint(f"saved {path.name}")
    elif ch == " ":
        rt.paused = not rt.paused
    elif ch == "d":
        console.details = not console.details
        console.flash_hint("technical details shown" if console.details else "technical details hidden")
    return False


if __name__ == "__main__":
    sys.exit(main())
