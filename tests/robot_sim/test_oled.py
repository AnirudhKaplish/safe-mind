import numpy as np

from robot_sim.behaviours import Behaviour
from robot_sim.oled import BLINK_FRAMES, OledParams, TextureCache, render
from robot_sim.states import SPECS, STATE_ORDER


def test_every_state_renders_128x64_bool():
    b = Behaviour(seed=1)
    for st in STATE_ORDER:
        b.enter(st, 0.0)
        fb = render(b.oled(0.5))
        assert fb.shape == (64, 128) and fb.dtype == bool and fb.any()


def test_states_are_visually_distinct():
    b = Behaviour(seed=1)
    frames = []
    for st in STATE_ORDER:
        b.enter(st, 0.0)
        frames.append(render(b.oled(0.5)).tobytes())
    assert len(set(frames)) == len(frames)


def test_blink_frames():
    assert BLINK_FRAMES[0] == 1.0 and 0.0 in BLINK_FRAMES and BLINK_FRAMES[-1] == 1.0
    heights = []
    for o in (1.0, 0.5, 0.0):
        fb = render(OledParams(openness=o))
        heights.append(fb[:40].any(axis=1).sum())
    assert heights[0] > heights[1] > heights[2] > 0


def test_blink_happens_and_flagged_holds_eyes_open():
    b = Behaviour(seed=3)
    b.enter(SPECS.__iter__().__next__(), 0.0)
    seen = {b.oled(t / 100).openness for t in range(0, 900)}
    assert 0.0 in seen
    from robot_sim.states import RobotState
    b.enter(RobotState.SCREEN_FLAGGED, 0.0)
    early = {b.oled(t / 100).openness for t in range(0, 300)}
    assert early == {1.0}


def test_no_mouth_region_below_eyes_except_text():
    # rows 40..41 are always blank: nothing between eyes and text (no mouth)
    b = Behaviour(seed=1)
    for st in STATE_ORDER:
        b.enter(st, 0.0)
        assert not render(b.oled(0.2))[40:42].any()


def test_gaze_follows_pan():
    from robot_sim.states import RobotState
    b = Behaviour()
    b.enter(RobotState.PROCESSING, 0.0)
    left = render(b.oled(0.1, pan_deg=75))
    right = render(b.oled(0.1, pan_deg=105))
    cols_l = np.where(left[:40].any(axis=0))[0]
    cols_r = np.where(right[:40].any(axis=0))[0]
    assert cols_r.min() - cols_l.min() == 12


def test_texture_cache_writes_only_on_change(tmp_path):
    c = TextureCache(tmp_path)
    fb = render(OledParams(lines=("Ready",)))
    h1, p1, ch1 = c.update(fb)
    h2, p2, ch2 = c.update(fb.copy())
    assert ch1 and not ch2 and c.writes == 1 and p1 == p2 and p1.exists()
    fb2 = render(OledParams(openness=0.0, lines=("Ready",)))
    _, _, ch3 = c.update(fb2)
    assert ch3 and c.writes == 2
    c.update(fb)                  # back to first frame: already on disk
    assert c.writes == 2
