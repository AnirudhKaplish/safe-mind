"""Write runs/robot/oled_states.png: one OLED frame per state (documentation)."""
from __future__ import annotations

from robot_sim import config
from robot_sim.behaviours import Behaviour
from robot_sim.oled import contact_sheet, render
from robot_sim.states import SPECS, STATE_ORDER


def main() -> None:
    b = Behaviour(seed=7)
    frames = []
    for st in STATE_ORDER:
        b.enter(st, 0.0)
        p = b.oled(1.2, pan_deg=99.0)
        frames.append((SPECS[st].label + f"  ({st.value})", render(p), p.dim))
    config.RUNS_DIR.mkdir(parents=True, exist_ok=True)
    out = config.RUNS_DIR / "oled_states.png"
    contact_sheet(frames).save(out)
    print(out)


if __name__ == "__main__":
    main()
