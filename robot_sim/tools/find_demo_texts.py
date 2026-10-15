"""LOCAL-ONLY helper: which lines of edge_case_tests.txt give which status?

Runs the dashboard's real build_report over every non-empty line of
edge_case_tests.txt (consent granted) and prints ONLY line numbers and
statuses, so you can pick a DISAGREEMENT text for the demo. It never prints
or writes the texts themselves.

    python -m robot_sim.tools.find_demo_texts [--file PATH]
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

from robot_sim import config


def find_file(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    hits = [p for p in config.REPO_ROOT.rglob("edge_case_tests.txt") if ".venv" not in p.parts]
    if not hits:
        sys.exit("edge_case_tests.txt not found; pass --file PATH")
    return hits[0]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file")
    args = ap.parse_args(argv)
    path = find_file(args.file)
    sys.path.insert(0, str(config.REPO_ROOT))
    try:
        from src.dashboard import app as dash
    except Exception as e:
        sys.exit(f"could not import the dashboard (needs torch/transformers and results/bert_model/): {e!r}")
    granted = next((getattr(dash, n) for n in dir(dash)
                    if "consent" in n.lower() and "granted" in n.lower() and isinstance(getattr(dash, n), (str, Path))),
                   config.REPO_ROOT / "results" / "blockchain_consent_state_granted.json")
    # The status does not depend on the LIME explanations, which are slow
    # (hundreds of BERT passes per text); skip them for this scan only.
    if hasattr(dash, "explain_lime_baseline"):
        dash.explain_lime_baseline = lambda *a, **k: []
    if hasattr(dash, "explain_lime_bert"):
        dash.explain_lime_bert = lambda *a, **k: []
    by_status: dict[str, list[int]] = defaultdict(list)
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    for n, line in enumerate(lines, 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        try:
            status = dash.build_report(line.strip(), "demo", str(granted)).get("status", "?")
        except Exception as e:
            status = f"exception:{type(e).__name__}"
        by_status[status].append(n)
        print(f"line {n:4d}: {status}")
    print("\nsummary (line numbers only):")
    for st in sorted(by_status):
        print(f"  {st:<26} {len(by_status[st]):3d}  lines {by_status[st][:20]}")
    if not (by_status.get("DISAGREEMENT") or by_status.get("DISAGREEMENT_NEEDS_REVIEW")):
        print("  no disagreement line found; use MANUAL mode (key 7) for REVIEW_NEEDED in the demo.")
    print(f"\nsource: {path.relative_to(config.REPO_ROOT) if path.is_relative_to(config.REPO_ROOT) else path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
