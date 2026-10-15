import importlib
import importlib.util
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REAL_APP = ROOT / "src" / "dashboard" / "app.py"


def _stub_bert():
    """Insert a stub for src.models.bert_inference so torch/BERT are never needed."""
    if "src.models.bert_inference" in sys.modules:
        return
    stub = types.ModuleType("src.models.bert_inference")

    def _unavailable(*a, **k):
        raise RuntimeError("BERT is stubbed out in robot tests")

    stub.__getattr__ = lambda name: _unavailable          # any imported name resolves
    sys.modules["src.models.bert_inference"] = stub


def _load_reference():
    path = Path(__file__).with_name("reference_dashboard.py")
    spec = importlib.util.spec_from_file_location("robot_reference_dashboard", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["robot_reference_dashboard"] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_real():
    _stub_bert()
    mod = importlib.import_module("src.dashboard.app")
    if not hasattr(mod, "robot_state"):
        pytest.fail("src/dashboard/app.py exists but lacks the robot edits (docs/ROBOT_SIMULATION.md §Integration)")
    return mod


@pytest.fixture(params=["real", "reference"])
def dashboard(request, monkeypatch):
    """(module, kind). The real dashboard is used whenever it exists; the
    reference stand-in only runs when the real app is absent."""
    if request.param == "real":
        if not REAL_APP.exists():
            pytest.skip("real dashboard not present here (src/dashboard/app.py)")
        mod = _load_real()
    else:
        if REAL_APP.exists():
            pytest.skip("real dashboard present; reference stand-in not needed")
        mod = _load_reference()
    monkeypatch.chdir(ROOT)          # the dashboard reads results/ relative to the repo root
    mod.app.config["TESTING"] = True
    mod.app.config["PROPAGATE_EXCEPTIONS"] = True
    return mod, request.param
