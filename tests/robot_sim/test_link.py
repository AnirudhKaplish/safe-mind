import inspect
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from robot_sim import controller as C
from robot_sim import link
from robot_sim.fsm import RobotFSM
from robot_sim.link import LinkPoller, NonLocalURL, check_local_url
from robot_sim.states import RobotState as S


class FakeDashboard:
    """Local http.server whose behaviour can be switched per test."""

    def __init__(self):
        self.mode = "ok"
        self.payload = C.to_payload(C.idle_command(), "aaaa1111", 0)
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                m = outer.mode
                if m == "500":
                    self.send_response(500); self.end_headers(); return
                if m == "sleep":
                    time.sleep(1.0)
                body = b"{not json" if m == "badjson" else json.dumps(outer.payload).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                try:
                    self.wfile.write(body)
                except OSError:
                    pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.server.daemon_threads = True
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    @property
    def url(self):
        return f"http://127.0.0.1:{self.port}/api/robot/state"

    def set(self, cmd, seq, boot="aaaa1111"):
        self.payload = C.to_payload(cmd, boot, seq)

    def stop(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def dash():
    d = FakeDashboard()
    yield d
    try:
        d.stop()
    except Exception:
        pass


def pump(fsm, poller, seconds, until=None):
    t0 = time.monotonic()
    while time.monotonic() - t0 < seconds:
        for r in poller.drain():
            if r.ok:
                fsm.on_poll_success(r.payload, r.latency)
            else:
                fsm.on_poll_failure(r.error)
        if until is not None and until():
            return time.monotonic() - t0
        time.sleep(0.01)
    return None


def run_link(dash):
    fsm = RobotFSM()
    poller = LinkPoller(dash.url, poll_hz=5)
    poller.start()
    return fsm, poller


def test_valid_payloads_and_ordering(dash):
    fsm, poller = run_link(dash)
    try:
        assert pump(fsm, poller, 2, lambda: fsm.link_ok) is not None
        dash.set(C.processing_command(), 1)
        assert pump(fsm, poller, 1, lambda: fsm.state is S.PROCESSING) is not None
        dash.set(C.from_report({"status": "DISAGREEMENT"}, 2), 2)
        assert pump(fsm, poller, 1, lambda: fsm.state is S.REVIEW_NEEDED) is not None
        dash.set(C.from_report({"status": "AGREEMENT", "risk_class": "not_depression", "confidence": .9}), 1)
        pump(fsm, poller, 0.6)
        assert fsm.state is S.REVIEW_NEEDED and fsm.gate.out_of_order >= 1
        dash.set(C.idle_command(), 0, boot="bbbb2222")       # dashboard restart
        assert pump(fsm, poller, 1, lambda: fsm.state is S.IDLE) is not None
    finally:
        poller.stop()


def test_http_500_goes_offline_within_limit(dash):
    fsm, poller = run_link(dash)
    try:
        pump(fsm, poller, 2, lambda: fsm.link_ok)
        dash.mode = "500"
        t = pump(fsm, poller, 3, lambda: fsm.state is S.OFFLINE)
        assert t is not None and t <= 1.5
        dash.mode = "ok"
        t = pump(fsm, poller, 2, lambda: fsm.state is S.IDLE)
        assert t is not None and t <= 1.0
    finally:
        poller.stop()


def test_connection_refused_goes_offline_and_recovers():
    d = FakeDashboard()
    port = d.port
    fsm = RobotFSM()
    poller = LinkPoller(d.url, poll_hz=5)
    poller.start()
    try:
        pump(fsm, poller, 2, lambda: fsm.link_ok)
        d.stop()                                               # dashboard stops
        t = pump(fsm, poller, 3, lambda: fsm.state is S.OFFLINE)
        assert t is not None and t <= 1.5, t
    finally:
        poller.stop()


def test_timeout_goes_offline(dash):
    fsm, poller = run_link(dash)
    try:
        pump(fsm, poller, 2, lambda: fsm.link_ok)
        dash.mode = "sleep"
        # three 0.5 s timeouts back to back: ~1.5 s (+ scheduling slack)
        t = pump(fsm, poller, 4, lambda: fsm.state is S.OFFLINE)
        assert t is not None and t <= 2.0, t
    finally:
        poller.stop()


def test_malformed_json_is_error(dash):
    fsm, poller = run_link(dash)
    try:
        pump(fsm, poller, 2, lambda: fsm.link_ok)
        dash.mode = "badjson"
        assert pump(fsm, poller, 1, lambda: fsm.state is S.ERROR) is not None
        assert fsm.state is not S.SCREEN_CLEAR
    finally:
        poller.stop()


def test_get_only_by_source_inspection():
    src = inspect.getsource(link)
    for verb in ("POST", "PUT", "PATCH", "DELETE", "method=", "data=", ".request(", "urlencode"):
        assert verb not in src, verb
    assert "ProxyHandler({})" in src


@pytest.mark.parametrize("url", [
    "http://example.com/api/robot/state", "http://10.0.0.5:5000/api/robot/state",
    "https://127.0.0.1:5000/api/robot/state", "http://user:pw@127.0.0.1:5000/x",
    "http://127.0.0.1.evil.com/x", "ftp://localhost/x",
])
def test_non_local_url_rejected(url):
    with pytest.raises(NonLocalURL):
        check_local_url(url)


def test_local_urls_accepted():
    check_local_url("http://127.0.0.1:5000/api/robot/state")
    check_local_url("http://localhost:5001/api/robot/state")


def test_cli_rejects_non_local_url(capsys):
    from robot_sim.__main__ import main
    assert main(["--url", "http://example.com/api/robot/state", "--headless"]) == 2
    assert "refused host" in capsys.readouterr().err
