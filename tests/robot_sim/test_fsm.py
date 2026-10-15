from robot_sim import controller as C
from robot_sim.fsm import Mode, RobotFSM
from robot_sim.link import BAD_JSON
from robot_sim.states import RobotState as S



class FakeClock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


def payload(state_cmd, seq, boot="aaaa1111"):
    return C.to_payload(state_cmd, boot, seq)


CLEAR = C.from_report({"status": "AGREEMENT", "risk_class": "not_depression", "confidence": 0.9}, 4)
FLAG = C.from_report({"status": "AGREEMENT", "risk_class": "depression", "confidence": 0.9}, 5)
REVIEW = C.from_report({"status": "DISAGREEMENT"}, 6)


def make(mode=Mode.LIVE):
    clk = FakeClock()
    return RobotFSM(mode=mode, clock=clk), clk


def test_live_transitions_follow_payload():
    f, _ = make()
    f.on_poll_success(payload(C.idle_command(), 0))
    assert f.state is S.IDLE
    f.on_poll_success(payload(C.processing_command(), 1))
    assert f.state is S.PROCESSING
    f.on_poll_success(payload(REVIEW, 2))
    assert f.state is S.REVIEW_NEEDED and f.report["status"] == "DISAGREEMENT"
    assert [t.to for t in f.log] == ["IDLE", "PROCESSING", "REVIEW_NEEDED"]


def test_seq_and_boot_rules():
    f, _ = make()
    f.on_poll_success(payload(FLAG, 5))
    g = f.generation
    f.on_poll_success(payload(FLAG, 5))            # duplicate
    assert f.generation == g and f.gate.duplicates == 1
    f.on_poll_success(payload(CLEAR, 3))           # out of order
    assert f.state is S.SCREEN_FLAGGED and f.gate.out_of_order == 1
    f.on_poll_success(payload(C.idle_command(), 0, boot="bbbb2222"))   # restart
    assert f.state is S.IDLE


def test_same_state_new_seq_reenters():
    f, _ = make()
    f.on_poll_success(payload(CLEAR, 1))
    g = f.generation
    f.on_poll_success(payload(CLEAR, 2))
    assert f.generation == g + 1


def test_invalid_payload_is_error_never_clear():
    f, _ = make()
    f.on_poll_success(payload(CLEAR, 1))
    f.on_poll_success({"schema_version": 1, "state": "SCREEN_CLEAR"})
    assert f.state is S.ERROR
    f.on_poll_success(BAD_JSON)
    assert f.state is S.ERROR and f.invalid_payloads == 2


def test_offline_after_three_failures_and_recovery():
    f, _ = make()
    f.on_poll_success(payload(FLAG, 2))
    f.on_poll_failure("unreachable")
    f.on_poll_failure("unreachable")
    assert f.state is S.SCREEN_FLAGGED
    f.on_poll_failure("unreachable")
    assert f.state is S.OFFLINE and f.link_ok is False
    f.on_poll_success(payload(FLAG, 2))            # same seq still applied after offline
    assert f.state is S.SCREEN_FLAGGED


def test_processing_timeout():
    f, clk = make()
    f.on_poll_success(payload(C.processing_command(), 1))
    clk.advance(119)
    f.tick()
    assert f.state is S.PROCESSING
    clk.advance(2)
    f.tick()
    assert f.state is S.ERROR and f.cause == "timeout"


def test_manual_buffers_live_and_returns():
    f, _ = make()
    f.on_poll_success(payload(C.idle_command(), 0))
    f.set_mode(Mode.MANUAL)
    assert f.manual_select(S.SCREEN_CLEAR)
    assert f.source == "manual"
    f.on_poll_success(payload(REVIEW, 7))
    assert f.state is S.SCREEN_CLEAR and f.live_waiting
    f.on_poll_failure(); f.on_poll_failure(); f.on_poll_failure()
    assert f.state is S.SCREEN_CLEAR              # link loss not applied in manual
    f.on_poll_success(payload(REVIEW, 7))
    f.set_mode(Mode.LIVE)
    assert f.state is S.REVIEW_NEEDED
    assert "report #6" in f.current_flash()


def test_manual_keys_ignored_in_live():
    f, _ = make()
    assert f.manual_select(S.SCREEN_FLAGGED) is False
    assert f.state is S.IDLE


def test_return_to_live_while_offline():
    f, _ = make(Mode.MANUAL)
    for _ in range(3):
        f.on_poll_failure()
    f.set_mode(Mode.LIVE)
    assert f.state is S.OFFLINE


def test_dismiss_and_confirm():
    f, _ = make()
    f.on_poll_success(payload(FLAG, 1))
    f.confirm()
    assert f.state is S.SCREEN_FLAGGED and f.events[-1][1] == "acknowledged"
    assert f.dismiss()
    assert f.state is S.IDLE and f.report_dismissed and f.report is not None
    f.on_poll_success(payload(CLEAR, 2))
    assert f.state is S.SCREEN_CLEAR and not f.report_dismissed


def test_dismiss_cannot_hide_offline_or_processing():
    f, _ = make()
    f.on_poll_success(payload(C.processing_command(), 1))
    assert f.dismiss() is False and f.state is S.PROCESSING
    for _ in range(3):
        f.on_poll_failure()
    assert f.dismiss() is False and f.state is S.OFFLINE


def test_transition_csv_has_no_report_content(tmp_path):
    f, _ = make()
    flag = C.from_report({"status": "AGREEMENT", "risk_class": "depression", "confidence": 0.4321,
                          "consent_status": "granted"}, 5)
    f.on_poll_success(payload(flag, 1))
    path = f.write_csv(tmp_path / "t.csv")
    text = path.read_text()
    assert "SCREEN_FLAGGED" in text
    for leak in ("confidence", "4321", "granted", "AGREEMENT"):
        assert leak not in text


def test_dashboard_restart_clears_old_report():
    f, _ = make()
    f.on_poll_success(payload(FLAG, 4))
    assert f.report is not None
    for _ in range(3):
        f.on_poll_failure()
    f.on_poll_success(payload(C.idle_command(), 0, boot="bbbb2222"))
    assert f.state is S.IDLE and f.report is None
