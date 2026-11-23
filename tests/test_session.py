import csv
import threading
import time

import numpy as np
import pytest
from conftest import wait_until
from test_retarget import synthetic_hand

from inspire_teleop.retarget import Calibration
from inspire_teleop.session import CALIBRATION_SAMPLES, TeleopSession, pick_hand
from inspire_teleop.tracking import HandObservation


@pytest.fixture
def session(tmp_path):
    s = TeleopSession(tmp_path / "calib.json")
    yield s
    s.close()


def observation(
    curl: float, handedness: str = "Right", score: float = 0.9, thumb_curl: float = 0.0
) -> HandObservation:
    return HandObservation(
        world=synthetic_hand(curl, thumb_curl),
        image=np.zeros((21, 2)),
        handedness=handedness,
        score=score,
    )


def test_save_calibration_keeps_previous_values_for_tiny_ranges(session, tmp_path):
    default = Calibration()
    session._calib_results = {
        "open": [0.1, 0.1, 0.1, 0.1, 0.2, 0.0],
        "closed": [2.5, 2.5, 2.5, 0.15, 1.0, 0.5],
    }
    assert session.save_calibration() == ["index"]
    saved = Calibration.load(tmp_path / "calib.json")
    assert saved.open[0] == 0.1 and saved.closed[0] == 2.5
    assert saved.open[3] == default.open[3] and saved.closed[3] == default.closed[3]
    assert session.calibrated


def test_calibration_records_median_of_samples(session):
    session.start_calibration_pose("open")
    for _ in range(CALIBRATION_SAMPLES):
        session._process(observation(0.0), time.monotonic())
    session.start_calibration_pose("closed")
    for _ in range(CALIBRATION_SAMPLES):
        session._process(observation(1.2, thumb_curl=0.8), time.monotonic())
    assert session.calibration_ready()
    assert session.recorded_poses() == ["open", "closed"]
    assert session.save_calibration() == []
    assert session.calibration.closed[3] > 2.0


def test_unknown_calibration_pose_is_rejected(session):
    with pytest.raises(ValueError):
        session.start_calibration_pose("peace")


def test_process_maps_open_and_closed_hand(session):
    session._process(observation(0.0), 1.0)
    assert np.all(session._target[:4] > 900)
    session.filter.reset()
    session._process(observation(1.2), 2.0)
    assert np.all(session._target[:4] < 100)


def test_pick_hand_by_handedness_and_score():
    left, right = observation(0, "Left", 0.6), observation(0, "Right", 0.9)
    assert pick_hand([left, right], "any") is right
    assert pick_hand([left, right], "left") is left
    assert pick_hand([left], "right") is None
    assert pick_hand([], "any") is None


def test_mock_connect_stream_and_disconnect(session):
    assert session.connect(None, mock=True) == 1
    session.set_teleop(True)
    target = [0, 100, 200, 300, 400, 500]
    session.link.submit(np.array(target), time.monotonic())
    assert wait_until(lambda: np.allclose(session.hand.get_angles(), target, atol=5))
    session.disconnect()
    assert session.hand is None and not session.teleop_on


def test_teleop_requires_a_connected_hand(session):
    session.set_teleop(True)
    assert not session.teleop_on


def test_connect_without_port_raises(session):
    with pytest.raises(ValueError):
        session.connect(None, mock=False)


def test_motion_limits_open_hand_and_clear_errors(session):
    session.connect(None, mock=True)
    session.set_motion_limits(300, 200)
    assert session.hand.bus.read_registers(1522, 6) == [300] * 6
    assert session.hand.bus.read_registers(1498, 6) == [200] * 6
    session.set_teleop(True)
    session.open_hand()
    assert not session.teleop_on
    session.link.errors, session.link.last_error = 3, "x"
    session.clear_errors()
    assert session.link.errors == 0


def test_logging_writes_csv(session, tmp_path):
    log = tmp_path / "run.csv"
    session.set_logging(log)
    session.connect(None, mock=True)
    session.set_teleop(True)
    session.link.submit(np.full(6, 500.0), time.monotonic())
    assert wait_until(lambda: len(log.read_text().splitlines()) > 5)
    session.set_logging(None)
    rows = list(csv.DictReader(log.open()))
    assert "target_index" in rows[0] and "force_thumb_rotate" in rows[0]


def test_routine_blocks_teleop_until_done(session):
    session.connect(None, mock=True)
    release, done = threading.Event(), threading.Event()
    errors = []
    session.run_routine(
        "test", lambda hand: release.wait(2), on_done=lambda e: (errors.append(e), done.set())
    )
    session.set_teleop(True)
    assert session.busy == "test" and not session.teleop_on
    release.set()
    assert done.wait(2)
    assert errors == [None]
    session.set_teleop(True)
    assert session.teleop_on


def test_routine_errors_are_reported(session):
    session.connect(None, mock=True)
    done = threading.Event()
    errors = []

    def broken(hand):
        raise RuntimeError("boom")

    session.run_routine("broken", broken, on_done=lambda e: (errors.append(e), done.set()))
    assert done.wait(2)
    assert isinstance(errors[0], RuntimeError) and session.busy == ""


def test_snapshot_reflects_state(session):
    session.connect(None, mock=True)
    assert wait_until(lambda: session.snapshot().connected)
    snap = session.snapshot()
    assert snap.mock and not snap.camera_on and snap.frame is None
