import threading
import time

import numpy as np

from inspire_teleop.retarget import Calibration
from inspire_teleop.session import TeleopSession


def test_save_calibration_keeps_previous_values_for_tiny_ranges(tmp_path):
    session = TeleopSession(tmp_path / "calib.json")
    try:
        default = Calibration()
        session._calib_results = {
            "open": [0.1, 0.1, 0.1, 0.1, 0.2, 0.0],
            "closed": [2.5, 2.5, 2.5, 0.15, 1.0, 0.5],
        }
        rejected = session.save_calibration()
        assert rejected == ["index"]
        saved = Calibration.load(tmp_path / "calib.json")
        assert saved.open[0] == 0.1 and saved.closed[0] == 2.5
        assert saved.open[3] == default.open[3] and saved.closed[3] == default.closed[3]
    finally:
        session.close()


def test_mock_connect_stream_and_disconnect(tmp_path):
    session = TeleopSession(tmp_path / "calib.json")
    try:
        assert session.connect(None, mock=True) == 1
        session.set_teleop(True)
        session.link.submit(np.array([0, 100, 200, 300, 400, 500]), time.monotonic())
        time.sleep(1.2)
        np.testing.assert_allclose(session.hand.get_angles(), [0, 100, 200, 300, 400, 500], atol=5)
        session.disconnect()
        assert session.hand is None and not session.teleop_on
    finally:
        session.close()


def test_routine_blocks_teleop_until_done(tmp_path):
    session = TeleopSession(tmp_path / "calib.json")
    try:
        session.connect(None, mock=True)
        release, done = threading.Event(), threading.Event()
        session.run_routine("test", lambda hand: release.wait(2), on_done=lambda e: done.set())
        session.set_teleop(True)
        assert session.busy == "test" and not session.teleop_on
        release.set()
        assert done.wait(2)
        session.set_teleop(True)
        assert session.teleop_on
    finally:
        session.close()
