"""End-to-end tests on real hand images: MediaPipe tracking -> retargeting -> actuator angles."""

import cv2
import numpy as np
import pytest
from conftest import DATA, wait_until

import inspire_teleop.teleop as teleop_module
from inspire_teleop.driver import FINGERS
from inspire_teleop.retarget import Calibration, features
from inspire_teleop.session import TeleopSession
from inspire_teleop.tracking import HandTracker, draw_hand

OPEN, CLOSED = 700, 300  # angle thresholds: >700 clearly open, <300 clearly closed


def load(name: str) -> np.ndarray:
    """Images are mirrored like the live camera view, so handedness labels match."""
    image = cv2.imread(str(DATA / f"{name}.jpg"))
    assert image is not None, name
    return cv2.flip(image, 1)


def angles_for(name: str) -> dict[str, float]:
    tracker = HandTracker()
    try:
        hands = tracker.detect(load(name), 1.0)
    finally:
        tracker.close()
    assert len(hands) == 1, f"expected one hand in {name}"
    values = Calibration().to_angles(features(hands[0].world, hands[0].handedness))
    return dict(zip(FINGERS, values, strict=True))


def test_victory_opens_index_and_middle_only():
    a = angles_for("victory")
    assert a["index"] > OPEN and a["middle"] > OPEN
    assert a["ring"] < CLOSED and a["little"] < CLOSED


def test_pointing_opens_index_only():
    a = angles_for("pointing_up")
    assert a["index"] > OPEN
    assert a["middle"] < CLOSED and a["ring"] < CLOSED and a["little"] < CLOSED


def test_fist_closes_all_fingers_and_thumb():
    a = angles_for("fist")
    assert all(a[f] < CLOSED for f in ("little", "ring", "middle", "index", "thumb_bend"))


def test_thumb_up_extends_thumb_with_fingers_closed():
    a = angles_for("thumb_up")
    assert a["thumb_bend"] > OPEN
    assert all(a[f] < CLOSED for f in ("little", "ring", "middle"))


def test_no_hand_in_empty_image():
    tracker = HandTracker()
    try:
        assert tracker.detect(np.full((240, 320, 3), 127, np.uint8), 1.0) == []
    finally:
        tracker.close()


def test_draw_hand_marks_the_image():
    tracker = HandTracker()
    image = load("victory")
    try:
        (hand,) = tracker.detect(image, 1.0)
    finally:
        tracker.close()
    before = image.copy()
    draw_hand(image, hand)
    assert np.any(image != before)


def test_session_drives_mock_hand_from_camera(still_camera, tmp_path):
    still_camera("victory")
    session = TeleopSession(tmp_path / "calib.json")
    try:
        session.connect(None, mock=True)
        session.start_camera(0)
        session.set_teleop(True)
        hand = session.hand
        assert hand is not None
        index = FINGERS.index("index")
        ring = FINGERS.index("ring")
        assert wait_until(lambda: session.snapshot().hand_detected, timeout=10)
        assert wait_until(
            lambda: hand.get_angles()[index] > OPEN and hand.get_angles()[ring] < CLOSED,
            timeout=10,
        )
        snap = session.snapshot()
        assert snap.camera_on and snap.fps > 0 and snap.handedness in ("Left", "Right")
    finally:
        session.close()


@pytest.fixture
def headless_cv2(monkeypatch):
    """Replace OpenCV windows; press `keys` in order, one per frame, then `q`."""
    shown: list[np.ndarray] = []

    def setup(keys: list[int]) -> list[np.ndarray]:
        pending = list(keys)

        def wait_key(_delay: int) -> int:
            return pending.pop(0) if pending else ord("q")

        monkeypatch.setattr(teleop_module.cv2, "imshow", lambda _name, img: shown.append(img))
        monkeypatch.setattr(teleop_module.cv2, "waitKey", wait_key)
        monkeypatch.setattr(teleop_module.cv2, "destroyAllWindows", lambda: None)
        return shown

    return setup


def test_cli_run_window_with_pause_open_and_record(still_camera, headless_cv2, tmp_path):
    still_camera("victory")
    shown = headless_cv2([-1] * 20 + [ord(" "), -1, ord("r"), ord("o")] + [-1] * 3)
    session = TeleopSession(tmp_path / "calib.json")
    video = tmp_path / "demo.mp4"
    try:
        session.connect(None, mock=True)
        teleop_module.run_teleop(session, 0, video)
        assert not session.teleop_on  # paused with space, then opened with o
    finally:
        session.close()
    assert shown and shown[-1].shape[1] == shown[0].shape[1]
    assert video.stat().st_size > 0


def test_cli_run_command_with_mock_hand(still_camera, headless_cv2, tmp_path, capsys):
    from inspire_teleop.cli import main

    still_camera("victory")
    headless_cv2([-1] * 10)
    log = tmp_path / "run.csv"
    main(["run", "--mock", "--calibration", str(tmp_path / "c.json"), "--log", str(log)])
    assert "Connected to hand id 1 (simulated)" in capsys.readouterr().out
    assert log.read_text().startswith("t,")


def test_cli_calibration_window_saves_both_poses(still_camera, headless_cv2, tmp_path):
    still_camera("fist")
    headless_cv2([-1] * 5 + [ord(" ")] + [-1] * 80 + [ord(" ")] + [-1] * 80)
    session = TeleopSession(tmp_path / "calib.json")
    try:
        teleop_module.run_calibration(session, 0)
    finally:
        session.close()
    assert (tmp_path / "calib.json").exists()
