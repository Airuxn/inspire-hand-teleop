"""Camera thread and model download, with OpenCV capture and the network replaced."""

import threading

import numpy as np
import pytest

from inspire_teleop import tracking


class FakeCapture:
    def __init__(self, opened: bool = True, fail_first: int = 0):
        self.opened, self.fail_first = opened, fail_first
        self.props: dict[int, float] = {}
        self.frames = 0
        self.released = threading.Event()

    def isOpened(self) -> bool:  # noqa: N802 (OpenCV API)
        return self.opened

    def set(self, prop: int, value: float) -> bool:
        self.props[prop] = value
        return True

    def read(self):
        if self.fail_first:
            self.fail_first -= 1
            return False, None
        self.frames += 1
        return True, np.full((4, 4, 3), self.frames % 256, np.uint8)

    def release(self) -> None:
        self.released.set()


def test_camera_delivers_newer_frames_and_configures_capture(monkeypatch):
    cap = FakeCapture(fail_first=2)
    monkeypatch.setattr(tracking.cv2, "VideoCapture", lambda *args: cap)
    camera = tracking.Camera(0, width=320, height=240, fps=60)
    try:
        frame, t, seq = camera.read(0, timeout=2.0)
        _, t2, seq2 = camera.read(seq, timeout=2.0)
        assert frame.shape == (4, 4, 3)
        assert seq2 > seq and t2 >= t
        assert cap.props[tracking.cv2.CAP_PROP_FRAME_WIDTH] == 320
        assert cap.props[tracking.cv2.CAP_PROP_FPS] == 60
    finally:
        camera.close()
    assert cap.released.is_set()


def test_camera_falls_back_to_default_backend(monkeypatch):
    calls = []

    def capture(*args):
        calls.append(args)
        return FakeCapture(opened=len(calls) > 1)

    monkeypatch.setattr(tracking.cv2, "VideoCapture", capture)
    tracking.Camera(3).close()
    assert calls == [(3, tracking.cv2.CAP_V4L2), (3,)]


def test_camera_that_cannot_open_raises(monkeypatch):
    monkeypatch.setattr(tracking.cv2, "VideoCapture", lambda *args: FakeCapture(opened=False))
    with pytest.raises(RuntimeError, match="cannot open camera 7"):
        tracking.Camera(7)


def test_camera_read_times_out_without_frames(monkeypatch):
    cap = FakeCapture(fail_first=10**9)
    monkeypatch.setattr(tracking.cv2, "VideoCapture", lambda *args: cap)
    camera = tracking.Camera(0)
    try:
        with pytest.raises(TimeoutError):
            camera.read(0, timeout=0.05)
    finally:
        camera.close()


def test_ensure_model_downloads_once(tmp_path, monkeypatch):
    downloads = []

    def fake_download(url, dest):
        downloads.append(url)
        dest.write_bytes(b"model")

    monkeypatch.setattr(tracking.urllib.request, "urlretrieve", fake_download)
    path = tmp_path / "models" / "hand.task"
    assert tracking.ensure_model(path) == path
    assert tracking.ensure_model(path) == path
    assert path.read_bytes() == b"model"
    assert downloads == [tracking.MODEL_URL]
    assert not path.with_suffix(".part").exists()
