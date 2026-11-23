"""Webcam capture and MediaPipe hand landmark tracking."""

from __future__ import annotations

import threading
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
DEFAULT_MODEL_PATH = Path.home() / ".cache" / "inspire-hand-teleop" / "hand_landmarker.task"

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
]  # fmt: skip


def ensure_model(path: Path = DEFAULT_MODEL_PATH) -> Path:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading hand landmarker model to {path} ...")
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(MODEL_URL, tmp)
        tmp.rename(path)
    return path


@dataclass
class HandObservation:
    world: np.ndarray  # (21, 3) metres, origin near the hand centre
    image: np.ndarray  # (21, 2) normalised image coordinates
    handedness: str  # "Left" / "Right" as seen by the user (input is mirrored)
    score: float


class HandTracker:
    def __init__(self, model_path: Path | None = None, num_hands: int = 1, min_conf: float = 0.5):
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import (
            HandLandmarker,
            HandLandmarkerOptions,
            RunningMode,
        )

        options = HandLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=str(ensure_model(model_path or DEFAULT_MODEL_PATH))
            ),
            running_mode=RunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=min_conf,
            min_hand_presence_confidence=min_conf,
            min_tracking_confidence=min_conf,
        )
        self._landmarker = HandLandmarker.create_from_options(options)
        self._last_ts = -1

    def detect(self, frame_bgr: np.ndarray, t: float) -> list[HandObservation]:
        """`frame_bgr` should already be mirrored (selfie view) so handedness labels match."""
        import mediapipe as mp

        ts_ms = max(int(t * 1000), self._last_ts + 1)
        self._last_ts = ts_ms
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self._landmarker.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts_ms
        )
        hands = []
        for world, image, handed in zip(
            result.hand_world_landmarks, result.hand_landmarks, result.handedness, strict=True
        ):
            hands.append(
                HandObservation(
                    world=np.array([[p.x, p.y, p.z] for p in world]),
                    image=np.array([[p.x, p.y] for p in image]),
                    handedness=handed[0].category_name,
                    score=handed[0].score,
                )
            )
        return hands

    def close(self) -> None:
        self._landmarker.close()


class Camera:
    """Grabs frames on a background thread so the control loop always sees the newest one."""

    def __init__(self, index: int = 0, width: int = 640, height: int = 480, fps: int = 30):
        self.cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open camera {index}")
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._frame: np.ndarray | None = None
        self._t = 0.0
        self._seq = 0
        self._cond = threading.Condition()
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while self._running:
            ok, frame = self.cap.read()
            t = time.monotonic()
            if not ok:
                time.sleep(0.01)
                continue
            with self._cond:
                self._frame, self._t = frame, t
                self._seq += 1
                self._cond.notify_all()

    def read(self, last_seq: int = 0, timeout: float = 1.0) -> tuple[np.ndarray, float, int]:
        """Block until a frame newer than `last_seq` arrives; returns (frame, capture_time, seq)."""
        with self._cond:
            if not self._cond.wait_for(lambda: self._seq > last_seq, timeout):
                raise TimeoutError("no camera frame received")
            assert self._frame is not None
            return self._frame, self._t, self._seq

    def close(self) -> None:
        self._running = False
        self._thread.join(timeout=1.0)
        self.cap.release()


def draw_hand(frame: np.ndarray, obs: HandObservation) -> None:
    h, w = frame.shape[:2]
    pts = [(int(x * w), int(y * h)) for x, y in obs.image]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0, 200, 255), 2, cv2.LINE_AA)
    for p in pts:
        cv2.circle(frame, p, 3, (255, 255, 255), -1, cv2.LINE_AA)
