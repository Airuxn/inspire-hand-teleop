"""Teleoperation session: camera -> tracking -> retargeting -> filter -> hand link.

Runs the vision pipeline on a worker thread. Front ends (the Qt app and the OpenCV
window) call the control methods and poll `snapshot()` for display.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .driver import FINGERS, NUM_DOF, InspireHand, ModbusError, ModbusRTU
from .filters import OneEuroFilter
from .link import HandLink
from .retarget import Calibration, features
from .tracking import Camera, HandObservation, HandTracker, draw_hand

CALIBRATION_POSES = {
    "open": "Open hand: fingers straight and together, thumb flat beside the index finger",
    "closed": "Fist: fingers fully curled, thumb folded across the middle of the fingers",
}
CALIBRATION_SAMPLES = 45
MIN_CALIBRATION_RANGE = 0.25  # radians between open and fist


def pick_hand(hands: list[HandObservation], which: str) -> HandObservation | None:
    if which == "any":
        return max(hands, key=lambda h: h.score, default=None)
    return next((h for h in hands if h.handedness == which.capitalize()), None)


@dataclass
class Snapshot:
    frame: np.ndarray | None = None  # mirrored, annotated BGR
    camera_on: bool = False
    hand_detected: bool = False
    handedness: str = ""
    connected: bool = False
    mock: bool = False
    teleop_on: bool = False
    busy: str = ""  # name of a running hardware routine, if any
    target: np.ndarray | None = None
    actual: np.ndarray = field(default_factory=lambda: np.full(NUM_DOF, np.nan))
    forces: np.ndarray = field(default_factory=lambda: np.full(NUM_DOF, np.nan))
    fps: float = 0.0
    write_ms: float = float("nan")
    pipeline_ms: float = float("nan")
    bus_errors: int = 0
    last_error: str = ""
    calibrated: bool = False
    calibrating: str = ""
    calib_progress: int = 0
    logging_to: str = ""


class TeleopSession:
    def __init__(self, calibration_path: Path = Path("calibration.json"), which_hand: str = "any"):
        self.calibration_path = calibration_path
        self.which_hand = which_hand
        self.calibration, self.calibrated = Calibration.load_or_default(calibration_path)
        self.filter = OneEuroFilter()
        self.speed, self.force = 1000, 500

        self.camera: Camera | None = None
        self.tracker: HandTracker | None = None
        self.hand: InspireHand | None = None
        self.link: HandLink | None = None
        self.mock = False
        self.teleop_on = False
        self.busy = ""
        self.log_path: Path | None = None

        self._lock = threading.Lock()
        self._snapshot = Snapshot()
        self._target: np.ndarray | None = None
        self._calib_pose = ""
        self._calib_buffer: list[np.ndarray] = []
        self._calib_results: dict[str, list[float]] = {}
        self._running = True
        self._worker = threading.Thread(target=self._loop, daemon=True)
        self._worker.start()

    # camera ---------------------------------------------------------------

    def start_camera(self, index: int = 0) -> None:
        self.stop_camera()
        if self.tracker is None:
            self.tracker = HandTracker()
        self.camera = Camera(index)

    def stop_camera(self) -> None:
        camera, self.camera = self.camera, None
        if camera is not None:
            camera.close()
        self.filter.reset()

    # hand -----------------------------------------------------------------

    def connect(
        self,
        port: str | None,
        mock: bool = False,
        slave_id: int = 1,
        baudrate: int = 115200,
        rate_hz: float = 60.0,
    ) -> int:
        """Open the hand (or a simulated one) and start streaming; returns the hand id."""
        self.disconnect()
        if mock:
            from .sim_serial import FakeInspireSerial

            hand = InspireHand(ModbusRTU(FakeInspireSerial(slave_id, latency=True), slave_id))
        else:
            if port is None:
                raise ValueError("no serial port selected")
            hand = InspireHand.open(port, baudrate, slave_id)
        try:
            hand_id = hand.hand_id()
            hand.set_speeds(self.speed)
            hand.set_forces(self.force)
        except ModbusError:
            hand.close()
            raise
        self.hand, self.mock = hand, mock
        self.link = HandLink(hand, rate_hz=rate_hz)
        self.link.enabled = False
        self.link.log_to(self.log_path)
        self.link.start()
        return hand_id

    def disconnect(self) -> None:
        self.teleop_on = False
        link, self.link = self.link, None
        hand, self.hand = self.hand, None
        if link is not None:
            link.stop()
        if hand is not None:
            try:
                hand.open_all()
            except ModbusError:
                pass
            hand.close()

    def set_motion_limits(self, speed: int, force: int) -> None:
        self.speed, self.force = speed, force
        if self.hand is not None:
            self.hand.set_speeds(speed)
            self.hand.set_forces(force)

    def set_teleop(self, on: bool) -> None:
        if on and (self.hand is None or self.busy):
            return
        self.teleop_on = on
        if self.link is not None:
            self.link.enabled = on

    def open_hand(self) -> None:
        self.set_teleop(False)
        if self.hand is not None:
            self.hand.open_all()

    def clear_errors(self) -> None:
        if self.hand is not None:
            self.hand.clear_errors()
        if self.link is not None:
            self.link.errors, self.link.last_error = 0, ""

    def run_routine(
        self,
        name: str,
        routine: Callable[[InspireHand], None],
        on_done: Callable[[Exception | None], None] = lambda exc: None,
    ) -> None:
        """Run a blocking hardware routine (e.g. finger test) on its own thread."""
        if self.hand is None or self.busy:
            return
        self.set_teleop(False)
        self.busy = name
        hand = self.hand

        def target() -> None:
            error: Exception | None = None
            try:
                routine(hand)
            except Exception as exc:  # reported to the UI, not raised on a daemon thread
                error = exc
            finally:
                self.busy = ""
                on_done(error)

        threading.Thread(target=target, daemon=True).start()

    def set_logging(self, path: Path | None) -> None:
        self.log_path = path
        if self.link is not None:
            self.link.log_to(path)

    # calibration ----------------------------------------------------------

    def start_calibration_pose(self, pose: str) -> None:
        if pose not in CALIBRATION_POSES:
            raise ValueError(pose)
        with self._lock:
            self._calib_pose, self._calib_buffer = pose, []

    def calibration_ready(self) -> bool:
        return set(self._calib_results) == set(CALIBRATION_POSES)

    def recorded_poses(self) -> list[str]:
        return list(self._calib_results)

    def save_calibration(self) -> list[str]:
        """Save the recorded poses; returns actuators that kept their previous calibration
        because open and fist were too similar (a tiny range would amplify tracking noise)."""
        new_open, new_closed = self._calib_results["open"], self._calib_results["closed"]
        calib = Calibration(open=list(self.calibration.open), closed=list(self.calibration.closed))
        rejected = []
        for i, finger in enumerate(FINGERS):
            if abs(new_closed[i] - new_open[i]) < MIN_CALIBRATION_RANGE:
                rejected.append(finger)
            else:
                calib.open[i], calib.closed[i] = new_open[i], new_closed[i]
        calib.save(self.calibration_path)
        self.calibration, self.calibrated = calib, True
        self._calib_results = {}
        return rejected

    # loop -----------------------------------------------------------------

    def snapshot(self) -> Snapshot:
        with self._lock:
            return self._snapshot

    def _loop(self) -> None:
        seq, fps, last_t = 0, 0.0, time.monotonic()
        while self._running:
            camera, tracker = self.camera, self.tracker
            frame, obs = None, None
            if camera is not None and tracker is not None:
                try:
                    frame, capture_t, seq = camera.read(seq, timeout=0.5)
                except (TimeoutError, AttributeError):
                    continue
                frame = cv2.flip(frame, 1)
                obs = pick_hand(tracker.detect(frame, capture_t), self.which_hand)
                now = time.monotonic()
                fps = 0.9 * fps + 0.1 / max(now - last_t, 1e-3)
                last_t = now
                if obs is not None:
                    draw_hand(frame, obs)
                    self._process(obs, capture_t)
                else:
                    self.filter.reset()
            else:
                fps = 0.0
                time.sleep(0.03)
            self._publish(frame, obs, fps)

    def _process(self, obs: HandObservation, capture_t: float) -> None:
        feats = features(obs.world, obs.handedness)
        with self._lock:
            if self._calib_pose:
                self._calib_buffer.append(feats)
                if len(self._calib_buffer) >= CALIBRATION_SAMPLES:
                    median = np.median(np.array(self._calib_buffer), axis=0)
                    self._calib_results[self._calib_pose] = median.round(4).tolist()
                    self._calib_pose, self._calib_buffer = "", []
        raw = self.calibration.to_angles(feats)
        self._target = np.clip(self.filter(raw, capture_t), 0, 1000)
        link = self.link
        if link is not None and self.teleop_on:
            link.submit(self._target, capture_t)

    def _publish(self, frame: np.ndarray | None, obs: HandObservation | None, fps: float) -> None:
        link = self.link
        snap = Snapshot(
            frame=frame,
            camera_on=self.camera is not None,
            hand_detected=obs is not None,
            handedness=obs.handedness if obs else "",
            connected=self.hand is not None,
            mock=self.mock,
            teleop_on=self.teleop_on,
            busy=self.busy,
            target=None if self._target is None else self._target.copy(),
            fps=fps,
            calibrated=self.calibrated,
            calibrating=self._calib_pose,
            calib_progress=len(self._calib_buffer),
            logging_to=str(self.log_path) if self.log_path else "",
        )
        if link is not None:
            snap.actual, snap.forces = link.actual.copy(), link.forces.copy()
            snap.write_ms, snap.pipeline_ms = link.write_ms, link.pipeline_ms
            snap.bus_errors, snap.last_error = link.errors, link.last_error
        with self._lock:
            self._snapshot = snap

    def close(self) -> None:
        self._running = False
        self._worker.join(timeout=1.0)
        self.disconnect()
        self.stop_camera()
        if self.tracker is not None:
            self.tracker.close()
