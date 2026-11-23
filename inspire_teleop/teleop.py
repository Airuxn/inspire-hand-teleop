"""Minimal OpenCV front end (keyboard-driven) for `run` and `calibrate` on the command line."""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from .driver import FINGERS, NUM_DOF
from .session import CALIBRATION_POSES, CALIBRATION_SAMPLES, Snapshot, TeleopSession

WINDOW = "inspire-hand-teleop"
LABELS = ("little", "ring", "middle", "index", "th bend", "th rot")
PANEL_W = 300
GREY = (220, 220, 220)


def _put(img: np.ndarray, text: str, org: tuple[int, int], scale: float = 0.45, color=GREY):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def compose(frame: np.ndarray, lines: list[str], snap: Snapshot) -> np.ndarray:
    """Camera view with a side panel: status lines on top, per-actuator closure bars below."""
    h = max(frame.shape[0], 360)
    panel = np.full((h, PANEL_W, 3), 24, dtype=np.uint8)
    for i, line in enumerate(lines):
        _put(panel, line, (10, 22 + i * 20))

    bar_w, gap = 34, 12
    top, bottom = 40 + 20 * len(lines), h - 30
    height = bottom - top
    _put(panel, "closure  orange=target  green=measured", (10, top - 10), 0.38)
    for i in range(NUM_DOF):
        x = 12 + i * (bar_w + gap)
        cv2.rectangle(panel, (x, top), (x + bar_w, bottom), (70, 70, 70), 1)
        if snap.target is not None:
            y = int(bottom - (1 - snap.target[i] / 1000) * height)
            cv2.rectangle(panel, (x + 2, y), (x + bar_w - 2, bottom), (0, 170, 255), -1)
        if not np.isnan(snap.actual[i]):
            y = int(bottom - (1 - snap.actual[i] / 1000) * height)
            cv2.line(panel, (x - 3, y), (x + bar_w + 3, y), (80, 255, 80), 2)
        _put(panel, LABELS[i], (x - 2, bottom + 16), 0.33)

    if frame.shape[0] < h:
        pad = np.zeros((h - frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
        frame = np.vstack([frame, pad])
    return np.hstack([frame, panel])


def _wait_frame(session: TeleopSession, last: Snapshot | None) -> tuple[Snapshot, np.ndarray]:
    while True:
        snap = session.snapshot()
        if snap.frame is not None and (last is None or snap.frame is not last.frame):
            return snap, snap.frame
        time.sleep(0.003)


def run_teleop(
    session: TeleopSession,
    camera_index: int = 0,
    record_path: Path | None = None,
) -> None:
    session.start_camera(camera_index)
    session.set_teleop(True)
    writer = None
    snap = None
    print("Keys: [space] pause/resume  [o] open hand  [r] reset filter  [q/esc] quit")
    try:
        while True:
            snap, frame = _wait_frame(session, snap)
            lines = [
                "RUNNING" if snap.teleop_on else "PAUSED (hand holds pose)",
                "tracking" if snap.hand_detected else "no hand (holding last pose)",
                f"camera {snap.fps:4.1f} fps",
                f"modbus write {snap.write_ms:5.1f} ms",
                f"camera -> hand {snap.pipeline_ms:5.1f} ms",
                f"bus errors {snap.bus_errors}",
                "calibrated" if snap.calibrated else "NOT calibrated (run calibrate)",
            ]
            view = compose(frame, lines, snap)
            if record_path is not None:
                if writer is None:
                    h, w = view.shape[:2]
                    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
                    writer = cv2.VideoWriter(str(record_path), fourcc, 30, (w, h))
                writer.write(view)
            cv2.imshow(WINDOW, view)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                session.set_teleop(not snap.teleop_on)
            elif key == ord("o"):
                session.open_hand()
            elif key == ord("r"):
                session.filter.reset()
    finally:
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
    if snap is not None and snap.last_error:
        print(f"{snap.bus_errors} bus errors, last: {snap.last_error}")


def run_calibration(session: TeleopSession, camera_index: int = 0) -> None:
    session.start_camera(camera_index)
    snap = None
    try:
        for pose, instruction in CALIBRATION_POSES.items():
            started = False
            while pose not in session.recorded_poses():
                snap, frame = _wait_frame(session, snap)
                frame = frame.copy()
                if snap.calibrating:
                    status = f"recording {snap.calib_progress}/{CALIBRATION_SAMPLES}"
                else:
                    status = "press SPACE to record"
                for i, line in enumerate(
                    [instruction, status, "hand detected" if snap.hand_detected else "no hand"]
                ):
                    y = 22 + i * 20
                    cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
                    _put(frame, line, (10, y), 0.5, (255, 255, 255))
                cv2.imshow(WINDOW, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    raise KeyboardInterrupt
                if key == ord(" ") and not started:
                    session.start_calibration_pose(pose)
                    started = True
    finally:
        cv2.destroyAllWindows()

    rejected = session.save_calibration()
    calib = session.calibration
    print(f"Saved calibration to {session.calibration_path}")
    for name, lo, hi in zip(FINGERS, calib.open, calib.closed, strict=True):
        note = "  <-- poses too similar, kept previous values" if name in rejected else ""
        print(f"  {name:13s} open {lo:+.3f}  closed {hi:+.3f}  range {hi - lo:+.3f}{note}")
