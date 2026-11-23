"""Interactive webcam teleoperation and calibration windows."""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from .driver import FINGERS, NUM_DOF, InspireHand
from .filters import OneEuroFilter
from .link import HandLink
from .retarget import Calibration, features
from .tracking import Camera, HandObservation, HandTracker, draw_hand

WINDOW = "inspire-hand-teleop"
LABELS = ("little", "ring", "middle", "index", "th bend", "th rot")


def _pick(hands: list[HandObservation], which: str) -> HandObservation | None:
    if which == "any":
        return max(hands, key=lambda h: h.score, default=None)
    wanted = which.capitalize()
    return next((h for h in hands if h.handedness == wanted), None)


PANEL_W = 300
GREY = (220, 220, 220)


def _put(img: np.ndarray, text: str, org: tuple[int, int], scale: float = 0.45, color=GREY):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def _compose(
    frame: np.ndarray, lines: list[str], target: np.ndarray | None, actual: np.ndarray
) -> np.ndarray:
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
        if target is not None:
            y = int(bottom - (1 - target[i] / 1000) * height)
            cv2.rectangle(panel, (x + 2, y), (x + bar_w - 2, bottom), (0, 170, 255), -1)
        if not np.isnan(actual[i]):
            y = int(bottom - (1 - actual[i] / 1000) * height)
            cv2.line(panel, (x - 3, y), (x + bar_w + 3, y), (80, 255, 80), 2)
        _put(panel, LABELS[i], (x - 2, bottom + 16), 0.33)

    if frame.shape[0] < h:
        pad = np.zeros((h - frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
        frame = np.vstack([frame, pad])
    return np.hstack([frame, panel])


def _text(frame: np.ndarray, lines: list[str]) -> None:
    for i, line in enumerate(lines):
        y = 22 + i * 20
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        _put(frame, line, (10, y), 0.5, (255, 255, 255))


def run_teleop(
    hand: InspireHand,
    camera_index: int = 0,
    calibration_path: Path | None = None,
    rate_hz: float = 60.0,
    which_hand: str = "any",
    speed: int = 1000,
    force: int = 500,
    log_path: Path | None = None,
    record_path: Path | None = None,
    min_cutoff: float = 1.5,
    beta: float = 0.02,
) -> None:
    calib, calibrated = Calibration.load_or_default(calibration_path)
    hand.set_speeds(speed)
    hand.set_forces(force)
    link = HandLink(hand, rate_hz=rate_hz, log_path=log_path)
    link.start()
    camera = Camera(camera_index)
    tracker = HandTracker()
    smooth = OneEuroFilter(min_cutoff=min_cutoff, beta=beta)
    writer = None
    target: np.ndarray | None = None
    seq, fps, last_t = 0, 0.0, time.monotonic()
    lost_since: float | None = None

    print("Keys: [space] pause/resume  [o] open hand  [r] reset filter  [q/esc] quit")
    try:
        while True:
            frame, capture_t, seq = camera.read(seq)
            frame = cv2.flip(frame, 1)
            obs = _pick(tracker.detect(frame, capture_t), which_hand)
            now = time.monotonic()
            fps = 0.9 * fps + 0.1 / max(now - last_t, 1e-3)
            last_t = now

            if obs is not None:
                lost_since = None
                raw = calib.to_angles(features(obs.world, obs.handedness))
                target = np.clip(smooth(raw, capture_t), 0, 1000)
                link.submit(target, capture_t)
                draw_hand(frame, obs)
            elif lost_since is None:
                lost_since = now
                smooth.reset()

            lines = [
                "RUNNING" if link.enabled else "PAUSED (hand holds pose)",
                f"no hand for {now - lost_since:.1f} s (holding)" if lost_since else "tracking",
                f"camera {fps:4.1f} fps",
                f"modbus write {link.write_ms:5.1f} ms",
                f"camera -> hand {link.pipeline_ms:5.1f} ms",
                f"bus errors {link.errors}",
                "calibrated" if calibrated else "NOT calibrated (run calibrate)",
            ]
            view = _compose(frame, lines, target, link.actual)
            if record_path is not None:
                if writer is None:
                    h, w = view.shape[:2]
                    writer = cv2.VideoWriter(
                        str(record_path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h)
                    )
                writer.write(view)
            cv2.imshow(WINDOW, view)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                link.enabled = not link.enabled
            elif key == ord("o"):
                link.enabled = False
                hand.open_all()
            elif key == ord("r"):
                smooth.reset()
    finally:
        link.stop()
        try:
            hand.open_all()
        finally:
            camera.close()
            tracker.close()
            if writer is not None:
                writer.release()
            cv2.destroyAllWindows()
    if link.last_error:
        print(f"{link.errors} bus errors, last: {link.last_error}")


CALIBRATION_POSES = (
    ("open", "Open hand, fingers straight and together, thumb flat beside the index finger"),
    ("closed", "Fist: fingers fully curled, thumb folded across the middle of the fingers"),
)


def run_calibration(
    output: Path, camera_index: int = 0, which_hand: str = "any", samples: int = 45
) -> Calibration:
    camera = Camera(camera_index)
    tracker = HandTracker()
    collected: dict[str, list[float]] = {}
    seq = 0
    try:
        for pose, instruction in CALIBRATION_POSES:
            buffer: list[np.ndarray] = []
            recording = False
            while len(buffer) < samples:
                frame, t, seq = camera.read(seq)
                frame = cv2.flip(frame, 1)
                obs = _pick(tracker.detect(frame, t), which_hand)
                if obs is not None:
                    draw_hand(frame, obs)
                    if recording:
                        buffer.append(features(obs.world, obs.handedness))
                status = f"recording {len(buffer)}/{samples}" if recording else "press SPACE"
                _text(
                    frame,
                    [
                        f"Pose '{pose}': {instruction}",
                        status,
                        "hand detected" if obs else "no hand detected",
                    ],
                )
                cv2.imshow(WINDOW, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    raise KeyboardInterrupt
                if key == ord(" ") and obs is not None:
                    recording = True
            collected[pose] = np.median(np.array(buffer), axis=0).round(4).tolist()
    finally:
        camera.close()
        tracker.close()
        cv2.destroyAllWindows()

    calib = Calibration(open=collected["open"], closed=collected["closed"])
    calib.save(output)
    print(f"Saved calibration to {output}")
    for name, lo, hi in zip(FINGERS, calib.open, calib.closed, strict=True):
        warn = "  <-- small range, check this pose" if abs(hi - lo) < 0.25 else ""
        print(f"  {name:13s} open {lo:+.3f}  closed {hi:+.3f}  range {hi - lo:+.3f}{warn}")
    return calib
