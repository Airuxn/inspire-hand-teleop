"""Background thread that streams the newest target to the hand at a fixed rate."""

from __future__ import annotations

import csv
import threading
import time
from pathlib import Path

import numpy as np

from .driver import FINGERS, NUM_DOF, InspireHand, ModbusError

LOG_HEADER = (
    ["t", "capture_t", "write_ms"]
    + [f"target_{f}" for f in FINGERS]
    + [f"actual_{f}" for f in FINGERS]
    + [f"force_{f}" for f in FINGERS]
)


class HandLink(threading.Thread):
    def __init__(
        self,
        hand: InspireHand,
        rate_hz: float = 60.0,
        deadband: int = 4,
        read_every: int = 3,
    ):
        super().__init__(daemon=True)
        self.hand = hand
        self.period = 1.0 / rate_hz
        self.deadband = deadband
        self.read_every = read_every
        self.enabled = True

        self.actual = np.full(NUM_DOF, np.nan)
        self.forces = np.full(NUM_DOF, np.nan)
        self.write_ms = float("nan")
        self.pipeline_ms = float("nan")
        self.errors = 0
        self.last_error = ""

        self._lock = threading.Lock()
        self._target: np.ndarray | None = None
        self._target_capture_t = 0.0
        self._sent: np.ndarray | None = None
        self._stop = threading.Event()
        self._log_file = None
        self._log = None

    def submit(self, target: np.ndarray, capture_t: float) -> None:
        with self._lock:
            self._target = np.asarray(target, dtype=float).copy()
            self._target_capture_t = capture_t

    def log_to(self, path: Path | None) -> None:
        """Start logging to `path` (CSV), or stop logging when `path` is None."""
        with self._lock:
            if self._log_file:
                self._log_file.close()
            self._log_file = self._log = None
            if path is not None:
                self._log_file = open(path, "w", newline="")  # noqa: SIM115
                self._log = csv.writer(self._log_file)
                self._log.writerow(LOG_HEADER)

    def stop(self) -> None:
        self._stop.set()
        self.join(timeout=1.0)
        self.log_to(None)

    def run(self) -> None:
        cycle = 0
        next_t = time.monotonic()
        while not self._stop.is_set():
            with self._lock:
                target, capture_t = self._target, self._target_capture_t
            try:
                sent_now = False
                if self.enabled and target is not None and self._should_send(target):
                    t0 = time.monotonic()
                    self.hand.set_angles(target)
                    t1 = time.monotonic()
                    self.write_ms = (t1 - t0) * 1000
                    self.pipeline_ms = (t1 - capture_t) * 1000
                    self._sent = target
                    sent_now = True
                if cycle % self.read_every == 0:
                    self.actual = np.array(self.hand.get_angles(), dtype=float)
                    self.forces = np.array(self.hand.get_forces(), dtype=float)
                if sent_now or cycle % self.read_every == 0:
                    self._write_log(capture_t)
            except ModbusError as exc:
                self.errors += 1
                self.last_error = str(exc)
            cycle += 1
            next_t += self.period
            delay = next_t - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                next_t = time.monotonic()

    def _write_log(self, capture_t: float) -> None:
        with self._lock:
            if self._log is None:
                return
            tgt = self._sent if self._sent is not None else np.full(NUM_DOF, np.nan)
            self._log.writerow(
                [f"{time.monotonic():.4f}", f"{capture_t:.4f}", f"{self.write_ms:.2f}"]
                + [f"{v:.0f}" for v in tgt]
                + [f"{v:.0f}" for v in self.actual]
                + [f"{v:.0f}" for v in self.forces]
            )

    def _should_send(self, target: np.ndarray) -> bool:
        if self._sent is None:
            return True
        return bool(np.max(np.abs(np.round(target) - np.round(self._sent))) >= self.deadband)
