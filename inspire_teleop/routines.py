"""Hardware test routines shared by the CLI and the app."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

from .driver import FINGERS, NUM_DOF, InspireHand


def wait_settled(hand: InspireHand, timeout: float = 4.0, still_for: float = 0.25) -> float:
    """Poll until no actuator moves for `still_for` seconds; returns the elapsed time."""
    t0 = time.monotonic()
    last, still_since = hand.get_angles(), time.monotonic()
    while (now := time.monotonic()) - t0 < timeout:
        time.sleep(0.02)
        angles = hand.get_angles()
        if max(abs(a - b) for a, b in zip(angles, last, strict=True)) > 2:
            last, still_since = angles, now
        elif now - still_since >= still_for:
            break
    return time.monotonic() - t0


def finger_test(
    hand: InspireHand,
    depth: int = 400,
    report: Callable[[str], None] = print,
    should_stop: Callable[[], bool] = lambda: False,
) -> None:
    """Close each actuator in turn to `depth` and reopen it, reporting the measured angle."""
    hand.open_all()
    wait_settled(hand)
    for i, finger in enumerate(FINGERS):
        if should_stop():
            break
        angles = [1000] * NUM_DOF
        angles[i] = depth
        hand.set_angles(angles)
        elapsed = wait_settled(hand)
        report(
            f"{finger:13s} commanded {depth:4d} -> measured {hand.get_angles()[i]:4d} "
            f"(settled after {elapsed:.2f} s)"
        )
        hand.open_all()
        wait_settled(hand)


@dataclass
class StepResult:
    start: int
    target: int
    samples: list[tuple[float, int]]  # (seconds since command, measured angle)

    def _first_time(self, fraction: float) -> float | None:
        span = self.target - self.start
        for t, a in self.samples:
            if span and (a - self.start) / span >= fraction:
                return t
        return None

    @property
    def dead_time(self) -> float | None:
        return self._first_time(0.02)

    @property
    def rise_time(self) -> float | None:
        t10, t90 = self._first_time(0.1), self._first_time(0.9)
        return t90 - t10 if t10 is not None and t90 is not None else None

    @property
    def poll_hz(self) -> float:
        return len(self.samples) / self.samples[-1][0] if self.samples else 0.0

    def summary(self) -> str:
        def ms(v: float | None) -> str:
            return f"{v * 1000:6.1f} ms" if v is not None else "   n/a"

        final = self.samples[-1][1] if self.samples else -1
        return (
            f"step {self.start:4d} -> {self.target:4d}: dead time {ms(self.dead_time)}, "
            f"10-90% rise {ms(self.rise_time)}, final {final:4d}, polling {self.poll_hz:.0f} Hz"
        )


def step_response(
    hand: InspireHand, finger: str, low: int = 200, duration: float = 1.5
) -> list[StepResult]:
    """Close one actuator to `low` and reopen it, sampling the measured angle at full speed."""
    i = FINGERS.index(finger)
    hand.open_all()
    wait_settled(hand)
    results = []
    for target in (low, 1000):
        angles = [1000] * NUM_DOF
        angles[i] = target
        start = hand.get_angles()[i]
        samples: list[tuple[float, int]] = []
        t0 = time.monotonic()
        hand.set_angles(angles)
        while (t := time.monotonic() - t0) < duration:
            samples.append((t, hand.get_angles()[i]))
        results.append(StepResult(start, target, samples))
    return results
