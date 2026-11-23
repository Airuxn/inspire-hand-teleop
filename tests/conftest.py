from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

import pytest

from inspire_teleop import routines, sim_serial
from inspire_teleop.driver import InspireHand, ModbusRTU
from inspire_teleop.sim_serial import FakeClock, FakeInspireSerial

DATA = Path(__file__).parent / "data"


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> FakeClock:
    """Simulated time shared by the fake hand and the routines, so motion tests run instantly."""
    fake = FakeClock()
    monkeypatch.setattr(routines, "time", fake)
    real_cls = sim_serial.FakeInspireSerial

    def make(slave_id: int = 1, latency: bool = False, clock=None) -> FakeInspireSerial:
        return real_cls(slave_id, latency=True, clock=clock or fake)

    monkeypatch.setattr(sim_serial, "FakeInspireSerial", make)
    return fake


@pytest.fixture
def hand(clock: FakeClock) -> InspireHand:
    return InspireHand(ModbusRTU(FakeInspireSerial(latency=True, clock=clock)))


class StillCamera:
    """Stands in for the webcam: returns the same image at ~30 fps."""

    def __init__(self, image):
        self.image, self.seq = image, 0

    def read(self, last_seq: int = 0, timeout: float = 1.0):
        time.sleep(1 / 30)
        self.seq += 1
        return self.image.copy(), time.monotonic(), self.seq

    def close(self) -> None:
        pass


@pytest.fixture
def still_camera(monkeypatch: pytest.MonkeyPatch):
    """`still_camera("victory")` makes every Camera return that test image."""
    import cv2

    from inspire_teleop import session

    def use(name: str) -> None:
        image = cv2.imread(str(DATA / f"{name}.jpg"))
        assert image is not None, name
        monkeypatch.setattr(session, "Camera", lambda index=0: StillCamera(image))

    return use


def wait_until(condition: Callable[[], bool], timeout: float = 5.0) -> bool:
    """Poll `condition` instead of sleeping a fixed time; robust on slow CI machines."""
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if condition():
            return True
        time.sleep(0.01)
    return condition()
