"""Smoke test of the Qt app, headless: drives the window like a user would."""

import os
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")

import inspire_teleop.app as app_module  # noqa: E402
from inspire_teleop.driver import FINGERS  # noqa: E402

OPEN, CLOSED = 700, 300


@pytest.fixture(scope="module")
def qapp():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    app_module._dark_palette(app)
    return app


@pytest.fixture
def window(qapp, still_camera, clock, tmp_path, monkeypatch):
    still_camera("victory")
    monkeypatch.setattr(app_module, "list_serial_ports", lambda: [])
    win = app_module.MainWindow(tmp_path / "calib.json", camera_index=0)
    win.show()
    yield win
    win.close()


def pump(qapp, condition, timeout: float = 10.0) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        qapp.processEvents()
        if condition():
            return True
        time.sleep(0.01)
    return condition()


def log_text(win) -> str:
    return win.log_view.toPlainText()


def test_full_user_flow(qapp, window, tmp_path, monkeypatch):
    win = window
    assert pump(qapp, lambda: win.session.snapshot().hand_detected)
    assert win.mock_check.isChecked(), "no serial ports -> simulated hand preselected"
    assert not win.teleop_btn.isEnabled()

    win.connect_btn.click()
    assert pump(qapp, lambda: win.teleop_btn.isEnabled())
    assert "Connected to hand id 1 (simulated)" in log_text(win)

    win.test_btn.click()
    assert pump(qapp, lambda: "Routine finished." in log_text(win))
    assert log_text(win).count("commanded  400 -> measured  400") == 6

    win.step_btn.click()
    assert pump(qapp, lambda: log_text(win).count("Routine finished.") == 2)
    assert "step 1000 ->  200" in log_text(win)

    assert pump(qapp, lambda: win.calib_open_btn.isEnabled())
    win.calib_open_btn.click()
    assert pump(qapp, lambda: "open" in win.session.recorded_poses())
    assert pump(qapp, lambda: win.calib_fist_btn.isEnabled())
    win.calib_fist_btn.click()
    assert pump(qapp, lambda: win.session.calibration_ready())
    assert pump(qapp, lambda: win.calib_save_btn.isEnabled())
    win.calib_save_btn.click()
    assert "looked almost the same" in log_text(win), "same image for both poses is rejected"
    assert (tmp_path / "calib.json").exists()

    log_file = tmp_path / "log.csv"
    monkeypatch.setattr(
        app_module.QFileDialog, "getSaveFileName", lambda *a, **k: (str(log_file), "")
    )
    win.log_check.setChecked(True)

    win.teleop_btn.click()
    index, ring = FINGERS.index("index"), FINGERS.index("ring")
    assert pump(
        qapp,
        lambda: (
            win.session.hand.get_angles()[index] > OPEN
            and win.session.hand.get_angles()[ring] < CLOSED
        ),
    )
    assert "Pause" in win.teleop_btn.text()

    win.smooth_slider.setValue(100)
    assert win.session.filter.min_cutoff == pytest.approx(0.3)
    win.speed_slider.setValue(500)
    win._apply_limits()
    assert win.session.hand.bus.read_registers(1522, 6) == [500] * 6

    win.open_btn.click()
    assert not win.session.teleop_on
    win.log_check.setChecked(False)
    assert len(log_file.read_text().splitlines()) > 1

    win.clear_btn.click()
    assert "Error flags cleared." in log_text(win)

    win.connect_btn.click()
    assert pump(qapp, lambda: win.connect_btn.text() == "Connect")
    win.camera_btn.click()
    assert pump(qapp, lambda: "Camera off" in win.camera_status.text())
    win.grab()  # renders bars, plot and video view


def test_connect_failure_is_reported(qapp, window, monkeypatch):
    win = window
    win.mock_check.setChecked(False)

    def unpowered(*args, **kwargs):
        raise app_module.ModbusError("timeout: expected 2 bytes, got 0")

    monkeypatch.setattr(win.session, "connect", unpowered)
    win.connect_btn.click()
    assert "power supply" in log_text(win)
