"""Graphical app: connect, test, calibrate and teleoperate the hand from one window."""

from __future__ import annotations

import glob
import re
import sys
import time
import traceback
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QObject, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QIcon, QImage, QPainter, QPalette, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from .driver import (
    FINGERS,
    NO_REPLY_HINT,
    NUM_DOF,
    ModbusError,
    describe_errors,
    find_ch340_ports,
    list_serial_ports,
)
from .routines import finger_test, step_response
from .session import CALIBRATION_POSES, CALIBRATION_SAMPLES, Snapshot, TeleopSession

ORANGE = QColor(255, 170, 0)
GREEN = QColor(80, 220, 100)
RED = QColor(230, 80, 70)
MUTED = "#9aa0a6"
FINGER_LABELS = ("Little", "Ring", "Middle", "Index", "Thumb\nbend", "Thumb\nrotate")
HISTORY_S = 10.0


class Bridge(QObject):
    """Carries messages from worker threads to the GUI thread."""

    message = Signal(str)
    routine_done = Signal(str)


class VideoView(QLabel):
    def __init__(self):
        super().__init__("Camera off")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(480, 360)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setStyleSheet(f"background:#111; color:{MUTED}; font-size:16px; border-radius:6px;")

    def show_frame(self, frame: np.ndarray | None) -> None:
        if frame is None:
            self.setPixmap(QPixmap())
            self.setText("Camera off")
            return
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        image = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
        self.setPixmap(
            QPixmap.fromImage(image).scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )


class FingerBars(QWidget):
    """Per-actuator closure: filled bar = webcam target, line = measured on the hand."""

    def __init__(self):
        super().__init__()
        self.target: np.ndarray | None = None
        self.actual = np.full(NUM_DOF, np.nan)
        self.forces = np.full(NUM_DOF, np.nan)
        self.setMinimumSize(360, 220)

    def update_values(self, snap: Snapshot) -> None:
        self.target, self.actual, self.forces = snap.target, snap.actual, snap.forces
        self.update()

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        top, bottom = 44, h - 40
        slot = w / NUM_DOF
        bar_w = min(44.0, slot * 0.55)
        p.setPen(QColor(MUTED))
        p.setFont(QFont(self.font().family(), 8))
        p.drawText(
            QRectF(0, 0, w, 18),
            Qt.AlignmentFlag.AlignLeft,
            "Closure per actuator   (orange = webcam target, green = measured, F = force)",
        )
        for i in range(NUM_DOF):
            x = slot * i + (slot - bar_w) / 2
            frame = QRectF(x, top, bar_w, bottom - top)
            p.setPen(QColor(70, 70, 70))
            p.setBrush(QColor(30, 30, 30))
            p.drawRoundedRect(frame, 4, 4)
            if self.target is not None:
                closure = 1 - self.target[i] / 1000
                fill = QRectF(x, bottom - closure * (bottom - top), bar_w, closure * (bottom - top))
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(ORANGE)
                p.drawRoundedRect(fill, 4, 4)
            if not np.isnan(self.actual[i]):
                y = bottom - (1 - self.actual[i] / 1000) * (bottom - top)
                p.setPen(pg.mkPen(GREEN, width=3))
                p.drawLine(int(x - 5), int(y), int(x + bar_w + 5), int(y))
            p.setPen(QColor(220, 220, 220))
            label_rect = QRectF(slot * i, bottom + 4, slot, 34)
            p.drawText(label_rect, Qt.AlignmentFlag.AlignHCenter, FINGER_LABELS[i])
            if not np.isnan(self.forces[i]):
                p.setPen(QColor(MUTED))
                p.drawText(
                    QRectF(slot * i, 22, slot, 18),
                    Qt.AlignmentFlag.AlignHCenter,
                    f"F {self.forces[i]:.0f}",
                )
        p.end()


def _dot(color: str) -> str:
    return f"<span style='color:{color}; font-size:14px'>●</span>"


def _cameras() -> list[int]:
    indices = sorted(
        int(m.group(1)) for d in glob.glob("/dev/video*") if (m := re.search(r"(\d+)$", d))
    )
    return indices or [0]


class MainWindow(QMainWindow):
    def __init__(self, calibration_path: Path, camera_index: int):
        super().__init__()
        self.setWindowTitle("Inspire Hand Teleop")
        self.resize(1400, 860)
        self.session = TeleopSession(calibration_path)
        self.bridge = Bridge()
        self.bridge.message.connect(self.log)
        self.bridge.routine_done.connect(self._routine_done)
        self.history: deque[tuple[float, float, float]] = deque()
        self._last_snapshot = Snapshot()

        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.addLayout(self._build_main_area(), stretch=1)
        root.addWidget(self._build_sidebar())

        self._refresh_ports()
        self.camera_combo.setCurrentText(str(camera_index))
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(33)
        self.log("Welcome. Follow steps 1-5 on the right, top to bottom.")
        self._toggle_camera()

    # layout ---------------------------------------------------------------

    def _build_main_area(self) -> QVBoxLayout:
        layout = QVBoxLayout()
        self.video = VideoView()
        layout.addWidget(self.video, stretch=3)

        bottom = QHBoxLayout()
        self.bars = FingerBars()
        bottom.addWidget(self.bars, stretch=1)

        plot_box = QVBoxLayout()
        header = QHBoxLayout()
        header.addWidget(QLabel("History"))
        self.plot_finger = QComboBox()
        self.plot_finger.addItems([f.replace("_", " ") for f in FINGERS])
        self.plot_finger.setCurrentIndex(FINGERS.index("index"))
        self.plot_finger.currentIndexChanged.connect(lambda _: self.history.clear())
        header.addWidget(self.plot_finger)
        header.addStretch()
        plot_box.addLayout(header)
        self.plot = pg.PlotWidget()
        self.plot.setBackground("#151515")
        self.plot.setYRange(0, 1000)
        self.plot.setXRange(-HISTORY_S, 0)
        self.plot.setLabel("left", "angle (0 closed, 1000 open)")
        self.plot.setLabel("bottom", "seconds")
        self.plot.addLegend(offset=(10, 10))
        self.curve_target = self.plot.plot(pen=pg.mkPen(ORANGE, width=2), name="webcam target")
        self.curve_actual = self.plot.plot(pen=pg.mkPen(GREEN, width=2), name="measured on hand")
        plot_box.addWidget(self.plot)
        bottom.addLayout(plot_box, stretch=1)
        layout.addLayout(bottom, stretch=2)
        return layout

    def _build_sidebar(self) -> QScrollArea:
        panel = QWidget()
        col = QVBoxLayout(panel)
        col.setSpacing(10)

        # 1. hand
        box = QGroupBox("1  Connect the hand")
        grid = QGridLayout(box)
        self.port_combo = QComboBox()
        refresh = QPushButton("↻")
        refresh.setFixedWidth(34)
        refresh.setToolTip("Rescan serial ports")
        refresh.clicked.connect(self._refresh_ports)
        self.mock_check = QCheckBox("Simulated hand (no hardware)")
        self.mock_check.toggled.connect(lambda on: self.port_combo.setEnabled(not on))
        self.connect_btn = QPushButton("Connect")
        self.connect_btn.clicked.connect(self._toggle_connection)
        self.hand_status = QLabel()
        grid.addWidget(QLabel("Port"), 0, 0)
        grid.addWidget(self.port_combo, 0, 1)
        grid.addWidget(refresh, 0, 2)
        grid.addWidget(self.mock_check, 1, 0, 1, 3)
        grid.addWidget(self.connect_btn, 2, 0, 1, 3)
        grid.addWidget(self.hand_status, 3, 0, 1, 3)
        col.addWidget(box)

        # 2. camera
        box = QGroupBox("2  Camera")
        grid = QGridLayout(box)
        self.camera_combo = QComboBox()
        self.camera_combo.addItems([str(i) for i in _cameras()])
        self.camera_btn = QPushButton("Start camera")
        self.camera_btn.clicked.connect(self._toggle_camera)
        self.camera_status = QLabel()
        grid.addWidget(QLabel("Camera"), 0, 0)
        grid.addWidget(self.camera_combo, 0, 1)
        grid.addWidget(self.camera_btn, 1, 0, 1, 2)
        grid.addWidget(self.camera_status, 2, 0, 1, 2)
        col.addWidget(box)

        # 3. test
        box = QGroupBox("3  Test the hand")
        v = QVBoxLayout(box)
        hint = QLabel("Moves each finger slowly, one at a time. Check order and direction.")
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color:{MUTED}")
        self.test_btn = QPushButton("Test fingers one by one")
        self.test_btn.clicked.connect(self._finger_test)
        self.step_btn = QPushButton("Measure response time (index)")
        self.step_btn.clicked.connect(self._step_test)
        self.clear_btn = QPushButton("Clear error flags")
        self.clear_btn.clicked.connect(self._clear_errors)
        for w in (hint, self.test_btn, self.step_btn, self.clear_btn):
            v.addWidget(w)
        col.addWidget(box)

        # 4. calibrate
        box = QGroupBox("4  Calibrate to your hand")
        v = QVBoxLayout(box)
        self.calib_hint = QLabel()
        self.calib_hint.setWordWrap(True)
        self.calib_hint.setStyleSheet(f"color:{MUTED}")
        row = QHBoxLayout()
        self.calib_open_btn = QPushButton("Record open hand")
        self.calib_open_btn.setToolTip(CALIBRATION_POSES["open"])
        self.calib_open_btn.clicked.connect(lambda: self._record_pose("open"))
        self.calib_fist_btn = QPushButton("Record fist")
        self.calib_fist_btn.setToolTip(CALIBRATION_POSES["closed"])
        self.calib_fist_btn.clicked.connect(lambda: self._record_pose("closed"))
        row.addWidget(self.calib_open_btn)
        row.addWidget(self.calib_fist_btn)
        self.calib_progress = QProgressBar()
        self.calib_progress.setRange(0, CALIBRATION_SAMPLES)
        self.calib_progress.setTextVisible(False)
        self.calib_progress.setFixedHeight(8)
        self.calib_save_btn = QPushButton("Save calibration")
        self.calib_save_btn.clicked.connect(self._save_calibration)
        self.calib_status = QLabel()
        v.addWidget(self.calib_hint)
        v.addLayout(row)
        v.addWidget(self.calib_progress)
        v.addWidget(self.calib_save_btn)
        v.addWidget(self.calib_status)
        col.addWidget(box)

        # 5. teleoperate
        box = QGroupBox("5  Teleoperate")
        v = QVBoxLayout(box)
        self.teleop_btn = QPushButton("▶  Start teleoperation")
        self.teleop_btn.setMinimumHeight(48)
        self.teleop_btn.clicked.connect(self._toggle_teleop)
        self.open_btn = QPushButton("Open hand (stops teleoperation)")
        self.open_btn.clicked.connect(self._open_hand)
        v.addWidget(self.teleop_btn)
        v.addWidget(self.open_btn)
        grid = QGridLayout()
        self.speed_slider = self._slider(grid, 0, "Speed", 100, 1000, self.session.speed)
        self.force_slider = self._slider(grid, 1, "Grip force", 100, 1000, self.session.force)
        self.smooth_slider = self._slider(grid, 2, "Smoothing", 0, 100, 46)
        self.speed_slider.sliderReleased.connect(self._apply_limits)
        self.force_slider.sliderReleased.connect(self._apply_limits)
        self.smooth_slider.valueChanged.connect(self._apply_smoothing)
        self._apply_smoothing(self.smooth_slider.value())
        v.addLayout(grid)
        self.log_check = QCheckBox("Log targets and measurements to CSV")
        self.log_check.toggled.connect(self._toggle_logging)
        v.addWidget(self.log_check)
        col.addWidget(box)

        # status + log
        self.stats = QLabel()
        self.stats.setStyleSheet("font-family: monospace;")
        col.addWidget(self.stats)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMinimumHeight(140)
        self.log_view.setStyleSheet("font-family: monospace; font-size: 11px;")
        col.addWidget(self.log_view, stretch=1)

        scroll = QScrollArea()
        scroll.setWidget(panel)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scrollbar_w = scroll.verticalScrollBar().sizeHint().width()
        scroll.setFixedWidth(max(400, panel.sizeHint().width() + scrollbar_w + 8))
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        return scroll

    def _slider(self, grid: QGridLayout, row: int, name: str, lo: int, hi: int, value: int):
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(lo, hi)
        slider.setValue(value)
        label = QLabel(str(value))
        label.setFixedWidth(36)
        slider.valueChanged.connect(lambda v: label.setText(str(v)))
        grid.addWidget(QLabel(name), row, 0)
        grid.addWidget(slider, row, 1)
        grid.addWidget(label, row, 2)
        return slider

    # actions --------------------------------------------------------------

    def log(self, text: str) -> None:
        self.log_view.appendPlainText(f"{time.strftime('%H:%M:%S')}  {text}")

    def _refresh_ports(self) -> None:
        ch340 = set(find_ch340_ports())
        ports = sorted(list_serial_ports(), key=lambda p: p.device not in ch340)
        self.port_combo.clear()
        for p in ports:
            label = f"{p.device}  (Inspire adapter)" if p.device in ch340 else p.device
            self.port_combo.addItem(label, p.device)
        if not ports:
            self.port_combo.addItem("no serial ports found", None)
            self.mock_check.setChecked(True)

    def _toggle_connection(self) -> None:
        if self.session.hand is not None:
            self.session.disconnect()
            self.log("Disconnected (hand opened).")
            return
        mock = self.mock_check.isChecked()
        port = None if mock else self.port_combo.currentData()
        try:
            hand_id = self.session.connect(port, mock)
        except ModbusError as exc:
            self.log(f"Connection failed: {exc}")
            if "got 0" in str(exc):
                self.log(NO_REPLY_HINT)
            return
        except (ValueError, OSError) as exc:
            self.log(f"Connection failed: {exc}")
            return
        except Exception as exc:  # serial.SerialException (permissions, unplugged)
            self.log(f"Cannot open {port}: {exc}")
            return
        self.log(f"Connected to hand id {hand_id} ({'simulated' if mock else port}).")
        self._log_errors()

    def _log_errors(self) -> None:
        hand = self.session.hand
        if hand is None:
            return
        try:
            errors = hand.get_errors()
            temps = hand.get_temperatures()
        except ModbusError as exc:
            self.log(f"Could not read status: {exc}")
            return
        bad = [f"{f}: {describe_errors(e)}" for f, e in zip(FINGERS, errors, strict=True) if e]
        self.log("Error flags: " + ("; ".join(bad) if bad else "none"))
        self.log("Temperatures (C): " + ", ".join(str(t) for t in temps))

    def _toggle_camera(self) -> None:
        if self.session.camera is not None:
            self.session.stop_camera()
            self.log("Camera stopped.")
            return
        index = int(self.camera_combo.currentText())
        try:
            self.session.start_camera(index)
        except RuntimeError as exc:
            self.log(f"Camera error: {exc}")
            return
        self.log(f"Camera {index} started.")

    def _finger_test(self) -> None:
        self.log("Finger test: each actuator closes to 400 and reopens.")
        self.session.set_motion_limits(300, 300)
        self.session.run_routine(
            "finger test",
            lambda hand: finger_test(hand, 400, report=self.bridge.message.emit),
            on_done=lambda exc: self.bridge.routine_done.emit(str(exc) if exc else ""),
        )

    def _step_test(self) -> None:
        self.log("Measuring step response of the index finger...")
        self.session.set_motion_limits(self.speed_slider.value(), self.force_slider.value())

        def routine(hand) -> None:
            for result in step_response(hand, "index"):
                self.bridge.message.emit(result.summary())

        self.session.run_routine(
            "response test",
            routine,
            on_done=lambda exc: self.bridge.routine_done.emit(str(exc) if exc else ""),
        )

    def _routine_done(self, error: str) -> None:
        self.log(f"Routine failed: {error}" if error else "Routine finished.")
        try:
            self._apply_limits()
        except ModbusError:
            pass

    def _clear_errors(self) -> None:
        try:
            self.session.clear_errors()
            self.log("Error flags cleared.")
            self._log_errors()
        except ModbusError as exc:
            self.log(f"Clear failed: {exc}")

    def _record_pose(self, pose: str) -> None:
        self.session.start_calibration_pose(pose)
        self.log(f"Recording '{pose}': {CALIBRATION_POSES[pose]}")

    def _save_calibration(self) -> None:
        rejected = self.session.save_calibration()
        self.log(f"Calibration saved to {self.session.calibration_path}.")
        if rejected:
            self.log(
                "Open and fist looked almost the same for: "
                + ", ".join(rejected)
                + ". Kept their previous calibration; record both poses again more clearly."
            )

    def _toggle_teleop(self) -> None:
        self.session.set_teleop(not self.session.teleop_on)
        self.log("Teleoperation started." if self.session.teleop_on else "Teleoperation paused.")

    def _open_hand(self) -> None:
        try:
            self.session.open_hand()
            self.log("Hand opened; teleoperation stopped.")
        except ModbusError as exc:
            self.log(f"Open failed: {exc}")

    def _apply_limits(self) -> None:
        try:
            self.session.set_motion_limits(self.speed_slider.value(), self.force_slider.value())
        except ModbusError as exc:
            self.log(f"Could not set speed/force: {exc}")

    def _apply_smoothing(self, value: int) -> None:
        """0 = 6 Hz cutoff (raw, jittery) ... 100 = 0.3 Hz (very smooth, laggy); log scale."""
        self.session.filter.min_cutoff = 0.3 * 20 ** (1 - value / 100)

    def _toggle_logging(self, on: bool) -> None:
        if not on:
            self.session.set_logging(None)
            self.log("CSV logging stopped.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Log to CSV", f"teleop_{time.strftime('%Y%m%d_%H%M%S')}.csv", "CSV (*.csv)"
        )
        if not path:
            self.log_check.setChecked(False)
            return
        self.session.set_logging(Path(path))
        self.log(f"Logging to {path}")

    # refresh --------------------------------------------------------------

    def _tick(self) -> None:
        snap = self.session.snapshot()
        if snap.frame is not self._last_snapshot.frame or not snap.camera_on:
            self.video.show_frame(snap.frame if snap.camera_on else None)
        self._last_snapshot = snap
        self.bars.update_values(snap)
        self._update_plot(snap)
        self._update_controls(snap)

    def _update_plot(self, snap: Snapshot) -> None:
        i = self.plot_finger.currentIndex()
        now = time.monotonic()
        target = snap.target[i] if snap.target is not None else np.nan
        self.history.append((now, target, snap.actual[i]))
        while self.history and now - self.history[0][0] > HISTORY_S:
            self.history.popleft()
        data = np.array(self.history)
        t = data[:, 0] - now
        self.curve_target.setData(t, data[:, 1], connect="finite")
        self.curve_actual.setData(t, data[:, 2], connect="finite")

    def _update_controls(self, snap: Snapshot) -> None:
        connected, busy = snap.connected, bool(snap.busy)
        if connected:
            source = "simulated hand" if snap.mock else "hand"
            self.hand_status.setText(f"{_dot('#50dc64')} Connected to {source}")
        else:
            self.hand_status.setText(f"{_dot(MUTED)} Not connected")
        self.connect_btn.setText("Disconnect" if connected else "Connect")
        self.port_combo.setEnabled(not connected and not self.mock_check.isChecked())
        self.mock_check.setEnabled(not connected)

        self.camera_btn.setText("Stop camera" if snap.camera_on else "Start camera")
        self.camera_combo.setEnabled(not snap.camera_on)
        if not snap.camera_on:
            self.camera_status.setText(f"{_dot(MUTED)} Camera off")
        elif snap.hand_detected:
            self.camera_status.setText(
                f"{_dot('#50dc64')} {snap.handedness} hand tracked · {snap.fps:.0f} fps"
            )
        else:
            self.camera_status.setText(f"{_dot('#ffaa00')} No hand in view · {snap.fps:.0f} fps")

        for btn in (self.test_btn, self.step_btn, self.clear_btn, self.open_btn):
            btn.setEnabled(connected and not busy)

        recorded = self.session.recorded_poses()
        if snap.calibrating:
            self.calib_hint.setText(f"Hold still: {CALIBRATION_POSES[snap.calibrating]}")
        else:
            self.calib_hint.setText(
                "Show one hand to the camera, hold the pose, then press the button. "
                "Hover a button for the exact pose."
            )
        self.calib_progress.setValue(snap.calib_progress if snap.calibrating else 0)
        can_record = snap.camera_on and snap.hand_detected and not snap.calibrating
        self.calib_open_btn.setEnabled(can_record)
        self.calib_fist_btn.setEnabled(can_record)
        self.calib_open_btn.setText("Open hand ✓" if "open" in recorded else "Record open hand")
        self.calib_fist_btn.setText("Fist ✓" if "closed" in recorded else "Record fist")
        self.calib_save_btn.setEnabled(self.session.calibration_ready())
        self.calib_status.setText(
            f"{_dot('#50dc64')} Calibrated"
            if snap.calibrated
            else f"{_dot('#ffaa00')} Using default calibration"
        )

        can_teleop = connected and snap.camera_on and not busy
        self.teleop_btn.setEnabled(can_teleop)
        style = "font-size:15px; font-weight:bold;"
        if busy:
            self.teleop_btn.setText(f"Busy: {snap.busy}…")
        elif snap.teleop_on:
            self.teleop_btn.setText("⏸  Pause teleoperation")
            style += "background:#b3412f;"
        elif can_teleop:
            self.teleop_btn.setText("▶  Start teleoperation")
            style += "background:#2e7d46;"
        else:
            self.teleop_btn.setText("Connect hand and camera first")
        self.teleop_btn.setStyleSheet(style)

        def fmt(v: float) -> str:
            return "   –" if np.isnan(v) else f"{v:5.1f}"

        self.stats.setText(
            f"modbus write    {fmt(snap.write_ms)} ms\n"
            f"camera → hand   {fmt(snap.pipeline_ms)} ms\n"
            f"bus errors      {snap.bus_errors}"
            + (f"\nlast error: {snap.last_error[:40]}" if snap.last_error else "")
        )

    def closeEvent(self, event) -> None:
        self.timer.stop()
        self.session.close()
        super().closeEvent(event)


def _dark_palette(app: QApplication) -> None:
    app.setStyle("Fusion")
    palette = QPalette()
    for role, color in {
        QPalette.ColorRole.Window: "#1e1f22",
        QPalette.ColorRole.WindowText: "#e8e8e8",
        QPalette.ColorRole.Base: "#151515",
        QPalette.ColorRole.AlternateBase: "#2a2b2e",
        QPalette.ColorRole.Text: "#e8e8e8",
        QPalette.ColorRole.Button: "#2c2e33",
        QPalette.ColorRole.ButtonText: "#e8e8e8",
        QPalette.ColorRole.Highlight: "#ffaa00",
        QPalette.ColorRole.HighlightedText: "#000000",
        QPalette.ColorRole.ToolTipBase: "#2c2e33",
        QPalette.ColorRole.ToolTipText: "#e8e8e8",
    }.items():
        palette.setColor(role, QColor(color))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, QColor("#666"))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, QColor("#666"))
    app.setPalette(palette)
    app.setStyleSheet(
        "QGroupBox { font-weight: bold; border: 1px solid #3a3c42; border-radius: 6px;"
        " margin-top: 10px; padding: 10px 8px 8px 8px; }"
        "QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; color: #ffaa00; }"
        "QPushButton { padding: 6px 10px; border-radius: 4px; background: #2c2e33;"
        " border: 1px solid #4a4d55; }"
        "QPushButton:hover { background: #363940; border-color: #ffaa00; }"
        "QPushButton:disabled { color: #666; border-color: #33353a; background: #25262a; }"
    )


ICON_PATH = Path(__file__).parent / "assets" / "icon.svg"
CRASH_LOG = Path.home() / ".cache" / "inspire-hand-teleop" / "app.log"


def _report_unhandled(exc_type, exc, tb) -> None:
    """Without a terminal (desktop launcher) uncaught errors would vanish; log and show them."""
    text = "".join(traceback.format_exception(exc_type, exc, tb))
    CRASH_LOG.parent.mkdir(parents=True, exist_ok=True)
    with CRASH_LOG.open("a") as f:
        f.write(f"--- {time.strftime('%Y-%m-%d %H:%M:%S')}\n{text}\n")
    if sys.__stderr__ is not None:
        sys.__stderr__.write(text)
    QMessageBox.critical(None, "Inspire Hand Teleop", f"{exc}\n\nDetails in {CRASH_LOG}")


def run_app(calibration_path: Path, camera_index: int = 0) -> None:
    QApplication.setApplicationName("inspire-hand-teleop")
    QApplication.setDesktopFileName("inspire-hand-teleop")
    existing = QApplication.instance()
    app = existing if isinstance(existing, QApplication) else QApplication(sys.argv)
    app.setWindowIcon(QIcon(str(ICON_PATH)))
    sys.excepthook = _report_unhandled
    _dark_palette(app)
    window = MainWindow(calibration_path, camera_index)
    window.show()
    app.exec()
