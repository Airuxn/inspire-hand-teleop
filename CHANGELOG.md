# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- End-to-end vision tests on MediaPipe sample photos (fist, victory, pointing, thumbs-up):
  image to actuator angles.
- Headless smoke test of the Qt app that clicks through the whole workflow.
- Tests for the hardware routines, all CLI commands, the camera thread and the `run` and
  `calibrate` windows.
- Coverage in CI (minimum 85%, report in the run summary) and Pyright type checking.

### Changed

- The emulated hand takes an injectable clock, so motion tests no longer depend on sleeps.
- Driver setters accept any sized collection of numbers (including NumPy arrays).

## [0.2.0] - 2026-11-23

First version tested on a physical Inspire RH56 hand.

### Added

- Qt desktop app (`inspire-teleop app`): camera view, per-actuator target vs. measured bars,
  live history plot, and a step-by-step sidebar for connecting, testing, calibrating and
  teleoperating.
- `TeleopSession`, shared by the app and the command-line window.
- `scripts/install_desktop.sh`: installs the app and adds an application-menu and desktop
  launcher.
- Hardware notes in [`docs/hardware.md`](docs/hardware.md), including measured Modbus latency.
- CI on Python 3.10, 3.12 and 3.13.

### Changed

- Calibration keeps the previous values for actuators whose open and fist poses are too
  similar, instead of amplifying tracking noise into full-range jumps.
- Calibration is stored in `~/.config/inspire-hand-teleop/calibration.json`.
- `check` is read-only. A missing reply prints a power and wiring checklist.

### Fixed

- Error, status and temperature registers are read as one register per actuator. The previous
  byte-splitting, taken from an existing driver, returned values for only three actuators on
  real hardware.
- Disconnecting crashed on Python 3.10–3.12 because the hand link's stop flag shadowed
  `threading.Thread._stop`.

## [0.1.0] - 2026-11-23

### Added

- Modbus RTU driver that reads responses by exact length with CRC checks.
- MediaPipe hand tracking, retargeting to the hand's 6 actuators, two-pose calibration and
  One Euro filtering.
- Fixed-rate hand link thread with deadband and CSV logging.
- Command-line tools: `ports`, `check`, `wiggle`, `calibrate`, `run`, `step`.
- Simulated hand (`--mock`) that runs the real driver over an emulated serial port.

[0.2.0]: https://github.com/Airuxn/inspire-hand-teleop/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/Airuxn/inspire-hand-teleop/releases/tag/v0.1.0
