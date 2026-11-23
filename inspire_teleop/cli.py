"""Command-line entry point: `inspire-teleop <command>`."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from .driver import (
    FINGERS,
    NO_REPLY_HINT,
    STATUS_NAMES,
    InspireHand,
    ModbusError,
    ModbusRTU,
    describe_errors,
    find_ch340_ports,
    list_serial_ports,
)
from .routines import finger_test, step_response

DEFAULT_CALIBRATION = Path.home() / ".config" / "inspire-hand-teleop" / "calibration.json"


def _resolve_port(args: argparse.Namespace) -> str | None:
    if args.mock or args.port:
        return args.port
    ports = find_ch340_ports()
    if not ports:
        sys.exit("No CH340 USB-serial adapter found. Plug in the hand or pass --port.")
    return ports[0]


def _open_hand(args: argparse.Namespace, set_limits: bool = True) -> InspireHand:
    if args.mock:
        from .sim_serial import FakeInspireSerial

        print("Using simulated hand (--mock)")
        hand = InspireHand(ModbusRTU(FakeInspireSerial(args.slave_id, latency=True), args.slave_id))
    else:
        port = _resolve_port(args)
        print(f"Connecting to {port} @ {args.baudrate} baud, slave id {args.slave_id}")
        try:
            hand = InspireHand.open(port, args.baudrate, args.slave_id)
        except Exception as exc:  # serial.SerialException has no common base worth importing
            sys.exit(f"Cannot open {port}: {exc}")
    if set_limits:
        hand.set_speeds(args.speed)
        hand.set_forces(args.force)
    return hand


def cmd_ports(args: argparse.Namespace) -> None:
    ch340 = set(find_ch340_ports())
    ports = list_serial_ports()
    for p in ports:
        mark = "  <-- CH340 (Inspire hand adapter)" if p.device in ch340 else ""
        print(f"{p.device:16s} {p.description}{mark}")
    if not ports:
        print("No serial ports found.")


def cmd_check(args: argparse.Namespace) -> None:
    with _open_hand(args, set_limits=False) as hand:
        if args.clear_errors:
            hand.clear_errors()
            print("Errors cleared.")
        print(f"hand id: {hand.hand_id()}")
        rows = {
            "angle": hand.get_angles(),
            "target": hand.get_target_angles(),
            "force": hand.get_forces(),
            "current": hand.get_currents(),
            "temp C": hand.get_temperatures(),
        }
        print(f"{'':10s}" + "".join(f"{f:>13s}" for f in FINGERS))
        for name, values in rows.items():
            print(f"{name:10s}" + "".join(f"{v:>13d}" for v in values))
        statuses = hand.get_statuses()
        errors = hand.get_errors()
        for finger, status, error in zip(FINGERS, statuses, errors, strict=True):
            print(
                f"  {finger:13s} status: {STATUS_NAMES.get(status, status)!s:18s} "
                f"error: {describe_errors(error)}"
            )


def cmd_wiggle(args: argparse.Namespace) -> None:
    with _open_hand(args) as hand:
        finger_test(hand, args.depth)
        print("Done. Each finger should have closed partly and reopened, in the order above.")


def cmd_step(args: argparse.Namespace) -> None:
    """Step response of one actuator: latency and rise time for system identification."""
    with _open_hand(args) as hand:
        results = step_response(hand, args.finger, args.low, args.duration)
    out = Path(args.output)
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_since_command", "target", "angle"])
        for r in results:
            print(r.summary())
            w.writerows((f"{t:.4f}", r.target, a) for t, a in r.samples)
    print(f"Wrote {sum(len(r.samples) for r in results)} samples to {out}")


def _session(args: argparse.Namespace):
    from .filters import OneEuroFilter
    from .session import TeleopSession

    session = TeleopSession(Path(args.calibration), args.which)
    if hasattr(args, "min_cutoff"):
        session.filter = OneEuroFilter(min_cutoff=args.min_cutoff, beta=args.beta)
    return session


def cmd_calibrate(args: argparse.Namespace) -> None:
    from .teleop import run_calibration

    session = _session(args)
    try:
        run_calibration(session, args.camera)
    finally:
        session.close()


def cmd_run(args: argparse.Namespace) -> None:
    from .teleop import run_teleop

    session = _session(args)
    try:
        session.speed, session.force = args.speed, args.force
        session.set_logging(Path(args.log) if args.log else None)
        port = _resolve_port(args)
        hand_id = session.connect(port, args.mock, args.slave_id, args.baudrate, args.rate)
        print(f"Connected to hand id {hand_id} ({'simulated' if args.mock else port})")
        run_teleop(session, args.camera, Path(args.record) if args.record else None)
    finally:
        session.close()


def cmd_app(args: argparse.Namespace) -> None:
    try:
        from .app import run_app
    except ImportError as exc:
        sys.exit(f"The app needs PySide6 and pyqtgraph ({exc}). Install with: uv sync")
    run_app(Path(args.calibration), args.camera)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="inspire-teleop", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def hand_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--port", help="serial port (default: first CH340 adapter found)")
        p.add_argument("--baudrate", type=int, default=115200)
        p.add_argument("--slave-id", type=int, default=1)
        p.add_argument("--mock", action="store_true", help="use a simulated hand")
        p.add_argument("--speed", type=int, default=1000, help="actuator speed 0-1000")
        p.add_argument("--force", type=int, default=500, help="force threshold 0-1000")

    def vision_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--camera", type=int, default=0)
        p.add_argument("--which", choices=["any", "left", "right"], default="any")
        p.add_argument("--calibration", default=str(DEFAULT_CALIBRATION))

    p = sub.add_parser("app", help="graphical app (recommended)")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--calibration", default=str(DEFAULT_CALIBRATION))
    p.set_defaults(func=cmd_app)

    p = sub.add_parser("ports", help="list serial ports")
    p.set_defaults(func=cmd_ports)

    p = sub.add_parser("check", help="read id, angles, forces, temperatures and errors")
    hand_args(p)
    p.add_argument("--clear-errors", action="store_true")
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("wiggle", help="move each actuator in turn to verify order and direction")
    hand_args(p)
    p.add_argument("--depth", type=int, default=400, help="angle to close to (0 = fully closed)")
    p.set_defaults(func=cmd_wiggle, speed=300, force=300)

    p = sub.add_parser("step", help="measure step response latency of one actuator")
    hand_args(p)
    p.add_argument("--finger", choices=FINGERS, default="index")
    p.add_argument("--low", type=int, default=200)
    p.add_argument("--duration", type=float, default=1.5)
    p.add_argument("--output", default="step_response.csv")
    p.set_defaults(func=cmd_step)

    p = sub.add_parser("calibrate", help="record open-hand and fist poses from the webcam")
    vision_args(p)
    p.set_defaults(func=cmd_calibrate)

    p = sub.add_parser("run", help="webcam teleoperation in a minimal keyboard-driven window")
    hand_args(p)
    vision_args(p)
    p.add_argument("--rate", type=float, default=60.0, help="command rate to the hand in Hz")
    p.add_argument("--log", help="CSV log of targets, measured angles and forces")
    p.add_argument("--record", help="save the annotated camera view to an .mp4")
    p.add_argument("--min-cutoff", type=float, default=1.5, help="One Euro filter min cutoff")
    p.add_argument("--beta", type=float, default=0.02, help="One Euro filter speed coefficient")
    p.set_defaults(func=cmd_run)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except ModbusError as exc:
        hint = f"\n{NO_REPLY_HINT}" if "got 0" in str(exc) else ""
        sys.exit(f"Hand communication failed: {exc}{hint}")
    except (RuntimeError, TimeoutError) as exc:
        sys.exit(f"Error: {exc}")
    except KeyboardInterrupt:
        print("Interrupted.")


if __name__ == "__main__":
    main()
