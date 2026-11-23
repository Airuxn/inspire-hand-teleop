"""Command-line entry point: `inspire-teleop <command>`."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

from .driver import (
    FINGERS,
    NUM_DOF,
    STATUS_NAMES,
    InspireHand,
    ModbusError,
    ModbusRTU,
    describe_errors,
    find_ch340_ports,
)

DEFAULT_CALIBRATION = Path("calibration.json")


def _open_hand(args: argparse.Namespace) -> InspireHand:
    if args.mock:
        from .sim_serial import FakeInspireSerial

        print("Using simulated hand (--mock)")
        return InspireHand(ModbusRTU(FakeInspireSerial(args.slave_id, latency=True), args.slave_id))
    port = args.port
    if port is None:
        ports = find_ch340_ports()
        if not ports:
            sys.exit("No CH340 USB-serial adapter found. Plug in the hand or pass --port.")
        port = ports[0]
    print(f"Connecting to {port} @ {args.baudrate} baud, slave id {args.slave_id}")
    try:
        return InspireHand.open(port, args.baudrate, args.slave_id)
    except Exception as exc:  # serial.SerialException has no common base worth importing
        sys.exit(f"Cannot open {port}: {exc}")


def cmd_ports(args: argparse.Namespace) -> None:
    from serial.tools import list_ports

    ch340 = set(find_ch340_ports())
    found = False
    for p in list_ports.comports():
        found = True
        mark = "  <-- CH340 (Inspire hand adapter)" if p.device in ch340 else ""
        print(f"{p.device:16s} {p.description}{mark}")
    if not found:
        print("No serial ports found.")


def cmd_check(args: argparse.Namespace) -> None:
    with _open_hand(args) as hand:
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
        hand.set_speeds(args.speed)
        hand.set_forces(args.force)
        hand.open_all()
        _wait_settled(hand)
        for i, finger in enumerate(FINGERS):
            angles = [1000] * NUM_DOF
            angles[i] = args.depth
            hand.set_angles(angles)
            elapsed = _wait_settled(hand)
            print(
                f"{finger:13s} commanded {args.depth:4d} -> measured {hand.get_angles()[i]:4d} "
                f"(settled after {elapsed:.2f} s)"
            )
            hand.open_all()
            _wait_settled(hand)
        print("Done. Each finger should have closed partly and reopened, in the order above.")


def _wait_settled(hand: InspireHand, timeout: float = 4.0, still_for: float = 0.25) -> float:
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


def cmd_step(args: argparse.Namespace) -> None:
    """Step response of one actuator: latency and rise time for system identification."""
    i = FINGERS.index(args.finger)
    rows: list[tuple[float, int, int]] = []
    with _open_hand(args) as hand:
        hand.set_speeds(args.speed)
        hand.set_forces(args.force)
        hand.open_all()
        time.sleep(1.5)
        for target in (args.low, 1000):
            angles = [1000] * NUM_DOF
            angles[i] = target
            start = hand.get_angles()[i]
            seg: list[tuple[float, int, int]] = []
            t0 = time.monotonic()
            hand.set_angles(angles)
            while (t := time.monotonic() - t0) < args.duration:
                seg.append((t, target, hand.get_angles()[i]))
            _report_step(seg, target, start)
            rows += seg
    out = Path(args.output)
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_since_command", "target", "angle"])
        w.writerows((f"{t:.4f}", tgt, a) for t, tgt, a in rows)
    print(f"Wrote {len(rows)} samples to {out}")


def _report_step(seg: list[tuple[float, int, int]], target: int, start: int) -> None:
    if len(seg) < 2:
        print("not enough samples")
        return
    span = target - start
    poll_hz = len(seg) / seg[-1][0]

    def first_time(fraction: float) -> float | None:
        for t, _, a in seg:
            if span and (a - start) / span >= fraction:
                return t
        return None

    dead, t10, t90 = first_time(0.02), first_time(0.1), first_time(0.9)
    fmt = lambda v: f"{v * 1000:6.1f} ms" if v is not None else "   n/a"  # noqa: E731
    print(
        f"step {start:4d} -> {target:4d}: dead time {fmt(dead)}, 10-90% rise "
        f"{fmt(t90 - t10) if t10 is not None and t90 is not None else '   n/a'}, "
        f"final {seg[-1][2]:4d}, polling {poll_hz:.0f} Hz"
    )


def cmd_calibrate(args: argparse.Namespace) -> None:
    from .teleop import run_calibration

    run_calibration(Path(args.output), args.camera, args.which)


def cmd_run(args: argparse.Namespace) -> None:
    from .teleop import run_teleop

    hand = _open_hand(args)
    try:
        run_teleop(
            hand,
            camera_index=args.camera,
            calibration_path=Path(args.calibration),
            rate_hz=args.rate,
            which_hand=args.which,
            speed=args.speed,
            force=args.force,
            log_path=Path(args.log) if args.log else None,
            record_path=Path(args.record) if args.record else None,
            min_cutoff=args.min_cutoff,
            beta=args.beta,
        )
    finally:
        hand.close()


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
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--which", choices=["any", "left", "right"], default="any")
    p.add_argument("--output", default=str(DEFAULT_CALIBRATION))
    p.set_defaults(func=cmd_calibrate)

    p = sub.add_parser("run", help="webcam teleoperation")
    hand_args(p)
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--which", choices=["any", "left", "right"], default="any")
    p.add_argument("--calibration", default=str(DEFAULT_CALIBRATION))
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
        sys.exit(f"Hand communication failed: {exc}")
    except KeyboardInterrupt:
        print("Interrupted.")


if __name__ == "__main__":
    main()
