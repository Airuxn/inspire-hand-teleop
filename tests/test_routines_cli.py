import csv

import pytest

from inspire_teleop import cli
from inspire_teleop.driver import ModbusError, describe_errors
from inspire_teleop.routines import finger_test, step_response, wait_settled


def test_wait_settled_returns_once_motion_stops(hand, clock):
    hand.set_speeds(1000)
    hand.set_angles([0] * 6)
    elapsed = wait_settled(hand)
    assert hand.get_angles() == [0] * 6
    assert 0.8 <= elapsed < 1.5  # full stroke takes 0.8 s at speed 1000


def test_finger_test_moves_each_actuator_in_order(hand):
    lines: list[str] = []
    finger_test(hand, depth=400, report=lines.append)
    assert [line.split()[0] for line in lines] == [
        "little",
        "ring",
        "middle",
        "index",
        "thumb_bend",
        "thumb_rotate",
    ]
    assert all("measured  400" in line for line in lines)
    assert hand.get_angles() == [1000] * 6


def test_finger_test_can_be_stopped(hand):
    lines: list[str] = []
    finger_test(hand, report=lines.append, should_stop=lambda: len(lines) >= 2)
    assert len(lines) == 2


def test_step_response_measures_simulated_dynamics(hand):
    hand.set_speeds(1000)
    close, reopen = step_response(hand, "index", low=200, duration=1.5)
    assert (close.start, close.target) == (1000, 200)
    assert (reopen.start, reopen.target) == (200, 1000)
    # 800 units at 1250 units/s: 10-90 % takes 0.8 * 0.64 s
    assert close.rise_time == pytest.approx(0.512, abs=0.01)
    assert close.dead_time is not None and close.dead_time < 0.02
    assert close.samples[-1][1] == 200
    assert "dead time" in close.summary()


def test_step_result_without_motion_reports_na(hand):
    (result,) = step_response(hand, "index", low=1000, duration=0.05)[:1]
    assert result.rise_time is None
    assert "n/a" in result.summary()


def test_describe_errors():
    assert describe_errors(0) == "ok"
    assert describe_errors(0x01 | 0x04) == "locked rotor, over current"


def test_cli_check_mock(clock, capsys):
    cli.main(["check", "--mock"])
    out = capsys.readouterr().out
    assert "hand id: 1" in out
    assert out.count("error: ok") == 6


def test_cli_check_mock_clear_errors(clock, capsys):
    cli.main(["check", "--mock", "--clear-errors"])
    assert "Errors cleared." in capsys.readouterr().out


def test_cli_wiggle_mock(clock, capsys):
    cli.main(["wiggle", "--mock"])
    out = capsys.readouterr().out
    assert out.count("commanded  400 -> measured  400") == 6


def test_cli_step_mock_writes_csv(clock, tmp_path, capsys):
    out_file = tmp_path / "step.csv"
    cli.main(["step", "--mock", "--finger", "middle", "--output", str(out_file)])
    rows = list(csv.DictReader(out_file.open()))
    assert {row["target"] for row in rows} == {"200", "1000"}
    assert "step 1000 ->  200" in capsys.readouterr().out


def test_cli_ports_lists_without_crashing(capsys):
    cli.main(["ports"])
    assert capsys.readouterr().out


def test_cli_without_adapter_exits_with_hint(monkeypatch):
    monkeypatch.setattr(cli, "find_ch340_ports", lambda: [])
    with pytest.raises(SystemExit, match="No CH340"):
        cli.main(["check"])


def test_cli_reports_no_reply_with_checklist(monkeypatch):
    def unpowered_hand(args, set_limits=True):
        raise ModbusError("timeout: expected 2 bytes, got 0")

    monkeypatch.setattr(cli, "_open_hand", unpowered_hand)
    with pytest.raises(SystemExit, match="power supply"):
        cli.main(["check", "--mock"])


def test_parser_defaults():
    args = cli.build_parser().parse_args(["wiggle"])
    assert (args.speed, args.force, args.depth) == (300, 300, 400)
    args = cli.build_parser().parse_args(["run", "--mock"])
    assert args.rate == 60.0 and args.mock
