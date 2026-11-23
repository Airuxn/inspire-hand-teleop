import time

import pytest

from inspire_teleop.driver import InspireHand, ModbusError, ModbusRTU, Reg, crc16
from inspire_teleop.sim_serial import FakeInspireSerial


def make_hand() -> tuple[InspireHand, FakeInspireSerial]:
    ser = FakeInspireSerial()
    return InspireHand(ModbusRTU(ser)), ser


def test_crc16_known_vector():
    assert crc16(bytes.fromhex("010300000001")) == bytes.fromhex("840a")


def test_set_angles_writes_angle_registers():
    hand, ser = make_hand()
    hand.set_angles([0, 100, 200, 300, 400, 500])
    assert [ser.regs[Reg.ANGLE_SET + 2 * i] for i in range(6)] == [0, 100, 200, 300, 400, 500]
    assert hand.get_target_angles() == [0, 100, 200, 300, 400, 500]


def test_angles_are_clamped_and_rounded():
    hand, ser = make_hand()
    hand.set_angles([-50, 1200, 499.6, 0, 0, 0])
    assert hand.get_target_angles()[:3] == [0, 1000, 500]


def test_set_angles_requires_six_values():
    hand, _ = make_hand()
    with pytest.raises(ValueError):
        hand.set_angles([0, 0, 0])


def test_byte_registers_return_one_value_per_actuator():
    hand, ser = make_hand()
    for i in range(6):
        ser.regs[Reg.TEMP + i] = 24 + i
        ser.regs[Reg.STATUS + i] = 0x0200 | 2
    assert hand.get_temperatures() == [24, 25, 26, 27, 28, 29]
    assert hand.get_statuses() == [2] * 6


def test_simulated_actuators_move_toward_target():
    hand, _ = make_hand()
    hand.set_speeds(1000)
    hand.set_angles([0] * 6)
    time.sleep(0.2)
    angles = hand.get_angles()
    assert all(0 < a < 1000 for a in angles)


def test_timeout_raises_modbus_error():
    class Silent(FakeInspireSerial):
        def read(self, size):
            return b""

    hand = InspireHand(ModbusRTU(Silent()))
    with pytest.raises(ModbusError, match="timeout"):
        hand.get_angles()


def test_corrupted_response_raises_crc_error():
    class Corrupt(FakeInspireSerial):
        def read(self, size):
            data = bytearray(super().read(size))
            if len(data) > 2:
                data[2] ^= 0xFF
            return bytes(data)

    hand = InspireHand(ModbusRTU(Corrupt()))
    with pytest.raises(ModbusError, match="CRC"):
        hand.get_angles()


def test_device_exception_is_reported():
    bus = ModbusRTU(FakeInspireSerial())
    with pytest.raises(ModbusError, match="exception"):
        bus._transact(bytes([1, 0x2B, 0, 0]), 8)
