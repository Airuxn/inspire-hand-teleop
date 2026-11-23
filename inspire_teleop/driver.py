"""Low-latency Modbus RTU driver for the Inspire RH56 dexterous hand.

Responses are read by their exact expected length instead of sleeping a fixed
interval, so a full write/read round trip takes a few milliseconds at 115200 baud.
"""

from __future__ import annotations

import struct
import threading
from collections.abc import Sequence
from typing import Protocol

FINGERS = ("little", "ring", "middle", "index", "thumb_bend", "thumb_rotate")
NUM_DOF = len(FINGERS)
ANGLE_MIN, ANGLE_MAX = 0, 1000  # 0 = closed, 1000 = open

CH340_VID, CH340_PID = 0x1A86, 0x7523


class Reg:
    # Addresses are byte offsets: per-finger 16-bit values sit 2 apart (ANGLE_SET + 2 * finger),
    # while the Modbus quantity field counts 16-bit values.
    HAND_ID = 1000
    CLEAR_ERROR = 1004
    SAVE = 1005
    FORCE_SENSOR_CALIB = 1009
    POS_SET = 1474
    ANGLE_SET = 1486
    FORCE_SET = 1498
    SPEED_SET = 1522
    POS_ACT = 1534
    ANGLE_ACT = 1546
    FORCE_ACT = 1582
    CURRENT = 1594
    ERROR = 1606
    STATUS = 1612
    TEMP = 1618


READ_HOLDING = 0x03
WRITE_SINGLE = 0x06
WRITE_MULTIPLE = 0x10

ERROR_BITS = {
    0x01: "locked rotor",
    0x02: "over temperature",
    0x04: "over current",
    0x08: "abnormal operation",
    0x10: "communication error",
}

STATUS_NAMES = {
    0: "unclenching",
    1: "grasping",
    2: "reached target",
    3: "reached force",
    5: "current protection",
    6: "locked rotor",
    7: "fault",
}


class ModbusError(RuntimeError):
    pass


class SerialLike(Protocol):
    def write(self, data: bytes) -> int | None: ...
    def read(self, size: int) -> bytes: ...
    def reset_input_buffer(self) -> None: ...
    def close(self) -> None: ...


def crc16(data: bytes) -> bytes:
    crc = 0xFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ 0xA001 if crc & 1 else crc >> 1
    return crc.to_bytes(2, "little")


class ModbusRTU:
    def __init__(self, ser: SerialLike, slave_id: int = 1):
        self.ser = ser
        self.slave_id = slave_id
        self._lock = threading.Lock()

    def _read_exact(self, n: int) -> bytes:
        data = self.ser.read(n)
        if len(data) != n:
            raise ModbusError(f"timeout: expected {n} bytes, got {len(data)}")
        return data

    def _transact(self, request: bytes, response_len: int) -> bytes:
        with self._lock:
            self.ser.reset_input_buffer()
            self.ser.write(request + crc16(request))
            head = self._read_exact(2)
            if head[1] & 0x80:
                response = head + self._read_exact(3)
                _check_crc(response)
                raise ModbusError(
                    f"device exception code {response[2]} for function 0x{request[1]:02X}"
                )
            response = head + self._read_exact(response_len - 2)
        _check_crc(response)
        if response[0] != self.slave_id or response[1] != request[1]:
            raise ModbusError(f"unexpected response header {response[:2].hex(' ')}")
        return response

    def read_registers(self, address: int, count: int) -> list[int]:
        request = struct.pack(">BBHH", self.slave_id, READ_HOLDING, address, count)
        response = self._transact(request, 5 + 2 * count)
        if response[2] != 2 * count:
            raise ModbusError(f"byte count {response[2]} != {2 * count}")
        return list(struct.unpack(f">{count}H", response[3 : 3 + 2 * count]))

    def write_register(self, address: int, value: int) -> None:
        request = struct.pack(">BBHH", self.slave_id, WRITE_SINGLE, address, value & 0xFFFF)
        self._transact(request, 8)

    def write_registers(self, address: int, values: Sequence[int]) -> None:
        n = len(values)
        request = struct.pack(">BBHHB", self.slave_id, WRITE_MULTIPLE, address, n, 2 * n)
        request += struct.pack(f">{n}H", *(v & 0xFFFF for v in values))
        self._transact(request, 8)


def _check_crc(frame: bytes) -> None:
    if crc16(frame[:-2]) != frame[-2:]:
        raise ModbusError(f"CRC mismatch in {frame.hex(' ')}")


def _split_bytes(registers: Sequence[int]) -> list[int]:
    out: list[int] = []
    for value in registers:
        out.append(value & 0xFF)
        out.append((value >> 8) & 0xFF)
    return out[:NUM_DOF]


def _six(values: int | Sequence[int], name: str) -> list[int]:
    if isinstance(values, int):
        values = [values] * NUM_DOF
    if len(values) != NUM_DOF:
        raise ValueError(f"{name} needs {NUM_DOF} values, got {len(values)}")
    return [int(min(ANGLE_MAX, max(ANGLE_MIN, round(v)))) for v in values]


class InspireHand:
    """Inspire RH56 hand. All per-finger lists follow the order in `FINGERS`."""

    def __init__(self, bus: ModbusRTU):
        self.bus = bus

    @classmethod
    def open(
        cls, port: str, baudrate: int = 115200, slave_id: int = 1, timeout: float = 0.05
    ) -> InspireHand:
        import serial

        ser = serial.Serial(port=port, baudrate=baudrate, timeout=timeout)
        return cls(ModbusRTU(ser, slave_id))

    def close(self) -> None:
        self.bus.ser.close()

    def __enter__(self) -> InspireHand:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def set_angles(self, angles: Sequence[int]) -> None:
        self.bus.write_registers(Reg.ANGLE_SET, _six(angles, "angles"))

    def set_speeds(self, speeds: int | Sequence[int]) -> None:
        self.bus.write_registers(Reg.SPEED_SET, _six(speeds, "speeds"))

    def set_forces(self, forces: int | Sequence[int]) -> None:
        self.bus.write_registers(Reg.FORCE_SET, _six(forces, "forces"))

    def get_angles(self) -> list[int]:
        return self.bus.read_registers(Reg.ANGLE_ACT, NUM_DOF)

    def get_target_angles(self) -> list[int]:
        return self.bus.read_registers(Reg.ANGLE_SET, NUM_DOF)

    def get_forces(self) -> list[int]:
        return [_signed(v) for v in self.bus.read_registers(Reg.FORCE_ACT, NUM_DOF)]

    def get_currents(self) -> list[int]:
        return self.bus.read_registers(Reg.CURRENT, NUM_DOF)

    def get_errors(self) -> list[int]:
        return _split_bytes(self.bus.read_registers(Reg.ERROR, 3))

    def get_statuses(self) -> list[int]:
        return _split_bytes(self.bus.read_registers(Reg.STATUS, 3))

    def get_temperatures(self) -> list[int]:
        return _split_bytes(self.bus.read_registers(Reg.TEMP, 3))

    def hand_id(self) -> int:
        return self.bus.read_registers(Reg.HAND_ID, 1)[0]

    def clear_errors(self) -> None:
        self.bus.write_register(Reg.CLEAR_ERROR, 1)

    def open_all(self) -> None:
        self.set_angles([ANGLE_MAX] * NUM_DOF)


def _signed(value: int) -> int:
    return value - 0x10000 if value & 0x8000 else value


def describe_errors(code: int) -> str:
    if code == 0:
        return "ok"
    return ", ".join(name for bit, name in ERROR_BITS.items() if code & bit) or f"0x{code:02X}"


NO_REPLY_HINT = (
    "No reply from the hand. Check: hand power supply on, RS485 A/B wires not swapped, "
    "adapter connected to the hand, baudrate (default 115200) and slave id (default 1)."
)


def list_serial_ports():
    """Serial ports that correspond to real devices (skips placeholder ttyS* entries)."""
    from serial.tools import list_ports

    return [p for p in list_ports.comports() if p.description != "n/a"]


def find_ch340_ports() -> list[str]:
    from serial.tools import list_ports

    return [p.device for p in list_ports.comports() if (p.vid, p.pid) == (CH340_VID, CH340_PID)]
