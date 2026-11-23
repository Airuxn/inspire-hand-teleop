"""In-process emulation of the hand's serial port, used by `--mock` and the tests.

The real `ModbusRTU` driver runs unchanged on top of it, so mock mode exercises
the same protocol code path as the physical hand.
"""

from __future__ import annotations

import struct
import time

from .driver import NUM_DOF, READ_HOLDING, WRITE_MULTIPLE, WRITE_SINGLE, Reg, crc16

FULL_STROKE_SECONDS = 0.8  # time for a full 0..1000 stroke at speed 1000
SIMULATED_TRANSFER_S = 0.002


class FakeInspireSerial:
    def __init__(self, slave_id: int = 1, latency: bool = False):
        self.slave_id = slave_id
        self.latency = latency
        self.is_open = True
        self.regs: dict[int, int] = {Reg.HAND_ID: slave_id}
        for i in range(NUM_DOF):
            self.regs[Reg.ANGLE_SET + 2 * i] = 1000
            self.regs[Reg.ANGLE_ACT + 2 * i] = 1000
            self.regs[Reg.SPEED_SET + 2 * i] = 1000
            self.regs[Reg.FORCE_SET + 2 * i] = 1000
        self._actual = [1000.0] * NUM_DOF
        self._last_step = time.monotonic()
        self._out = bytearray()

    def _step(self) -> None:
        now = time.monotonic()
        dt = now - self._last_step
        self._last_step = now
        for i in range(NUM_DOF):
            target = self.regs[Reg.ANGLE_SET + 2 * i]
            speed = self.regs[Reg.SPEED_SET + 2 * i]
            max_delta = speed / FULL_STROKE_SECONDS * dt
            delta = max(-max_delta, min(max_delta, target - self._actual[i]))
            self._actual[i] += delta
            self.regs[Reg.ANGLE_ACT + 2 * i] = int(round(self._actual[i]))

    def write(self, data: bytes) -> int:
        if crc16(data[:-2]) != data[-2:] or data[0] != self.slave_id:
            return len(data)
        self._step()
        func = data[1]
        if func == READ_HOLDING:
            address, count = struct.unpack(">HH", data[2:6])
            values = [self.regs.get(address + 2 * k, 0) for k in range(count)]
            body = struct.pack(">BBB", self.slave_id, func, 2 * count)
            body += struct.pack(f">{count}H", *values)
        elif func == WRITE_SINGLE:
            address, value = struct.unpack(">HH", data[2:6])
            self.regs[address] = value
            body = data[:6]
        elif func == WRITE_MULTIPLE:
            address, count = struct.unpack(">HH", data[2:6])
            values = struct.unpack(f">{count}H", data[7 : 7 + 2 * count])
            for k, value in enumerate(values):
                self.regs[address + 2 * k] = value
            body = data[:6]
        else:
            body = bytes([self.slave_id, func | 0x80, 0x01])
        self._out += body + crc16(body)
        if self.latency:
            time.sleep(SIMULATED_TRANSFER_S)
        return len(data)

    def read(self, size: int) -> bytes:
        chunk = bytes(self._out[:size])
        del self._out[:size]
        return chunk

    def reset_input_buffer(self) -> None:
        self._out.clear()

    def close(self) -> None:
        self.is_open = False
