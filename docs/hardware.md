# Hardware notes

## Setup

| Part | Used here |
|---|---|
| Hand | Inspire RH56, 6 actuators (little, ring, middle, index, thumb bend, thumb rotate) |
| Interface | RS485 to USB adapter with a CH340 chip (`1a86:7523`), appears as `/dev/ttyUSB0` |
| Protocol | Modbus RTU, 115200 baud, 8N1, slave id 1 (factory defaults) |
| Host | Ubuntu, Python 3.13 |

The hand needs its own power supply; the USB adapter only carries data. With the adapter
plugged in but the hand unpowered, the port opens normally and every request times out.
`inspire-teleop check` reports this with a checklist.

Linux permissions: the user must be in the `dialout` group. On Ubuntu, remove `brltty` if
the CH340 port disappears right after plugging it in.

## Register map

Addresses are byte offsets. Per-actuator 16-bit values are 2 apart (`ANGLE_SET + 2 * i`),
while the Modbus quantity field counts 16-bit values, so all six values are read or written
with a single request of quantity 6.

| Register | Address | Content |
|---|---|---|
| `ANGLE_SET` | 1486 | target angle, 0 (closed) … 1000 (open) |
| `FORCE_SET` | 1498 | force threshold 0 … 1000 |
| `SPEED_SET` | 1522 | speed 0 … 1000 |
| `ANGLE_ACT` | 1546 | measured angle |
| `FORCE_ACT` | 1582 | measured force (signed) |
| `CURRENT` | 1594 | actuator current |
| `ERROR` | 1606 | error bits per actuator |
| `STATUS` | 1612 | status per actuator |
| `TEMP` | 1618 | temperature per actuator (°C) |

`ERROR`, `STATUS` and `TEMP` are documented as one byte per actuator at consecutive
addresses. Over Modbus the hand returns them as **one register per actuator with the value in
the low byte**: reading 6 registers from `STATUS` returned six status values and none of the
`TEMP` values that start 6 addresses later. Reading three registers and splitting each into
two bytes, as some existing drivers do, yields values for only three actuators with zeros in
between.

## Measurements

Measured on the setup above with `inspire-teleop` 0.2.0, 200 requests each:

| Request | Median | p95 | Max |
|---|---|---|---|
| Read 6 measured angles | 4.55 ms | 4.65 ms | 4.79 ms |
| Write 6 target angles | 4.92 ms | 4.99 ms | 5.07 ms |

For comparison, a driver that sleeps a fixed 200 ms after every request is limited to about
5 requests per second. At the default 60 Hz control rate, one write plus a read of angles and
forces every third cycle fits within the 16.7 ms period.

Not yet measured: actuator dead time and rise time (`inspire-teleop step`), and end-to-end
camera-to-motion latency.

At rest with the hand fully open, the force readings were not zero (−51 … 117), so a force
sensor zero calibration is needed before force values are used for grasp detection.
