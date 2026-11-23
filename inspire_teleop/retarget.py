"""Map MediaPipe hand landmarks to the 6 actuators of the Inspire hand.

Each actuator gets one scalar feature (in radians) computed from 3D world
landmarks, then a per-user calibration maps it linearly onto 0..1000.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .driver import ANGLE_MAX, FINGERS, NUM_DOF

WRIST = 0
THUMB = (1, 2, 3, 4)
FINGER_CHAINS = {
    "little": (17, 18, 19, 20),
    "ring": (13, 14, 15, 16),
    "middle": (9, 10, 11, 12),
    "index": (5, 6, 7, 8),
}
INDEX_MCP, MIDDLE_MCP, LITTLE_MCP = 5, 9, 17


def _unit(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-9 else v


def _angle(u: np.ndarray, v: np.ndarray) -> float:
    return float(np.arccos(np.clip(np.dot(_unit(u), _unit(v)), -1.0, 1.0)))


def _palm_axes(lm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unit vectors along the palm (wrist -> middle MCP) and across it (little -> index MCP)."""
    up = _unit(lm[MIDDLE_MCP] - lm[WRIST])
    across = lm[INDEX_MCP] - lm[LITTLE_MCP]
    return up, _unit(across - np.dot(across, up) * up)


def finger_curl(lm: np.ndarray, chain: tuple[int, int, int, int]) -> float:
    """MCP flexion (spread removed) + PIP flexion."""
    mcp, pip, dip, _ = chain
    up, across = _palm_axes(lm)
    proximal = lm[pip] - lm[mcp]
    flexion_plane_proximal = proximal - np.dot(proximal, across) * across
    bend_mcp = _angle(up, flexion_plane_proximal)
    bend_pip = _angle(proximal, lm[dip] - lm[pip])
    return bend_mcp + bend_pip


def thumb_bend(lm: np.ndarray) -> float:
    cmc, mcp, ip, tip = THUMB
    return _angle(lm[mcp] - lm[cmc], lm[ip] - lm[mcp]) + _angle(lm[ip] - lm[mcp], lm[tip] - lm[ip])


def palm_normal(lm: np.ndarray, handedness: str) -> np.ndarray:
    n = _unit(np.cross(lm[INDEX_MCP] - lm[WRIST], lm[LITTLE_MCP] - lm[WRIST]))
    return -n if handedness == "Left" else n


def thumb_rotation(lm: np.ndarray, handedness: str) -> float:
    """Elevation of the thumb (CMC -> IP) out of the palm plane."""
    cmc, _, ip, _ = THUMB
    direction = _unit(lm[ip] - lm[cmc])
    return float(np.arcsin(np.clip(np.dot(direction, palm_normal(lm, handedness)), -1.0, 1.0)))


def features(world_landmarks: np.ndarray, handedness: str = "Right") -> np.ndarray:
    """Raw per-actuator features in `FINGERS` order."""
    lm = np.asarray(world_landmarks, dtype=float)
    if lm.shape != (21, 3):
        raise ValueError(f"expected (21, 3) landmarks, got {lm.shape}")
    return np.array(
        [finger_curl(lm, FINGER_CHAINS[name]) for name in FINGERS[:4]]
        + [thumb_bend(lm), thumb_rotation(lm, handedness)]
    )


@dataclass
class Calibration:
    """Feature values for an open hand and a fist; mapped to 1000 and 0 respectively."""

    open: list[float] = field(default_factory=lambda: [0.15] * 4 + [0.25, -0.10])
    closed: list[float] = field(default_factory=lambda: [2.6] * 4 + [1.1, 0.60])

    def to_angles(self, feats: np.ndarray) -> np.ndarray:
        lo = np.asarray(self.open, dtype=float)
        hi = np.asarray(self.closed, dtype=float)
        span = np.where(np.abs(hi - lo) < 1e-6, 1e-6, hi - lo)
        closure = np.clip((np.asarray(feats, dtype=float) - lo) / span, 0.0, 1.0)
        return ANGLE_MAX * (1.0 - closure)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> Calibration:
        data = json.loads(Path(path).read_text())
        calib = cls(open=list(data["open"]), closed=list(data["closed"]))
        if len(calib.open) != NUM_DOF or len(calib.closed) != NUM_DOF:
            raise ValueError(f"calibration in {path} must have {NUM_DOF} values per pose")
        return calib

    @classmethod
    def load_or_default(cls, path: str | Path | None) -> tuple[Calibration, bool]:
        if path is not None and Path(path).exists():
            return cls.load(path), True
        return cls(), False
