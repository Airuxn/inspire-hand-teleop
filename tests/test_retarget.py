import numpy as np

from inspire_teleop.filters import OneEuroFilter
from inspire_teleop.retarget import FINGER_CHAINS, THUMB, Calibration, features


def synthetic_hand(curl: float, thumb_curl: float = 0.0) -> np.ndarray:
    """Right hand in the x/y plane, palm facing -z, each finger joint bent by `curl` rad."""
    lm = np.zeros((21, 3))
    bases = {"index": 0.03, "middle": 0.01, "ring": -0.01, "little": -0.03}
    for name, chain in FINGER_CHAINS.items():
        pos = np.array([bases[name], 0.08, 0.0])
        lm[chain[0]] = pos
        heading = 0.0
        for joint in chain[1:]:
            heading += curl
            pos = pos + 0.03 * np.array([0.0, np.cos(heading), -np.sin(heading)])
            lm[joint] = pos
    pos = np.array([0.03, 0.02, 0.0])
    lm[THUMB[0]] = pos
    heading = 0.0
    for joint in THUMB[1:]:
        pos = pos + 0.025 * np.array([np.cos(heading), np.sin(heading) * 0.2, -np.sin(heading)])
        lm[joint] = pos
        heading += thumb_curl
    return lm


def test_straight_fingers_have_low_curl():
    feats = features(synthetic_hand(0.0))
    assert np.all(feats[:4] < 0.1)


def test_spreading_fingers_is_not_curl():
    lm = synthetic_hand(0.0)
    mcp, *rest = FINGER_CHAINS["index"]
    spread = np.array([[np.cos(0.3), -np.sin(0.3), 0], [np.sin(0.3), np.cos(0.3), 0], [0, 0, 1]])
    for joint in rest:
        lm[joint] = lm[mcp] + spread @ (lm[joint] - lm[mcp])
    assert features(lm)[3] < 0.1


def test_curl_increases_monotonically():
    curls = [features(synthetic_hand(c))[3] for c in (0.0, 0.4, 0.8, 1.2)]
    assert curls == sorted(curls)
    assert curls[-1] > 2.0


def test_calibration_maps_open_to_1000_and_closed_to_0():
    calib = Calibration(open=[0.0] * 6, closed=[2.0] * 6)
    np.testing.assert_allclose(calib.to_angles(np.zeros(6)), 1000)
    np.testing.assert_allclose(calib.to_angles(np.full(6, 2.0)), 0)
    np.testing.assert_allclose(calib.to_angles(np.full(6, 1.0)), 500)
    np.testing.assert_allclose(calib.to_angles(np.full(6, 5.0)), 0)


def test_calibration_handles_inverted_ranges():
    calib = Calibration(open=[0.5] * 6, closed=[-0.5] * 6)
    np.testing.assert_allclose(calib.to_angles(np.full(6, -0.5)), 0)


def test_calibration_roundtrip(tmp_path):
    calib = Calibration(open=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], closed=[1, 2, 3, 4, 5, 6])
    path = tmp_path / "calib.json"
    calib.save(path)
    assert Calibration.load(path) == calib


def test_one_euro_filter_smooths_noise_and_tracks_steps():
    f = OneEuroFilter(min_cutoff=1.0, beta=0.0)
    rng = np.random.default_rng(0)
    out = [f(np.array([500 + rng.normal(0, 20)]), t / 30)[0] for t in range(60)]
    assert np.std(out[30:]) < 10
    f2 = OneEuroFilter(min_cutoff=1.0, beta=0.05)
    out2 = [f2(np.array([0.0 if t < 10 else 1000.0]), t / 30)[0] for t in range(30)]
    assert out2[-1] > 900
