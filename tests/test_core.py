import math
import numpy as np
import pytest

from jdp.controller import Controller2024
from jdp.labels import encode, decode, flip_x
from jdp.track import Track
from jdp.camera import Fisheye, world_to_cam
from jdp.car import Car


def notebook_execute(xy, state):
    """rf_live_detection.ipynb cell 13, transcribed with numpy exactly as written."""
    x = xy[0]
    y = (0.5 - xy[1]) / 2.0
    speed_offset = 0.34
    angle_diff_bias = 0.1
    angle = np.arctan2(x, y)
    if abs(angle) + angle_diff_bias < abs(state["angle_last"]):
        angle_temp = -1 * angle * 0.3
    else:
        angle_temp = angle
    speed_value = speed_offset - abs(angle_temp) * 0.05
    if abs(angle_temp) < 0.05:
        angle_temp = 0
    state["angle_last"] = angle
    angle_temp *= 1.4
    return speed_value, angle_temp


def test_controller_matches_notebook_on_random_sequence():
    rng = np.random.default_rng(0)
    ctl, state = Controller2024(), {"angle_last": 0.0}
    for xy in rng.uniform(-1.5, 3.5, (500, 2)):
        a = ctl.step(xy)
        b = notebook_execute(xy, state)
        assert a == pytest.approx(b, abs=1e-12)


def test_controller_stop_zeroes_throttle_only():
    ctl = Controller2024()
    throttle, steer = ctl.step((0.5, -0.2), stop=True)
    assert throttle == 0.0 and steer != 0.0


@pytest.mark.parametrize("mode", ["2024", "symmetric"])
def test_label_round_trip(mode):
    for u, v in [(0, 0), (111.5, 80), (223, 223), (40, 150)]:
        assert decode(*encode(u, v, mode), mode) == pytest.approx((u, v))


def test_2024_flip_is_wrong_by_2p46():
    x = encode(150, 0, "2024")[0]
    right = flip_x(x, "2024", faithful=False)
    assert right == pytest.approx(encode(223 - 150, 0, "2024")[0])
    assert right - flip_x(x, "2024", faithful=True) == pytest.approx(2.46)


def test_symmetric_flip_is_negation_within_half_pixel():
    x = encode(150, 0, "symmetric")[0]
    assert flip_x(x, "symmetric", faithful=False) == pytest.approx(encode(223 - 150, 0, "symmetric")[0], abs=1 / 112)


def test_track_is_deterministic_and_reversible():
    a, b = Track(7), Track(7)
    np.testing.assert_array_equal(a.xy, b.xy)
    r = a.reversed()
    np.testing.assert_allclose(r.xy[0], a.xy[-1])
    s, lat, _ = r.locate(*r.pose_at(5.0, lateral=0.3)[:2])
    assert s == pytest.approx(5.0, abs=0.06) and lat == pytest.approx(0.3, abs=0.02)


def test_fisheye_projection_matches_remap_map():
    """A pixel's ray, pushed through `project`, lands back on that pixel."""
    f = Fisheye()
    for u, v in [(112, 112), (20, 30), (200, 190), (5, 218)]:
        sx, sy = f._sensor_of_pixel(u, v)
        theta, phi = math.hypot(sx, sy) / f.f, math.atan2(sy, sx)
        ray = np.array([math.sin(theta) * math.cos(phi), math.sin(theta) * math.sin(phi), -math.cos(theta)])
        pu, pv = f.project(ray)
        assert (float(pu), float(pv)) == pytest.approx((u, v), abs=1e-6)


def test_point_ahead_projects_to_centre_column():
    tr = Track(3)
    pose = np.array([0.0, 0.0, 0.0])
    u, _ = Fisheye().project(world_to_cam([3.0, 0.0, 0.0], pose))
    assert float(u) == pytest.approx(111.5, abs=0.01)


def test_positive_steer_turns_right():
    car = Car((0, 0, 0), latency_steps=0)
    for _ in range(40):
        car.step(0.34, 1.0)
    assert car.psi < 0 and car.y < 0
