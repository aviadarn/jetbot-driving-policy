"""Single source of truth for the physical and camera constants.

The 2024 car's geometry was never measured, so these are estimates read off the
2024 camera frames (docs/original_2024/) and a typical kids' ride-on car.
Anything that is a guess is marked GUESS; Phase 6 calibrates those on the car.
"""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class CarCfg:
    wheelbase: float = 0.65          # m, GUESS (kids' ride-on)
    half_width: float = 0.30         # m, GUESS
    max_steer: float = 0.45          # rad at |steer command| = 1, GUESS
    steer_tau: float = 0.20          # s, first-order steering lag, GUESS (actuator unknown)
    speed_per_throttle: float = 2.0  # m/s per unit throttle; 0.34 -> 0.68 m/s, GUESS
    control_hz: float = 20.0         # ~21 fps camera in 2024
    latency_steps: int = 1           # frames between capture and actuation (Phase 5 sweeps this)


@dataclass(frozen=True)
class CamCfg:
    out: int = 224                   # policy input, 224x224 like 2024
    # The lens is sold as 160 deg (IMX219-160), but rendering at 160 looks far wider than the
    # 2024 frames. 110 deg effective was fitted by eye against docs/original_2024/;
    # a checkerboard calibration on the real camera (Phase 6) replaces this.
    diag_fov_deg: float = 110.0
    sensor_w: float = 4.0            # 4:3 sensor squashed to a square frame, like jetbot.Camera
    sensor_h: float = 3.0
    render: int = 1024               # square pinhole render that is remapped to fisheye
    height: float = 0.40             # m above floor, GUESS from 2024 frames
    forward: float = 0.30            # m ahead of the rear axle, GUESS
    pitch_deg: float = 26.0          # down-tilt, fitted with the FOV to the 2024 horizon line


@dataclass(frozen=True)
class TrackCfg:
    half_width: float = 0.90         # cone lines at +-0.9 m from the centreline
    cone_spacing: float = 0.60       # m, matches spacing in the 2024 frames
    fail_lateral: float = 0.60       # m: car body reaches the cone line (0.9 - half_width)
    fail_heading_deg: float = 90.0
    length: float = 40.0             # m of corridor per track
    ds: float = 0.05                 # centreline sampling
    lookahead: float = 1.3           # m, where the expert "clicks"


CAR = CarCfg()
CAM = CamCfg()
TRACK = TrackCfg()
DT = 1.0 / CAR.control_hz
DEG = math.pi / 180.0
