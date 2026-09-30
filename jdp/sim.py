"""One corridor, one renderer, one car: render frames and run closed-loop episodes."""
import os
import math
import numpy as np

os.environ.setdefault("MUJOCO_GL", "glfw" if os.uname().sysname == "Darwin" else "egl")
import mujoco  # noqa: E402

from .camera import Fisheye  # noqa: E402
from .car import Car  # noqa: E402
from .config import CAM, TRACK, DT  # noqa: E402
from .scene import build_xml  # noqa: E402


class Look:
    """Per-episode image-space randomisation: the 2024 camera had a strong pink/magenta
    cast (IMX219 without an IR-cut filter), soft focus and sensor noise."""

    def __init__(self, rng, enabled=True):
        self.enabled = enabled
        pink = rng.uniform(0.0, 1.0)
        self.gain = np.array([1.0 + 0.12 * pink, 1.0 - 0.10 * pink, 1.0 + 0.10 * pink])
        self.gain *= rng.uniform(0.8, 1.2)
        self.gamma = rng.uniform(0.85, 1.15)
        self.noise = rng.uniform(1.0, 5.0)
        self.rng = rng

    def __call__(self, img):
        if not self.enabled:
            return img
        f = img.astype(np.float32) / 255.0
        f = np.power(np.clip(f * self.gain, 0, 1), self.gamma) * 255.0
        f += self.rng.normal(0, self.noise, f.shape)
        return np.clip(f, 0, 255).astype(np.uint8)


class Sim:
    def __init__(self, track, stop_boards=(), look_seed=None, randomise=True):
        self.track = track
        self.model = mujoco.MjModel.from_xml_string(build_xml(track, stop_boards))
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, CAM.render, CAM.render)
        self.cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "cam")
        self.fish = Fisheye()
        self.look = Look(np.random.default_rng(track.seed if look_seed is None else look_seed),
                         enabled=randomise)

    def close(self):
        self.renderer.close()

    def set_pose(self, pose):
        x, y, psi = pose
        self.data.mocap_pos[0] = (x, y, 0.0)
        self.data.mocap_quat[0] = (math.cos(psi / 2), 0.0, 0.0, math.sin(psi / 2))
        mujoco.mj_forward(self.model, self.data)

    def render(self, pose, raw=False):
        self.set_pose(pose)
        self.renderer.update_scene(self.data, camera=self.cam_id)
        img = self.fish.remap(self.renderer.render())
        return img if raw else self.look(img)

    def render_topdown(self, pose, size=320, span=None):
        """Overhead view for GIFs (free camera above the car)."""
        self.set_pose(pose)
        if not hasattr(self, "_top"):
            self._top = mujoco.Renderer(self.model, size, size)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = (pose[0], pose[1], 0.0)
        cam.distance = span or 9.0
        cam.elevation = -90.0
        cam.azimuth = 90.0
        self._top.update_scene(self.data, camera=cam)
        return self._top.render()


def run_episode(sim, policy, controller, start_s=0.3, latency_steps=None, max_time=None,
                record=None, stop_fn=None, period=1):
    """Closed-loop drive from the start of the corridor to its end.

    policy(img, pose) -> (x, y) in the policy's label convention (learned policies ignore
    pose; the expert ignores img and sets needs_image=False to skip rendering).
    Returns metrics dict. `record` (list) collects (onboard, pose) per step for GIFs.
    `period` > 1 models a loop slower than the camera: the latest frame is processed every
    `period` frames and the last command is held in between (latency_steps sets the delay).
    """
    tr = sim.track
    car = Car(tr.pose_at(start_s), latency_steps=latency_steps)
    controller.reset()
    max_time = max_time or tr.length / 0.2  # generous: 0.2 m/s average
    hint = tr.index(start_s)
    lat_hist, s_hist, steer_hist = [], [], []
    t, status, k = 0.0, "timeout", 0
    throttle, steer = 0.0, 0.0
    while t < max_time:
        fresh = k % period == 0
        k += 1
        img = None
        if fresh and (getattr(policy, "needs_image", True) or record is not None):
            img = sim.render(car.pose)
        if fresh:
            xy = policy(img, car.pose)
            stop = bool(stop_fn(img)) if stop_fn else False
            throttle, steer = controller.step(xy, stop=stop)
        if record is not None and img is not None:
            record.append((img, car.pose.copy(), float(steer)))
        car.step(throttle, steer)
        t += DT
        s, lat, hint = tr.locate(car.x, car.y, hint=hint)
        herr = (car.psi - tr.heading[hint] + math.pi) % (2 * math.pi) - math.pi
        lat_hist.append(lat)
        s_hist.append(s)
        steer_hist.append(steer)
        if abs(lat) > TRACK.fail_lateral:
            status = "off_left" if lat > 0 else "off_right"
            break
        if abs(herr) > math.radians(TRACK.fail_heading_deg):
            status = "turned_back"
            break
        if s >= tr.length - 1.0:
            status = "complete"
            break
    return _metrics(tr, status, t, np.array(s_hist), np.array(lat_hist), np.array(steer_hist))


def _metrics(tr, status, t, s, lat, steer):
    progress = float(s.max()) if len(s) else 0.0
    ds = np.diff(s, prepend=s[0]) if len(s) else s
    dist = float(np.clip(ds, 0, None).sum())
    # Weaving: lateral-error zero crossings per metre on straights. Not steering sign flips:
    # the 2024 controller's -0.3 counter-steer flips sign by design.
    idx = np.array([tr.index(v) for v in s]) if len(s) else np.array([], int)
    straight = np.abs(tr.kappa[idx]) < 0.05 if len(s) else np.array([], bool)
    crossings = _hysteresis_crossings(lat, 0.03) & straight if len(s) > 1 else np.array([])
    straight_m = float(np.clip(ds, 0, None)[straight].sum()) if len(s) else 0.0
    return {
        "status": status,
        "complete": status == "complete",
        "time_s": round(t, 3),
        "progress_m": round(progress, 3),
        "progress_frac": round(progress / tr.length, 4),
        "distance_m": round(dist, 3),
        "mean_abs_lateral_m": round(float(np.abs(lat).mean()), 4) if len(lat) else None,
        "max_abs_lateral_m": round(float(np.abs(lat).max()), 4) if len(lat) else None,
        "weave_per_m": round(float(crossings.sum()) / straight_m, 4) if straight_m > 1 else None,
        "steer_saturated_frac": round(float((np.abs(steer) >= 1.0).mean()), 4) if len(steer) else None,
    }


def _hysteresis_crossings(lat, band):
    """Boolean per step: lateral error swung from beyond -band to beyond +band (or back).
    Small wiggles inside the band don't count, so a steady driver scores ~0."""
    out = np.zeros(len(lat), bool)
    state = 0
    for i, v in enumerate(lat):
        if v > band and state <= 0:
            out[i] = state == -1
            state = 1
        elif v < -band and state >= 0:
            out[i] = state == 1
            state = -1
    return out
