"""160-degree IMX219 fisheye, modelled as an equidistant lens on a 4:3 sensor that
jetbot.Camera squashes to 224x224.

MuJoCo only renders pinhole images, so we render a wide square pinhole view and
remap it. `project` is the analytic forward model; the expert uses it to turn a
3D target into the pixel a human would have clicked. Same model both ways, so
labels and pixels agree by construction (tests/test_camera.py checks it).

Camera frame is MuJoCo's: x right, y up, looking down -z.
"""
import math
import numpy as np
import cv2

from .config import CAM


class Fisheye:
    def __init__(self, cfg=CAM):
        self.cfg = cfg
        half_diag = math.hypot(cfg.sensor_w, cfg.sensor_h) / 2
        self.theta_max = math.radians(cfg.diag_fov_deg / 2)
        self.f = half_diag / self.theta_max  # sensor units per radian
        # Smallest square pinhole that covers the fisheye's corner rays.
        phi_c = math.atan2(cfg.sensor_h, cfg.sensor_w)
        t = math.tan(self.theta_max)
        self.tan_alpha = max(t * math.cos(phi_c), t * math.sin(phi_c)) * 1.01
        self.render_fovy = 2 * math.degrees(math.atan(self.tan_alpha))
        self.map_x, self.map_y = self._build_map()
        self._build_pyramid_weights()

    def _sensor_of_pixel(self, u, v):
        c = self.cfg
        sx = (u + 0.5) / c.out * c.sensor_w - c.sensor_w / 2
        sy = c.sensor_h / 2 - (v + 0.5) / c.out * c.sensor_h
        return sx, sy

    def _build_map(self):
        c = self.cfg
        u, v = np.meshgrid(np.arange(c.out, dtype=np.float64), np.arange(c.out, dtype=np.float64))
        sx, sy = self._sensor_of_pixel(u, v)
        theta = np.hypot(sx, sy) / self.f
        phi = np.arctan2(sy, sx)
        # pinhole plane coordinates, tan(theta) along the azimuth
        px = np.tan(theta) * np.cos(phi)
        py = np.tan(theta) * np.sin(phi)
        half = c.render / 2
        mx = half + px / self.tan_alpha * half - 0.5
        my = half - py / self.tan_alpha * half - 0.5
        return mx.astype(np.float32), my.astype(np.float32)

    def _build_pyramid_weights(self, levels=3):
        """Near the fisheye edge one output pixel spans ~5 pinhole pixels, which aliases the
        floor speckle. Sample from a Gaussian pyramid level matched to the local footprint."""
        gy_x, gx_x = np.gradient(self.map_x)
        gy_y, gx_y = np.gradient(self.map_y)
        footprint = np.sqrt(np.abs(gx_x * gy_y - gy_x * gx_y))
        lvl = np.clip(np.log2(np.maximum(footprint, 1.0)), 0, levels - 1)
        self.levels = levels
        self.level_maps = [((self.map_x + 0.5) / 2 ** k - 0.5, (self.map_y + 0.5) / 2 ** k - 0.5)
                           for k in range(levels)]
        w = np.stack([np.clip(1 - np.abs(lvl - k), 0, 1) for k in range(levels)])
        self.level_w = (w / w.sum(0, keepdims=True)).astype(np.float32)[..., None]

    def remap(self, pinhole_rgb):
        out = np.zeros((self.cfg.out, self.cfg.out, 3), np.float32)
        img = pinhole_rgb
        for k in range(self.levels):
            if k:
                img = cv2.pyrDown(img)
            mx, my = self.level_maps[k]
            out += self.level_w[k] * cv2.remap(img, mx, my, cv2.INTER_LINEAR,
                                               borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
        return np.clip(out + 0.5, 0, 255).astype(np.uint8)

    def project(self, p_cam):
        """Camera-frame point(s) (..., 3) -> pixel (u, v) in the 224 frame. NaN if behind."""
        p = np.asarray(p_cam, dtype=np.float64)
        x, y, z = p[..., 0], p[..., 1], -p[..., 2]  # z: depth along the view axis
        theta = np.arctan2(np.hypot(x, y), z)
        phi = np.arctan2(y, x)
        r = self.f * theta
        sx, sy = r * np.cos(phi), r * np.sin(phi)
        c = self.cfg
        u = (sx + c.sensor_w / 2) / c.sensor_w * c.out - 0.5
        v = (c.sensor_h / 2 - sy) / c.sensor_h * c.out - 0.5
        bad = theta > math.pi / 2
        return np.where(bad, np.nan, u), np.where(bad, np.nan, v)


def cam_pose_world(car_pose, cfg=CAM):
    """Camera position and rotation (columns = cam x,y,z axes in world) for a car pose."""
    x, y, psi = car_pose
    p = math.radians(cfg.pitch_deg)
    c, s = math.cos(psi), math.sin(psi)
    pos = np.array([x + cfg.forward * c, y + cfg.forward * s, cfg.height])
    fwd = np.array([c * math.cos(p), s * math.cos(p), -math.sin(p)])  # view direction
    up = np.array([c * math.sin(p), s * math.sin(p), math.cos(p)])
    right = np.cross(fwd, up)
    return pos, np.stack([right, up, -fwd], axis=1)


def world_to_cam(points, car_pose, cfg=CAM):
    pos, R = cam_pose_world(car_pose, cfg)
    return (np.asarray(points, dtype=np.float64) - pos) @ R
