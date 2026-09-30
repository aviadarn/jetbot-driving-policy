"""MuJoCo scene for one track: terrazzo floor, orange cones, room walls, and a kinematic car
(mocap body) carrying the camera and a sliver of white hood, matching the 2024 frames.

Nothing here collides: the car is placed kinematically each step and failure is checked
geometrically, so every geom sets contype/conaffinity to 0 explicitly.
"""
import os
import math
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image

from .config import CAM, TRACK
from .camera import Fisheye

ASSET_DIR = os.path.join(os.path.dirname(__file__), "_assets")
STOP_H, STOP_HALF = 0.45, 0.15  # kid-scale sign: 30 cm board centred at cone height
NOCOLLIDE = {"contype": "0", "conaffinity": "0"}


def _terrazzo(path, seed=0, size=512):
    """Speckled grey floor, like the hallway in the 2024 frames."""
    if os.path.exists(path):
        return
    rng = np.random.default_rng(seed)
    img = np.full((size, size, 3), (120, 128, 130), np.float32) + rng.normal(0, 10, (size, size, 1))
    for colour, frac in [((240, 240, 240), 0.16), ((40, 40, 45), 0.14),
                         ((90, 140, 140), 0.08), ((190, 150, 150), 0.06)]:
        mask = rng.random((size, size)) < frac
        img[mask] = colour
    img = np.clip(img, 0, 255).astype(np.uint8)
    Image.fromarray(img).save(path)


def _cone_mesh(radius=0.12, height=0.45, n=16):
    ring = [(radius * math.cos(2 * math.pi * k / n), radius * math.sin(2 * math.pi * k / n), 0.03)
            for k in range(n)]
    verts = ring + [(0.0, 0.0, height)]
    return " ".join(f"{a:.4f} {b:.4f} {c:.4f}" for a, b, c in verts)


def build_xml(track, stop_boards=(), rng=None):
    """stop_boards: iterable of (x, y, yaw) for stop-sign boards (Phase 2)."""
    rng = rng or np.random.default_rng(track.seed)
    os.makedirs(ASSET_DIR, exist_ok=True)
    floor_png = os.path.join(ASSET_DIR, "terrazzo.png")
    _terrazzo(floor_png)
    fish = Fisheye()

    root = ET.Element("mujoco", model=f"corridor_{track.seed}")
    ET.SubElement(root, "compiler", angle="radian")
    # znear/zfar are scaled by stat.extent; pin extent so near-floor pixels aren't clipped.
    ET.SubElement(root, "statistic", extent="1", center="0 0 0.3")
    vis = ET.SubElement(root, "visual")
    ET.SubElement(vis, "global", offwidth=str(CAM.render), offheight=str(CAM.render))
    ET.SubElement(vis, "quality", shadowsize="2048", offsamples="4")
    amb = rng.uniform(0.15, 0.35)
    ET.SubElement(vis, "headlight", ambient=f"{amb:.2f} {amb:.2f} {amb:.2f}",
                  diffuse="0.3 0.3 0.3", specular="0 0 0")
    ET.SubElement(vis, "map", znear="0.02", zfar="100")

    asset = ET.SubElement(root, "asset")
    ET.SubElement(asset, "texture", name="floor", type="2d", file=floor_png)
    ET.SubElement(asset, "material", name="floor", texture="floor", texrepeat="1.2 1.2",
                  texuniform="true", reflectance="0.05")
    ET.SubElement(asset, "texture", name="sky", type="skybox", builtin="gradient",
                  rgb1="0.9 0.9 0.92", rgb2="0.6 0.6 0.62", width="64", height="64")
    ET.SubElement(asset, "mesh", name="cone", vertex=_cone_mesh())
    ET.SubElement(asset, "material", name="cone", rgba="1.0 0.33 0.12 1", specular="0.3")
    ET.SubElement(asset, "material", name="hood", rgba="0.95 0.95 0.95 1", specular="0.2")

    world = ET.SubElement(root, "worldbody")
    lo, hi = track.xy.min(0) - 4.0, track.xy.max(0) + 4.0
    centre, half = (lo + hi) / 2, (hi - lo) / 2
    ET.SubElement(world, "geom", name="floor", type="plane", material="floor",
                  pos=f"{centre[0]:.3f} {centre[1]:.3f} 0",
                  size=f"{half[0]:.3f} {half[1]:.3f} 0.1", **NOCOLLIDE)

    # Directional light with randomised direction/strength: the hallway had a skylight glare.
    d = rng.normal(0, 0.4, 2)
    lum = rng.uniform(0.3, 0.6)
    ET.SubElement(world, "light", directional="true", castshadow="true",
                  dir=f"{d[0]:.2f} {d[1]:.2f} -1", diffuse=f"{lum:.2f} {lum:.2f} {lum:.2f}")

    for k, (x, y) in enumerate(track.cone_positions()):
        ET.SubElement(world, "geom", name=f"cone{k}", type="mesh", mesh="cone", material="cone",
                      pos=f"{x:.3f} {y:.3f} 0", **NOCOLLIDE)
        ET.SubElement(world, "geom", type="box", material="cone", size="0.16 0.16 0.015",
                      pos=f"{x:.3f} {y:.3f} 0.015", **NOCOLLIDE)

    # Room walls and clutter (the boxes and shelving behind the cones in 2024).
    wall_rgb = rng.uniform(0.7, 0.9, 3)
    for (px, py, sx, sy) in [(centre[0], hi[1], half[0], 0.05), (centre[0], lo[1], half[0], 0.05),
                             (hi[0], centre[1], 0.05, half[1]), (lo[0], centre[1], 0.05, half[1])]:
        ET.SubElement(world, "geom", type="box", size=f"{sx:.2f} {sy:.2f} 1.5",
                      pos=f"{px:.2f} {py:.2f} 1.5", rgba=_rgba(wall_rgb), **NOCOLLIDE)
    for _ in range(40):
        side = rng.integers(4)
        t = rng.uniform(-1, 1)
        px, py = [(centre[0] + t * half[0], hi[1] - 0.5), (centre[0] + t * half[0], lo[1] + 0.5),
                  (hi[0] - 0.5, centre[1] + t * half[1]), (lo[0] + 0.5, centre[1] + t * half[1])][side]
        s = rng.uniform(0.2, 0.6, 3)
        ET.SubElement(world, "geom", type="box", size=f"{s[0]:.2f} {s[1]:.2f} {s[2]:.2f}",
                      pos=f"{px:.2f} {py:.2f} {s[2]:.2f}",
                      rgba=_rgba(rng.uniform(0.35, 0.85) * np.array([1.0, 0.85, 0.65])), **NOCOLLIDE)

    for k, (x, y, yaw) in enumerate(stop_boards):
        _stop_board(world, asset, k, x, y, yaw)

    car = ET.SubElement(world, "body", name="car", mocap="true", pos="0 0 0")
    p = math.radians(CAM.pitch_deg)
    ET.SubElement(car, "camera", name="cam", pos=f"{CAM.forward} 0 {CAM.height}",
                  xyaxes=f"0 -1 0 {math.sin(p):.5f} 0 {math.cos(p):.5f}",
                  fovy=f"{fish.render_fovy:.3f}")
    # Hood: only its leading edge is visible along the bottom of the frame.
    hood_top = CAM.height - 0.30
    ET.SubElement(car, "geom", type="box", material="hood", size="0.30 0.32 0.02",
                  pos=f"{CAM.forward - 0.27:.3f} 0 {hood_top - 0.02:.3f}", **NOCOLLIDE)
    ET.SubElement(car, "geom", type="box", rgba="0.15 0.15 0.15 1", size="0.01 0.05 0.004",
                  pos=f"{CAM.forward + 0.02:.3f} 0 {hood_top + 0.004:.3f}", **NOCOLLIDE)

    return ET.tostring(root, encoding="unicode")


def _stop_board(world, asset, k, x, y, yaw):
    """Octagonal STOP sign on a post (Phase 2). Texture drawn in code, no external image."""
    tex = os.path.join(ASSET_DIR, "stop.png")
    if not os.path.exists(tex):
        _draw_stop(tex)
    if asset.find("texture[@name='stop']") is None:
        ET.SubElement(asset, "texture", name="stop", type="2d", file=tex)
        ET.SubElement(asset, "material", name="stop", texture="stop")
    body = ET.SubElement(world, "body", name=f"stop{k}", pos=f"{x:.3f} {y:.3f} 0",
                         euler=f"0 0 {yaw:.4f}")
    ET.SubElement(body, "geom", type="cylinder", size=f"0.015 {STOP_H / 2:.3f}", pos=f"0 0 {STOP_H / 2:.3f}",
                  rgba="0.6 0.6 0.6 1", **NOCOLLIDE)
    # Plane-textured board facing -x of the body (toward an approaching car).
    ET.SubElement(body, "geom", name=f"stop{k}_face", type="box", size=f"0.004 {STOP_HALF} {STOP_HALF}",
                  pos=f"-0.02 0 {STOP_H:.3f}", material="stop", **NOCOLLIDE)


def _draw_stop(path, size=256):
    from PIL import ImageDraw, ImageFont
    img = Image.new("RGB", (size, size), (200, 200, 200))
    d = ImageDraw.Draw(img)
    r = size / 2
    pts = [(r + r * 0.98 * math.cos(math.pi / 8 + k * math.pi / 4),
            r + r * 0.98 * math.sin(math.pi / 8 + k * math.pi / 4)) for k in range(8)]
    d.polygon(pts, fill=(255, 255, 255))
    pts = [(r + r * 0.90 * math.cos(math.pi / 8 + k * math.pi / 4),
            r + r * 0.90 * math.sin(math.pi / 8 + k * math.pi / 4)) for k in range(8)]
    d.polygon(pts, fill=(190, 20, 30))
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", size // 4)
    except OSError:
        font = ImageFont.load_default()
    d.text((r, r), "STOP", fill=(255, 255, 255), anchor="mm", font=font)
    img.transpose(Image.FLIP_LEFT_RIGHT).save(path)  # MuJoCo maps it mirrored on the -x face


def _rgba(rgb):
    return f"{rgb[0]:.3f} {rgb[1]:.3f} {rgb[2]:.3f} 1"
