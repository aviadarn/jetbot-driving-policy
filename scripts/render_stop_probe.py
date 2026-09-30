"""Render the stop-sign probe set: a 30 cm STOP board (centre 0.45 m up) at the right edge of a straight,
seen from 1 to 6 m, plus the same views with no board (for false stops).

Ground-truth boxes come from projecting the board's corners through the same fisheye model
the renderer uses. Track seeds 2000+ are used nowhere else.

  python scripts/render_stop_probe.py --out data/stop_probe
"""
import argparse
import csv
import math
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from jdp.camera import world_to_cam  # noqa: E402
from jdp.expert import fisheye  # noqa: E402
from jdp.scene import STOP_H as SIGN_H, STOP_HALF as SIGN_HALF  # noqa: E402
from jdp.sim import Sim  # noqa: E402
from jdp.track import Track  # noqa: E402

DISTANCES = np.arange(1.0, 6.01, 0.5)
SIGN_LAT = -0.55  # right of the centreline, inside the cone line


def straight_start(track, need):
    """First s where the next `need` metres are straight."""
    straight = np.abs(track.kappa) < 1e-3
    run = int(need / track.cfg.ds)
    for i in range(int(1.0 / track.cfg.ds), len(straight) - run):
        if straight[i:i + run].all():
            return float(track.s[i])
    return None


def board_box(board_pose, car_pose):
    x, y, yaw = board_pose
    c, s = math.cos(yaw), math.sin(yaw)
    pts = []
    for dy in (-SIGN_HALF, SIGN_HALF):
        for dz in (-SIGN_HALF, SIGN_HALF):
            lx, ly = -0.02, dy
            pts.append([x + c * lx - s * ly, y + s * lx + c * ly, SIGN_H + dz])
    u, v = fisheye().project(world_to_cam(np.array(pts), car_pose))
    return float(np.min(u)), float(np.min(v)), float(np.max(u)), float(np.max(v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/stop_probe")
    ap.add_argument("--tracks", type=int, default=12)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows, seed = [], 2000
    used = 0
    while used < a.tracks:
        track = Track(seed)
        s0 = straight_start(track, need=DISTANCES[-1] + 1.0)
        seed += 1
        if s0 is None:
            continue
        used += 1
        car = track.pose_at(s0)
        for d in DISTANCES:
            i = track.index(s0 + d)
            bx, by = track.pose_at(s0 + d, lateral=SIGN_LAT)[:2]
            board = (bx, by, float(track.heading[i]))
            for with_sign in (True, False):
                sim = Sim(track, stop_boards=[board] if with_sign else (), look_seed=seed * 100 + int(d * 10))
                img = sim.render(car)
                sim.close()
                name = f"t{track.seed}_d{d:.1f}_{'pos' if with_sign else 'neg'}.png"
                Image.fromarray(img).save(os.path.join(a.out, name))
                box = board_box(board, car) if with_sign else (None,) * 4
                rows.append({"file": name, "track": track.seed, "distance_m": d, "sign": int(with_sign),
                             "x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3]})
        print(f"track {track.seed}: s0={s0:.1f}", flush=True)
    with open(os.path.join(a.out, "labels.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} frames -> {a.out}")


if __name__ == "__main__":
    main()
