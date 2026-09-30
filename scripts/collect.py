"""Build the frame pool the experiment arms are sampled from.

Two sources per track:
  perturbed  snapshots at poses offset from the centreline, the way the 2024 frames were
             taken (place the car, click where it should steer). Includes recovery views.
  drive      every 3rd frame of the expert driving the corridor cleanly: the "natural"
             distribution a recorder would capture.

Track seeds: train 0-59, val 500-519 (checkpoint selection), eval 1000+ (closed loop only,
never collected). Labels are stored as raw pixel clicks (u, v); the label convention is
applied at training time.

  python scripts/collect.py --out data/pool
  python scripts/collect.py --out data/smoke --train 2 --val 1 --perturbed 50   # quick check
"""
import argparse
import csv
import math
import multiprocessing as mp
import os
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from jdp.config import DEG  # noqa: E402
from jdp.controller import Controller2024  # noqa: E402
from jdp.expert import ExpertPolicy, bucket, click  # noqa: E402
from jdp.sim import Sim, run_episode  # noqa: E402
from jdp.track import Track  # noqa: E402

FIELDS = ["path", "split", "track", "source", "s", "lateral", "dheading", "u", "v", "bucket", "kappa_ahead"]
TRAIN_SEEDS = range(0, 60)
VAL_SEEDS = range(500, 520)


def collect_track(args):
    seed, split, n_perturbed, out = args
    rng = np.random.default_rng(10_000 + seed)
    track = Track(seed)
    sim = Sim(track)
    rows = []
    tdir = os.path.join(out, "frames", f"t{seed:04d}")
    os.makedirs(tdir, exist_ok=True)

    def save(img, pose, source, k):
        s, lat, i = track.locate(pose[0], pose[1])
        u, v, _ = click(track, pose, hint=i)
        dh = (pose[2] - track.heading[i] + math.pi) % (2 * math.pi) - math.pi
        rel = os.path.join("frames", f"t{seed:04d}", f"{source[0]}{k:05d}.jpg")
        Image.fromarray(img).save(os.path.join(out, rel), quality=95)
        rows.append({"path": rel, "split": split, "track": seed, "source": source,
                     "s": round(s, 3), "lateral": round(lat, 4), "dheading": round(dh, 4),
                     "u": round(u, 2), "v": round(v, 2), "bucket": bucket(track, s),
                     "kappa_ahead": round(track.mean_curvature_ahead(s, 1.8), 4)})

    for k in range(n_perturbed):
        s = rng.uniform(0.5, track.length - 2.0)
        lat = rng.uniform(-0.45, 0.45)
        dh = float(np.clip(rng.normal(0, 15), -35, 35)) * DEG
        pose = track.pose_at(s, lateral=lat, dheading=dh)
        save(sim.render(pose), pose, "perturbed", k)

    rec = []
    run_episode(sim, ExpertPolicy(track), Controller2024(), record=rec)
    for k, (img, pose, _) in enumerate(rec[::3]):
        save(img, pose, "drive", k)
    sim.close()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/pool")
    ap.add_argument("--train", type=int, default=len(TRAIN_SEEDS))
    ap.add_argument("--val", type=int, default=len(VAL_SEEDS))
    ap.add_argument("--perturbed", type=int, default=500, help="snapshots per track")
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 2))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    jobs = [(s, "train", a.perturbed, a.out) for s in list(TRAIN_SEEDS)[: a.train]]
    jobs += [(s, "val", a.perturbed // 2, a.out) for s in list(VAL_SEEDS)[: a.val]]
    t0 = time.time()
    rows = []
    with mp.get_context("spawn").Pool(a.workers) as pool:
        for k, r in enumerate(pool.imap_unordered(collect_track, jobs), 1):
            rows += r
            print(f"[{k}/{len(jobs)}] {len(rows)} frames  {time.time() - t0:.0f}s", flush=True)
    rows.sort(key=lambda r: r["path"])
    with open(os.path.join(a.out, "pool.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    by = {}
    for r in rows:
        key = (r["split"], r["source"], r["bucket"])
        by[key] = by.get(key, 0) + 1
    for key in sorted(by):
        print(key, by[key])


if __name__ == "__main__":
    main()
