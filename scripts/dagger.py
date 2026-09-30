"""One DAgger iteration at a fixed 3000-frame budget.

The 2024 fix for a car that drove badly was to go back and collect more frames of what it
got wrong. DAgger is the principled version: let the trained policy drive, and have the
expert label the states the policy actually visits, including the ones just before it
leaves the corridor.

  1. roll out the `natural` policy (trained on 3000 clean expert-drive frames) on the
     training corridors, keep every 3rd frame, label each with the expert's click
  2. dataset = 1500 of the natural frames + 1500 of these on-policy frames
  3. train it with the same recipe as every other arm

  python scripts/dagger.py --base runs/natural_s0 --seed 0
  python scripts/train.py --arm dagger --seed 0 --manifest runs/dagger_s0/manifest_in.csv
"""
import argparse
import csv
import json
import math
import multiprocessing as mp
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.collect import FIELDS, TRAIN_SEEDS  # noqa: E402

_POLICY = {}


def rollout(job):
    import torch
    from jdp.controller import Controller2024
    from jdp.expert import bucket, click
    from jdp.model import SteeringNet, TorchPolicy
    from jdp.sim import Sim, run_episode
    from jdp.track import Track
    weights, seed, pool_dir, tag, device = job
    if weights not in _POLICY:
        torch.set_num_threads(1)
        m = SteeringNet(pretrained=False)
        m.load_state_dict(torch.load(weights, map_location="cpu"))
        _POLICY[weights] = TorchPolicy(m, device)
    track = Track(seed)
    sim = Sim(track, look_seed=seed + 7_000)  # new look, same corridor
    rec = []
    ep = run_episode(sim, _POLICY[weights], Controller2024(), record=rec)
    sim.close()
    rows = []
    rel_dir = os.path.join("frames", f"{tag}", f"t{seed:04d}")
    os.makedirs(os.path.join(pool_dir, rel_dir), exist_ok=True)
    for k, (img, pose, _) in enumerate(rec[::3]):
        s, lat, i = track.locate(pose[0], pose[1])
        if abs(lat) > track.cfg.half_width:  # past the cones: nothing sensible to click
            continue
        u, v, _ = click(track, pose, hint=i)
        dh = (pose[2] - track.heading[i] + math.pi) % (2 * math.pi) - math.pi
        rel = os.path.join(rel_dir, f"d{k:05d}.jpg")
        Image.fromarray(img).save(os.path.join(pool_dir, rel), quality=95)
        rows.append({"path": rel, "split": "train", "track": seed, "source": "dagger",
                     "s": round(s, 3), "lateral": round(lat, 4), "dheading": round(dh, 4),
                     "u": round(u, 2), "v": round(v, 2), "bucket": bucket(track, s),
                     "kappa_ahead": round(track.mean_curvature_ahead(s, 1.8), 4)})
    return rows, ep["status"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="runs/natural_s<seed>")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pool", default="data/pool")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--budget", type=int, default=3000)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--device", default="mps")
    a = ap.parse_args()

    tag = f"dagger_s{a.seed}"
    jobs = [(os.path.join(a.base, "best.pt"), s, a.pool, tag, a.device) for s in TRAIN_SEEDS]
    with mp.get_context("spawn").Pool(a.workers) as pool:
        out = pool.map(rollout, jobs)
    on_policy = [r for rows, _ in out for r in rows]
    statuses = [st for _, st in out]

    with open(os.path.join(a.base, "manifest.csv")) as f:
        base_rows = list(csv.DictReader(f))
    rng = np.random.default_rng(a.seed)
    half = a.budget // 2
    if len(on_policy) < half:
        raise SystemExit(f"only {len(on_policy)} on-policy frames, need {half}")
    pick_base = [base_rows[i] for i in rng.permutation(len(base_rows))[:half]]
    pick_new = [on_policy[i] for i in rng.permutation(len(on_policy))[: a.budget - half]]
    run = os.path.join(a.runs, tag)
    os.makedirs(run, exist_ok=True)
    with open(os.path.join(run, "manifest_in.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows([{k: r[k] for k in FIELDS} for r in pick_base + pick_new])
    info = {"base": a.base, "on_policy_frames": len(on_policy),
            "base_policy_on_train_tracks": {s: statuses.count(s) for s in set(statuses)},
            "mix": {"expert_drive": len(pick_base), "on_policy_relabelled": len(pick_new)}}
    json.dump(info, open(os.path.join(run, "dagger.json"), "w"), indent=1)
    print(json.dumps(info))


if __name__ == "__main__":
    main()
