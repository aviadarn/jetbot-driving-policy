"""Closed-loop evaluation on held-out corridors (track seeds 1000+), each driven both ways.

The policy's raw (x, y) goes straight into the verbatim 2024 controller, in the policy's own
label convention, exactly as on the car. A run is scored by completion (with a Wilson 95%
interval), not by loss.

  python scripts/eval.py --run runs/r3_ratio_s0
  python scripts/eval.py --policy expert            # ceiling
  python scripts/eval.py --policy zero              # floor: always (0, 0)
  python scripts/eval.py --run runs/r3_ratio_s0 --latency 4 --tag lat4
"""
import argparse
import json
import math
import multiprocessing as mp
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EVAL_SEEDS = range(1000, 1020)
_MODEL = {}


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def _policy(spec, track):
    from jdp.expert import ExpertPolicy
    if spec["policy"] == "expert":
        return ExpertPolicy(track, spec.get("label_mode", "symmetric"))
    if spec["policy"] == "zero":
        class Zero:
            needs_image = False

            def __call__(self, img, pose):
                return (0.0, 0.0)
        return Zero()
    import torch
    from jdp.model import SteeringNet, TorchPolicy
    key = spec["weights"]
    if key not in _MODEL:
        torch.set_num_threads(1)
        m = SteeringNet(pretrained=False)
        m.load_state_dict(torch.load(key, map_location="cpu"))
        _MODEL[key] = TorchPolicy(m, spec["device"])
    return _MODEL[key]


def run_job(job):
    from jdp.controller import Controller2024
    from jdp.sim import Sim, run_episode
    from jdp.track import Track
    spec, seed, rev = job
    track = Track(seed)
    track = track.reversed() if rev else track
    sim = Sim(track, look_seed=seed * 2 + rev)
    rec = [] if spec.get("gif") and seed == EVAL_SEEDS[0] and not rev else None
    m = run_episode(sim, _policy(spec, track), Controller2024(), latency_steps=spec.get("latency"), record=rec)
    if rec:
        _gif(sim, rec, spec["gif"])
    sim.close()
    m.update({"track": seed, "reversed": rev})
    return m


def _gif(sim, rec, path):
    import imageio.v2 as imageio
    frames = []
    for img, pose, _ in rec[::2]:
        top = sim.render_topdown(pose, size=224, span=7.0)
        frames.append(np.concatenate([img, top], axis=1))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    imageio.mimsave(path, frames, duration=0.1, loop=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", help="runs/<arm>_s<seed> with best.pt and train.json")
    ap.add_argument("--policy", choices=["net", "expert", "zero"], default="net")
    ap.add_argument("--label-mode", default="symmetric", help="for --policy expert")
    ap.add_argument("--tracks", type=int, default=len(EVAL_SEEDS))
    ap.add_argument("--latency", type=int, help="frames of actuation delay (default: config)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--gif", help="save the first episode as onboard|top-down GIF")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default="results/eval")
    a = ap.parse_args()

    if a.run:
        info = json.load(open(os.path.join(a.run, "train.json")))
        spec = {"policy": "net", "weights": os.path.join(a.run, "best.pt"), "device": a.device,
                "label_mode": info["label_mode"]}
        name = os.path.basename(a.run.rstrip("/"))
    else:
        spec = {"policy": a.policy, "label_mode": a.label_mode}
        name = a.policy + ("" if a.label_mode == "symmetric" else f"_{a.label_mode}")
    spec["latency"] = a.latency
    spec["gif"] = a.gif
    name += f"_{a.tag}" if a.tag else ""

    jobs = [(spec, s, rev) for s in list(EVAL_SEEDS)[: a.tracks] for rev in (False, True)]
    t0 = time.time()
    with mp.get_context("spawn").Pool(a.workers) as pool:
        eps = pool.map(run_job, jobs)
    k, n = sum(e["complete"] for e in eps), len(eps)
    lo, hi = wilson(k, n)

    def mean(key):
        vals = [e[key] for e in eps if e[key] is not None]
        return round(float(np.mean(vals)), 4) if vals else None

    status = {}
    for e in eps:
        status[e["status"]] = status.get(e["status"], 0) + 1
    summary = {"name": name, "complete": k, "episodes": n, "rate": round(k / n, 4),
               "wilson95": [round(lo, 4), round(hi, 4)], "progress_frac": mean("progress_frac"),
               "mean_abs_lateral_m": mean("mean_abs_lateral_m"), "weave_per_m": mean("weave_per_m"),
               "steer_saturated_frac": mean("steer_saturated_frac"), "status": status,
               "latency_steps": a.latency, "minutes": round((time.time() - t0) / 60, 2)}
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, f"{name}.json"), "w") as f:
        json.dump({"summary": summary, "spec": {k: v for k, v in spec.items() if k != "gif"},
                   "episodes": eps}, f, indent=1)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
