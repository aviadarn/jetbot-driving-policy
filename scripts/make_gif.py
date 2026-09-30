"""Side-by-side drive GIF: the same held-out corridor, one row per policy, each row
onboard fisheye | top-down, with the steering command drawn as a bar under the onboard view.

  python scripts/make_gif.py --runs hist_r3_s0 hist_r3_bug_s0 \
      --labels "centred labels" "2024 labels + flip" --out assets/drive_compare.gif
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from jdp.controller import Controller2024  # noqa: E402
from jdp.model import SteeringNet, TorchPolicy  # noqa: E402
from jdp.sim import Sim, run_episode  # noqa: E402
from jdp.track import Track  # noqa: E402


def drive(run, track_seed, device):
    m = SteeringNet(pretrained=False)
    m.load_state_dict(torch.load(os.path.join("runs", run, "best.pt"), map_location="cpu"))
    sim = Sim(Track(track_seed), look_seed=track_seed * 2)
    rec = []
    res = run_episode(sim, TorchPolicy(m, device), Controller2024(), record=rec)
    tops = [sim.render_topdown(pose, size=224, span=7.0) for _, pose, _ in rec]
    sim.close()
    return rec, tops, res


def row(img, top, steer, label, tile):
    pair = Image.fromarray(np.concatenate([img, top], axis=1)).resize((2 * tile, tile), Image.BILINEAR)
    canvas = Image.new("RGB", (2 * tile, tile + 22), (252, 252, 251))
    canvas.paste(pair, (0, 0))
    d = ImageDraw.Draw(canvas)
    d.text((4, tile + 5), label, fill=(11, 11, 11))
    # steering bar: centre tick, fill toward the commanded side, clipped at +-1 like the motor
    cx, w = tile + tile // 2 - 14, tile // 2 - 36
    s = float(np.clip(steer, -1, 1))
    d.rectangle([cx - w, tile + 8, cx + w, tile + 14], outline=(195, 194, 183))
    d.rectangle([min(cx, cx + s * w), tile + 8, max(cx, cx + s * w), tile + 14], fill=(42, 120, 214))
    d.line([cx, tile + 5, cx, tile + 17], fill=(82, 81, 78))
    d.text((cx + w + 6, tile + 5), "steer", fill=(137, 135, 129))
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--track", type=int, default=1000)
    ap.add_argument("--out", default="assets/drive_compare.gif")
    ap.add_argument("--every", type=int, default=3)
    ap.add_argument("--max-frames", type=int, default=110)
    ap.add_argument("--tile", type=int, default=176)
    ap.add_argument("--device", default="mps")
    a = ap.parse_args()
    drives = [drive(r, a.track, a.device) for r in a.runs]
    for r, (_, _, res) in zip(a.runs, drives):
        print(r, json.dumps(res))
    n = min(a.max_frames, max(len(rec) for rec, _, _ in drives[:]) // a.every)
    frames = []
    for k in range(n):
        rows = []
        for (rec, tops, res), label in zip(drives, a.labels):
            i = min(k * a.every, len(rec) - 1)  # a run that ended early holds its last frame
            tag = label if k * a.every < len(rec) else f"{label}: {res['status'].replace('_', ' ')}"
            rows.append(row(rec[i][0], tops[i], rec[i][2], tag, a.tile))
        stack = Image.new("RGB", (rows[0].width, sum(r.height for r in rows)))
        y = 0
        for r in rows:
            stack.paste(r, (0, y))
            y += r.height
        frames.append(stack.quantize(colors=64, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    frames[0].save(a.out, save_all=True, append_images=frames[1:], duration=150, loop=0, optimize=True)
    print(f"wrote {a.out}: {len(frames)} frames, {os.path.getsize(a.out) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
