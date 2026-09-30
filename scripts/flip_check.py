"""Does the bug-faithful net average flipped and unflipped views, or tell them apart?

For held-out frames I with click (u, v), the 2024 pipeline trained on
  (I, (u-50)/50)   and   (mirror(I), -(u-50)/50).
If the net cannot tell a mirrored frame from a real one it must average those two
targets, which gives (u-111.5)/50 on real frames (the "flip-mean"). If it can, it outputs
the raw 2024 label on real frames and its negation on mirrored ones.

  python scripts/flip_check.py --run runs/hist_r3_bug_s0
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from jdp.data import read_pool, val_rows  # noqa: E402
from jdp.model import SteeringNet  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/hist_r3_bug_s0")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--pool", default="data/pool")
    ap.add_argument("--out", default="results/flip_check.json")
    a = ap.parse_args()
    m = SteeringNet(pretrained=False)
    m.load_state_dict(torch.load(os.path.join(a.run, "best.pt"), map_location="cpu"))
    m.eval()
    rows = val_rows(read_pool(a.pool))[: a.n]
    real, mirr, raw, mean = [], [], [], []
    with torch.no_grad():
        for r in rows:
            img = np.array(Image.open(os.path.join(a.pool, r["path"])).convert("RGB"))
            for arr, out in ((img, real), (img[:, ::-1], mirr)):
                t = torch.from_numpy(arr[..., ::-1].copy()).permute(2, 0, 1)[None].float() / 255
                out.append(float(m(t)[0, 0]))
            raw.append((r["u"] - 50) / 50)
            mean.append((r["u"] - 111.5) / 50)
    real, mirr, raw, mean = map(np.array, (real, mirr, raw, mean))
    res = {"run": a.run, "frames": len(rows),
           "real_vs_raw_2024_label_mae": round(float(np.abs(real - raw).mean()), 3),
           "real_vs_flipmean_label_mae": round(float(np.abs(real - mean).mean()), 3),
           "mirror_vs_negated_raw_label_mae": round(float(np.abs(mirr + raw).mean()), 3),
           "mean_x_real": round(float(real.mean()), 3), "mean_x_mirror": round(float(mirr.mean()), 3)}
    print(json.dumps(res, indent=1))
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
