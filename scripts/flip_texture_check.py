"""Is the bug net's mirror detection a simulator artefact? Re-render held-out poses with a
floor texture it never saw (a different speckle seed), then compare predictions on each frame
and its mirror image. A fixed, repeating floor tile is mirror-detectable; a real speckled
floor is not, so if the asymmetry vanishes on the new texture, the tile was the cue.

  python scripts/flip_texture_check.py --run runs/hist_r3_bug_s0
"""
import argparse
import json
import os
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import jdp.scene as scene  # noqa: E402
from jdp.data import read_pool, val_rows  # noqa: E402
from jdp.model import SteeringNet  # noqa: E402
from jdp.track import Track  # noqa: E402


def predict(m, img):
    t = torch.from_numpy(img[..., ::-1].copy()).permute(2, 0, 1)[None].float() / 255
    with torch.no_grad():
        return float(m(t)[0, 0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/hist_r3_bug_s0")
    ap.add_argument("--n", type=int, default=240)
    ap.add_argument("--out", default="results/flip_texture_check.json")
    a = ap.parse_args()
    m = SteeringNet(pretrained=False)
    m.load_state_dict(torch.load(os.path.join(a.run, "best.pt"), map_location="cpu"))
    m.eval()
    rows = [r for r in val_rows(read_pool("data/pool")) if r["source"] == "perturbed"][: a.n]
    out = {}
    for tex_seed in (0, 7):  # 0 = the training texture, 7 = unseen
        tmp = tempfile.mkdtemp()
        orig = scene._terrazzo
        scene.ASSET_DIR = tmp
        scene._terrazzo = lambda path, seed=0, size=512, _o=orig, _s=tex_seed: _o(path, seed=_s, size=size)
        from jdp.sim import Sim
        real, mirr, raw = [], [], []
        by_track = {}
        for r in rows:
            by_track.setdefault(r["track"], []).append(r)
        for tseed, rs in by_track.items():
            tr = Track(tseed)
            sim = Sim(tr)
            for r in rs:
                pose = tr.pose_at(r["s"], lateral=r["lateral"])
                pose[2] += r["dheading"]
                img = sim.render(pose)
                real.append(predict(m, img))
                mirr.append(predict(m, img[:, ::-1]))
                raw.append((r["u"] - 50) / 50)
            sim.close()
        scene._terrazzo = orig
        real, mirr, raw = map(np.array, (real, mirr, raw))
        out[f"texture_seed_{tex_seed}"] = {
            "frames": len(real), "mean_x_real": round(float(real.mean()), 3),
            "mean_x_mirror": round(float(mirr.mean()), 3),
            "real_vs_raw_2024_label_mae": round(float(np.abs(real - raw).mean()), 3),
            "real_vs_flipmean_label_mae": round(float(np.abs(real - (raw - 1.23)).mean()), 3),
            "asymmetry_mean_real_plus_mirror": round(float((real + mirr).mean()), 3)}
        print(tex_seed, json.dumps(out[f"texture_seed_{tex_seed}"]), flush=True)
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
