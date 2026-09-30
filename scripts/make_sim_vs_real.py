"""Real frames from the 2024 car (top) above simulator frames (bottom), same 224x224 size.

  python scripts/make_sim_vs_real.py --out assets/sim_vs_2024.jpg
"""
import argparse
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from jdp.sim import Sim  # noqa: E402
from jdp.track import Track  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="assets/sim_vs_2024.jpg")
    ap.add_argument("--tile", type=int, default=224)
    a = ap.parse_args()
    strip = Image.open("docs/original_2024/Screenshot_2023-11-02_at_3.43.33_PM.png").convert("RGB")
    w, h = strip.size
    real = [strip.crop((k * w // 3, 0, (k + 1) * w // 3, h)).resize((a.tile, a.tile), Image.BILINEAR) for k in range(3)]
    sim_frames = []
    for seed, s, lat in ((1000, 3.0, 0.1), (1003, 12.0, -0.15), (1007, 20.0, 0.2)):
        tr = Track(seed)
        sim = Sim(tr, look_seed=seed)
        sim_frames.append(Image.fromarray(sim.render(tr.pose_at(s, lateral=lat))).resize((a.tile, a.tile)))
        sim.close()
    gap, lab = 6, 22
    W = 3 * a.tile + 2 * gap
    out = Image.new("RGB", (W, 2 * (a.tile + lab) + gap), (252, 252, 251))
    d = ImageDraw.Draw(out)
    for row, (frames, text) in enumerate(((real, "the 2024 car's camera"), (sim_frames, "the simulator"))):
        y = row * (a.tile + lab + gap)
        d.text((2, y + 5), text, fill=(82, 81, 78))
        for k, f in enumerate(frames):
            out.paste(f, (k * (a.tile + gap), y + lab))
    out.save(a.out, quality=85, optimize=True)
    print(f"wrote {a.out} {out.size} {os.path.getsize(a.out) / 1e3:.0f} KB")


if __name__ == "__main__":
    main()
