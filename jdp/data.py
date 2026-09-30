"""Experiment arms and the 2024 training-time data pipeline.

Arms are bucket counts (straight, left, right) sampled from the pool. All main arms are
N=3000 so only the distribution changes; the `hist_*` arms reproduce the literal 2024 rounds
(nested: R1 is a subset of R2, R2 of R3, like data that was added each round).

2024 augmentation, kept as-is (rf_train_model.ipynb XYDataset.__getitem__):
  hflip at p=0.5 with x = -x on EVERY sample (the random_hflips flag was ignored),
  ColorJitter(0.3, 0.3, 0.3, 0.3), resize 224, to_tensor, RGB->BGR via [::-1].
ImageNet normalisation lives in the model (jdp/model.py).
"""
import csv
import os

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from .labels import encode, flip_x

BUCKETS = ("straight", "left", "right")


def _ratio(a, b, c, n=3000):
    counts = np.array([a, b, c], float) / (a + b + c) * n
    counts = np.floor(counts).astype(int)
    counts[0] += n - counts.sum()
    return tuple(int(v) for v in counts)


ARMS = {
    # distribution arms, all N=3000
    "r1_ratio": {"source": "perturbed", "counts": _ratio(800, 600, 400)},
    "uniform": {"source": "perturbed", "counts": (1000, 1000, 1000)},
    "r3_ratio": {"source": "perturbed", "counts": _ratio(2500, 1000, 1000)},
    "natural": {"source": "drive", "n": 3000},
    # literal 2024 rounds (sizes differ; nested)
    "hist_r1": {"source": "perturbed", "counts": (800, 600, 400)},
    "hist_r2": {"source": "perturbed", "counts": (1000, 1000, 1000)},
    "hist_r3": {"source": "perturbed", "counts": (2500, 1000, 1000)},
    # R3 as 2024 actually trained it: 2024 label convention + the always-on x=-x flip
    "hist_r3_bug": {"source": "perturbed", "counts": (2500, 1000, 1000), "label_mode": "2024"},
}


def read_pool(pool_dir):
    with open(os.path.join(pool_dir, "pool.csv")) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("s", "lateral", "dheading", "u", "v", "kappa_ahead"):
            r[k] = float(r[k])
        r["track"] = int(r["track"])
    return rows


def sample_arm(rows, arm, seed):
    """Rows for an arm. Same seed -> same shuffled order per bucket, so arms that ask for
    fewer frames get a prefix of what larger arms get (nesting)."""
    spec = ARMS[arm]
    rng = np.random.default_rng(seed)
    train = [r for r in rows if r["split"] == "train" and r["source"] == spec["source"]]
    if "n" in spec:
        idx = rng.permutation(len(train))[: spec["n"]]
        return [train[i] for i in idx]
    out = []
    for b, n in zip(BUCKETS, spec["counts"]):
        pool = [r for r in train if r["bucket"] == b]
        order = np.random.default_rng([seed, BUCKETS.index(b)]).permutation(len(pool))
        if n > len(pool):
            raise ValueError(f"{arm}: need {n} {b} frames, pool has {len(pool)}")
        out += [pool[i] for i in order[:n]]
    rng.shuffle(out)
    return out


def val_rows(rows, n=1000, seed=0):
    """Fixed held-out-track set for checkpoint selection: half snapshots, half drives."""
    rng = np.random.default_rng(seed)
    out = []
    for src in ("perturbed", "drive"):
        pool = [r for r in rows if r["split"] == "val" and r["source"] == src]
        out += [pool[i] for i in rng.permutation(len(pool))[: n // 2]]
    return out


class XYDataset(torch.utils.data.Dataset):
    def __init__(self, pool_dir, rows, label_mode="symmetric", augment=True, faithful_flip=True):
        self.pool_dir = pool_dir
        self.rows = rows
        self.label_mode = label_mode
        self.augment = augment
        self.faithful_flip = faithful_flip
        self.color_jitter = transforms.ColorJitter(0.3, 0.3, 0.3, 0.3)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        image = Image.open(os.path.join(self.pool_dir, r["path"])).convert("RGB")
        x, y = encode(r["u"], r["v"], self.label_mode)
        if self.augment:
            if np.random.rand() > 0.5:  # 2024: float(np.random.rand(1)); numpy 2.5 rejects that form
                image = transforms.functional.hflip(image)
                x = flip_x(x, self.label_mode, self.faithful_flip)
            image = self.color_jitter(image)
        image = transforms.functional.resize(image, (224, 224))
        image = transforms.functional.to_tensor(image)
        image = torch.from_numpy(image.numpy()[::-1].copy())  # RGB -> BGR, as in 2024
        return image, torch.tensor([x, y]).float()
