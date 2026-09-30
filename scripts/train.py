"""Train one arm with the 2024 recipe: Adam lr 1e-3, MSE, batch 16, 70 epochs, 90/10 random split.

Two held-out numbers are logged every epoch:
  test_loss  the 2024 metric: random 10% of the arm's own frames (same tracks as training)
  val_loss   frames from 20 tracks never used for training; this one picks the checkpoint
Closed-loop score is never used for selection.

  python scripts/train.py --arm r3_ratio --seed 0
  python scripts/train.py --arm dagger --seed 0 --manifest runs/dagger_s0/manifest.csv
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from jdp.data import ARMS, XYDataset, read_pool, sample_arm, val_rows  # noqa: E402
from jdp.labels import decode  # noqa: E402
from jdp.model import SteeringNet  # noqa: E402


def git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return None


def seed_worker(worker_id):
    np.random.seed(torch.initial_seed() % 2**32)


@torch.no_grad()
def evaluate(model, loader, device, label_mode):
    model.eval()
    loss, px, n = 0.0, 0.0, 0
    for img, xy in loader:
        pred = model(img.to(device)).cpu()
        loss += torch.nn.functional.mse_loss(pred, xy, reduction="sum").item() / 2
        pu, pv = decode(pred[:, 0], pred[:, 1], label_mode)
        tu, tv = decode(xy[:, 0], xy[:, 1], label_mode)
        px += torch.hypot(pu - tu, pv - tv).sum().item()
        n += len(xy)
    return loss / n, px / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pool", default="data/pool")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--manifest", help="train on these pool rows instead of sampling an arm")
    ap.add_argument("--label-mode", help="override the arm's label convention")
    ap.add_argument("--name", help="run directory name (default: <arm>); the seed is appended")
    ap.add_argument("--epochs", type=int, default=70)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    run = os.path.join(a.runs, f"{a.name or a.arm}_s{a.seed}")
    os.makedirs(run, exist_ok=True)
    rows = read_pool(a.pool)
    if a.manifest:
        with open(a.manifest) as f:
            arm_rows = list(csv.DictReader(f))
        for r in arm_rows:
            r["u"], r["v"] = float(r["u"]), float(r["v"])
        extra = {}
    else:
        arm_rows = sample_arm(rows, a.arm, a.seed)
        extra = ARMS[a.arm]
    label_mode = a.label_mode or extra.get("label_mode", "symmetric")
    with open(os.path.join(run, "manifest.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(arm_rows[0].keys()))
        w.writeheader()
        w.writerows(arm_rows)

    perm = np.random.default_rng(a.seed).permutation(len(arm_rows))
    n_test = int(0.1 * len(arm_rows))
    test_rows = [arm_rows[i] for i in perm[:n_test]]
    train_rows = [arm_rows[i] for i in perm[n_test:]]
    held = val_rows(rows)

    def loader(rs, train):
        ds = XYDataset(a.pool, rs, label_mode=label_mode, augment=train)
        return torch.utils.data.DataLoader(ds, batch_size=a.batch, shuffle=train, num_workers=a.workers,
                                           worker_init_fn=seed_worker, persistent_workers=True,
                                           generator=torch.Generator().manual_seed(a.seed))

    train_dl, test_dl, val_dl = loader(train_rows, True), loader(test_rows, False), loader(held, False)
    device = torch.device(a.device)
    model = SteeringNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)

    hist, best, t0 = [], None, time.time()
    for epoch in range(a.epochs):
        model.train()
        train_loss, n = 0.0, 0
        for img, xy in train_dl:
            img, xy = img.to(device), xy.to(device)
            opt.zero_grad()
            loss = torch.nn.functional.mse_loss(model(img), xy)
            loss.backward()
            opt.step()
            train_loss += float(loss) * len(xy)
            n += len(xy)
        test_loss, test_px = evaluate(model, test_dl, device, label_mode)
        val_loss, val_px = evaluate(model, val_dl, device, label_mode)
        hist.append({"epoch": epoch, "train_loss": train_loss / n, "test_loss": test_loss,
                     "test_px": test_px, "val_loss": val_loss, "val_px": val_px})
        if best is None or val_loss < best["val_loss"]:
            best = hist[-1]
            torch.save(model.state_dict(), os.path.join(run, "best.pt"))
        print(f"{a.arm} s{a.seed} ep{epoch:02d} train {train_loss / n:.4f} test {test_loss:.4f} "
              f"val {val_loss:.4f} ({val_px:.1f}px)  {time.time() - t0:.0f}s", flush=True)

    best_by_test = min(hist, key=lambda h: h["test_loss"])
    out = {"arm": a.arm, "seed": a.seed, "label_mode": label_mode, "pool": a.pool, "n_frames": len(arm_rows),
           "n_train": len(train_rows), "buckets": {b: sum(r["bucket"] == b for r in arm_rows)
                                                  for b in ("straight", "left", "right")},
           "best": best, "best_epoch_by_2024_test_loss": best_by_test["epoch"],
           "history": hist, "minutes": round((time.time() - t0) / 60, 2), "git": git_sha(),
           "recipe": {"epochs": a.epochs, "batch": a.batch, "lr": a.lr, "opt": "Adam", "loss": "MSE"}}
    with open(os.path.join(run, "train.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
