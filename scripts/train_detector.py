"""Retrain the 2024 stop-sign detector with the 2024 recipe (object_detection/yolo_training.ipynb):

  split:  sklearn train_test_split(train/images, test_size=0.1, random_state=2000)
  train:  yolov5/train.py --weights yolov5s.pt --cfg yolov5s.yaml --epochs 30 --batch-size 16 --img 640

The original list order came from a Colab glob, so the exact split can't be recovered; the
file list is sorted here, then split with the same call. Runs in the YOLOv5 venv
(third_party/.venv-yolo) because YOLOv5 is AGPL-3.0 and not vendored into this repo.

  third_party/.venv-yolo/bin/python scripts/train_detector.py
"""
import glob
import os
import subprocess
import sys

from sklearn.model_selection import train_test_split

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "stop2024")
YOLO = os.path.join(ROOT, "third_party", "yolov5")


def main():
    imgs = sorted(glob.glob(os.path.join(DATA, "train", "images", "*.jpg")))
    tr, va = train_test_split(imgs, test_size=0.1, random_state=2000)
    for name, lst in (("train_2024.txt", tr), ("val_2024.txt", va)):
        with open(os.path.join(DATA, name), "w") as f:
            f.write("\n".join(lst) + "\n")
    with open(os.path.join(DATA, "split_2024.yaml"), "w") as f:
        f.write(f"path: {DATA}\ntrain: train_2024.txt\nval: val_2024.txt\nnc: 1\nnames: ['Stop sign']\n")
    print(f"split: {len(tr)} train / {len(va)} val (2024: 576 / 64)")
    # YOLOv5 enables torch.use_deterministic_algorithms(True); MPS has no deterministic
    # index_put, so downgrade that to warn-only and run train.py unchanged.
    launcher = ("import sys, runpy, torch; _d = torch.use_deterministic_algorithms; "
                "torch.use_deterministic_algorithms = lambda mode, warn_only=False: _d(mode, warn_only=True); "
                "sys.argv = sys.argv[1:]; runpy.run_path(sys.argv[0], run_name='__main__')")
    cmd = [sys.executable, "-c", launcher, os.path.join(YOLO, "train.py"), "--weights", "yolov5s.pt",
           "--cfg", os.path.join(YOLO, "models", "yolov5s.yaml"), "--data", os.path.join(DATA, "split_2024.yaml"),
           "--epochs", "30", "--batch-size", "16", "--img", "640", "--device", "mps", "--workers", "2",
           "--project", os.path.join(ROOT, "runs"), "--name", "detector_2024", "--exist-ok"]
    subprocess.run(cmd, check=True, cwd=YOLO)


if __name__ == "__main__":
    main()
