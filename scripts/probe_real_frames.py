"""The detectors on real frames from the 2024 car (no stop sign in any of them).

The frames are cut from the 2024 README screenshots (docs/original_2024/): a 3x3 grid and a
1x3 strip of 224x224 camera images from the hallway course. Each is fed the 2024 way:
stretched to 640, BGR and RGB. A detector that "stops" here would have stopped the car.

  third_party/.venv-yolo/bin/python scripts/probe_real_frames.py
"""
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "third_party", "yolov5"))
from models.common import DetectMultiBackend  # noqa: E402
from utils.general import non_max_suppression  # noqa: E402

SHOTS = [("Screenshot_2023-11-02_at_3.44.47_PM.png", 3, 3), ("Screenshot_2023-11-02_at_3.43.33_PM.png", 1, 3)]
DETECTORS = {"2024_retrained": ("runs/detector_2024/weights/best.pt", 0),
             "coco_yolov5s": ("third_party/yolov5/yolov5s.pt", 11)}


def frames():
    out = []
    for name, rows, cols in SHOTS:
        im = np.array(Image.open(os.path.join(ROOT, "docs", "original_2024", name)).convert("RGB"))
        h, w = im.shape[:2]
        for r in range(rows):
            for c in range(cols):
                tile = im[r * h // rows:(r + 1) * h // rows, c * w // cols:(c + 1) * w // cols]
                out.append(np.array(Image.fromarray(tile).resize((224, 224), Image.BILINEAR)))
    return out


def main():
    imgs = frames()
    res = []
    for name, (weights, cls) in DETECTORS.items():
        model = DetectMultiBackend(os.path.join(ROOT, weights), device=torch.device("cpu"))
        for order in ("bgr", "rgb"):
            fired, areas, confs = 0, [], []
            for img in imgs:
                x = img[..., ::-1] if order == "bgr" else img
                t = torch.from_numpy(x.copy()).permute(2, 0, 1).float().div(255)[None]
                t = torch.nn.functional.interpolate(t, size=(640, 640), mode="bilinear", align_corners=False)
                with torch.no_grad():
                    d = non_max_suppression(model(t), 0.25, 0.45)[0].numpy()
                d = d[d[:, 5] == cls] if len(d) else d
                if len(d):
                    fired += 1
                    b = d[d[:, 4].argmax()]
                    areas.append(float((b[2] - b[0]) * (b[3] - b[1]) / 640 ** 2))
                    confs.append(float(b[4]))
            res.append({"detector": name, "order": order, "frames": len(imgs), "stopped": fired,
                        "median_box_area_frac": round(float(np.median(areas)), 3) if areas else None,
                        "median_conf": round(float(np.median(confs)), 3) if confs else None})
            print(res[-1])
    with open(os.path.join(ROOT, "results", "detector_real2024.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
