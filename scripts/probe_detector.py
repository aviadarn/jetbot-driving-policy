"""Does the safety stop work? Stop-sign detectors on the rendered probe set.

Detectors:
  2024_retrained  YOLOv5s fine-tuned with the 2024 recipe on the 2024 stop-sign set
  coco_yolov5s    stock COCO YOLOv5s, class 11 = "stop sign" (no fine-tuning)

For each input size (the 224 frame stretched to 224 / 320 / 640, as 2024 stretched to 640)
and channel order (RGB as trained, BGR as 2024 fed it):
  stop_rate   frames with >= 1 box: what the car acted on (2024 stopped on any box)
  hit_rate    frames with a box overlapping the true sign (IoU >= 0.3)
  false_stop  frames WITHOUT a sign that still produced a box

Runs in the YOLOv5 venv:
  third_party/.venv-yolo/bin/python scripts/probe_detector.py
"""
import csv
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

PROBE = os.path.join(ROOT, "data", "stop_probe")
DETECTORS = {
    "2024_retrained": (os.path.join(ROOT, "runs", "detector_2024", "weights", "best.pt"), 0),
    "coco_yolov5s": (os.path.join(ROOT, "third_party", "yolov5", "yolov5s.pt"), 11),
}


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def main():
    rows = list(csv.DictReader(open(os.path.join(PROBE, "labels.csv"))))
    imgs = {r["file"]: np.array(Image.open(os.path.join(PROBE, r["file"])).convert("RGB")) for r in rows}
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    results = []
    for name, (weights, cls) in DETECTORS.items():
        model = DetectMultiBackend(weights, device=device)
        for size in (224, 320, 640):
            for order in ("rgb", "bgr"):
                per = []
                for r in rows:
                    img = imgs[r["file"]]
                    img = img[..., ::-1] if order == "bgr" else img
                    t = torch.from_numpy(img.copy()).permute(2, 0, 1).float().div(255)[None].to(device)
                    t = torch.nn.functional.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
                    with torch.no_grad():
                        det = non_max_suppression(model(t), conf_thres=0.25, iou_thres=0.45, max_det=1000)[0].cpu().numpy()
                    det = det[det[:, 5] == cls] if len(det) else det
                    boxes = det[:, :4] * (224.0 / size) if len(det) else []
                    hit = False
                    if int(r["sign"]) and len(boxes):
                        gt = [float(r[k]) for k in ("x1", "y1", "x2", "y2")]
                        gt = [max(0.0, gt[0]), max(0.0, gt[1]), min(224.0, gt[2]), min(224.0, gt[3])]
                        hit = any(iou(b, gt) >= 0.3 for b in boxes)
                    per.append({"d": float(r["distance_m"]), "sign": int(r["sign"]), "stop": len(boxes) > 0,
                                "hit": hit, "conf": float(det[:, 4].max()) if len(det) else 0.0})
                pos = [p for p in per if p["sign"]]
                neg = [p for p in per if not p["sign"]]
                by_d = {}
                for p in pos:
                    b = by_d.setdefault(p["d"], [0, 0, 0])
                    b[0] += p["stop"]
                    b[1] += p["hit"]
                    b[2] += 1
                res = {"detector": name, "size": size, "order": order,
                       "stop_rate": round(np.mean([p["stop"] for p in pos]), 3),
                       "hit_rate": round(np.mean([p["hit"] for p in pos]), 3),
                       "false_stop": round(np.mean([p["stop"] for p in neg]), 3),
                       "by_distance": {f"{d:.1f}": {"stop": v[0] / v[2], "hit": v[1] / v[2]} for d, v in sorted(by_d.items())},
                       "n_pos": len(pos), "n_neg": len(neg)}
                results.append(res)
                print(f"{name:15s} {size:3d} {order}: stop {res['stop_rate']:.2f}  hit {res['hit_rate']:.2f}  "
                      f"false_stop {res['false_stop']:.2f}", flush=True)
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    with open(os.path.join(ROOT, "results", "detector_probe.json"), "w") as f:
        json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
