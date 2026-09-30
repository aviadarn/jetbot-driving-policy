"""The on-car loop: camera -> steering net -> (detector every Nth frame) -> 2024 controller.

--dry-run computes motor commands but never touches the motor board. Timing is
frame-arrival-to-command: from cap.read() returning a frame to the command being ready.
It does not include the ISP/queue delay before the frame arrives, or motor response.

  python3 loop.py --policy policy_fp16.engine --dry-run --seconds 60
  python3 loop.py --policy policy_fp16.engine --detector yolo320_fp16.engine --det-size 320 \
                  --det-every 4 --dry-run --seconds 60 --out results_improved.json
  python3 loop.py --policy policy_fp32.engine --detector yolo640_fp32.engine --det-size 640 \
                  --det-every 1 --det-bgr --dry-run --seconds 60 --out results_2024cfg.json
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from controller import Controller2024  # noqa: E402  (copied from jdp/controller.py)
from trt_infer import TRTModel, yolo_boxes  # noqa: E402

GST = ("nvarguscamerasrc sensor-mode=3 ! video/x-raw(memory:NVMM), width=816, height=616, "
       "format=NV12, framerate=21/1 ! nvvidconv ! video/x-raw, width=224, height=224, format=BGRx ! "
       "videoconvert ! video/x-raw, format=BGR ! appsink drop=true max-buffers=1 sync=false")


class Tegrastats(threading.Thread):
    """Samples RAM and total board power (VDD_IN) while the loop runs."""

    def __init__(self):
        threading.Thread.__init__(self, daemon=True)
        self.ram, self.power = [], []
        self.proc = subprocess.Popen(["tegrastats", "--interval", "250"], stdout=subprocess.PIPE,
                                     universal_newlines=True)

    def run(self):
        for line in self.proc.stdout:
            parts = line.split()
            try:
                self.ram.append(int(parts[parts.index("RAM") + 1].split("/")[0]))
                self.power.append(int(parts[parts.index("POM_5V_IN") + 1].split("/")[0]))
            except (ValueError, IndexError):
                pass

    def stop(self):
        self.proc.terminate()


def pct(a, q):
    return round(float(np.percentile(a, q)), 2) if len(a) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--detector")
    ap.add_argument("--det-size", type=int, default=320)
    ap.add_argument("--det-every", type=int, default=1)
    ap.add_argument("--det-bgr", action="store_true", help="feed BGR like 2024 (trained on RGB)")
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default="loop_results.json")
    a = ap.parse_args()
    if not a.dry_run:
        sys.exit("motor output is Phase 6 and needs the board reconnected; use --dry-run")

    policy = TRTModel(a.policy)
    det = TRTModel(a.detector) if a.detector else None
    ctl = Controller2024()
    cap = cv2.VideoCapture(GST, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        sys.exit("camera pipeline failed to open")
    for _ in range(10):  # let auto-exposure settle
        cap.read()

    ts = Tegrastats()
    ts.start()
    t_pol, t_det, t_total, arrivals, stops = [], [], [], [], 0
    t_end = time.time() + a.seconds
    k = 0
    while time.time() < t_end:
        ok, frame = cap.read()
        t0 = time.perf_counter()
        if not ok:
            continue
        arrivals.append(t0)
        x = frame.astype(np.float32).transpose(2, 0, 1)[None] / 255.0  # BGR/255, as trained
        xy = policy(x)[0].reshape(-1)
        t1 = time.perf_counter()
        stop = False
        if det is not None and k % a.det_every == 0:
            img = frame if a.det_bgr else frame[..., ::-1]
            img = cv2.resize(img, (a.det_size, a.det_size), interpolation=cv2.INTER_LINEAR)
            d = img.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
            stop = len(yolo_boxes(det(d)[0])) > 0
            t_det.append((time.perf_counter() - t1) * 1e3)
        throttle, steer = ctl.step(xy, stop=stop)
        t2 = time.perf_counter()
        stops += int(stop)
        t_pol.append((t1 - t0) * 1e3)
        t_total.append((t2 - t0) * 1e3)
        k += 1
    ts.stop()
    cap.release()

    period = np.diff(arrivals) * 1e3
    res = {
        "frames": k, "seconds": a.seconds, "fps": round(k / a.seconds, 2),
        "policy": os.path.basename(a.policy), "detector": a.detector and os.path.basename(a.detector),
        "det_size": a.det_size, "det_every": a.det_every, "det_bgr": a.det_bgr,
        "policy_ms": {"p50": pct(t_pol, 50), "p95": pct(t_pol, 95), "p99": pct(t_pol, 99)},
        "detector_ms": {"p50": pct(t_det, 50), "p95": pct(t_det, 95), "p99": pct(t_det, 99)},
        "arrival_to_command_ms": {"p50": pct(t_total, 50), "p95": pct(t_total, 95), "p99": pct(t_total, 99)},
        "frame_period_ms": {"p50": pct(period, 50), "p95": pct(period, 95)},
        "stop_frames": stops,
        "ram_mb": {"max": max(ts.ram) if ts.ram else None},
        "board_power_mw": {"mean": round(float(np.mean(ts.power)), 0) if ts.power else None},
    }
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
