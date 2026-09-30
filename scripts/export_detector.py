"""Export the retrained 2024 detector to ONNX (opset 11, static batch 1) at 640 and 320 for
TensorRT 8.2 on the Nano. Runs in the YOLOv5 venv:

  third_party/.venv-yolo/bin/python scripts/export_detector.py
"""
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
YOLO = os.path.join(ROOT, "third_party", "yolov5")
WEIGHTS = os.path.join(ROOT, "runs", "detector_2024", "weights", "best.pt")
OUT = os.path.join(ROOT, "results", "onnx")


def main():
    os.makedirs(OUT, exist_ok=True)
    for size in (640, 320):
        subprocess.run([sys.executable, os.path.join(YOLO, "export.py"), "--weights", WEIGHTS,
                        "--include", "onnx", "--opset", "11", "--imgsz", str(size), "--device", "cpu"],
                       check=True, cwd=YOLO)
        dst = os.path.join(OUT, f"yolo{size}.onnx")
        shutil.move(os.path.splitext(WEIGHTS)[0] + ".onnx", dst)
        print(f"wrote {dst} ({os.path.getsize(dst) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
