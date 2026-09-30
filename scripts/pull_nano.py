"""Copy benchmark outputs from the Nano (ssh alias `nano`, dir ~/jdp) into results/nano/ and
parse the trtexec logs into results/nano/trt_bench.json.

  python scripts/pull_nano.py --clocks "pinned (jetson_clocks, GPU 921.6 MHz)"
"""
import argparse
import glob
import json
import os
import re
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "nano")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clocks", required=True, help="how clocks were set, recorded with the numbers")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    subprocess.run(["bash", "-c", f"scp -q 'nano:~/jdp/*_fp*.log' 'nano:~/jdp/loop_*.json' {OUT}/"], check=False)
    runs = []
    for log in sorted(glob.glob(os.path.join(OUT, "*_fp*.log"))):
        name = os.path.basename(log)[:-4]
        model, prec = name.rsplit("_", 1)
        text = open(log).read()
        m = re.search(r"GPU Compute Time: min = ([\d.]+) ms, max = ([\d.]+) ms, mean = ([\d.]+) ms, "
                      r"median = ([\d.]+) ms, percentile\(99%\) = ([\d.]+) ms", text)
        if not m:
            continue
        eng = subprocess.run(["ssh", "nano", f"stat -c %s ~/jdp/{name}.engine"], capture_output=True, text=True)
        runs.append({"model": model, "precision": prec, "median_ms": float(m.group(4)), "mean_ms": float(m.group(3)),
                     "p99_ms": float(m.group(5)), "min_ms": float(m.group(1)), "max_ms": float(m.group(2)),
                     "engine_mb": round(int(eng.stdout.strip() or 0) / 2**20, 1)})
    info = {"device": "Jetson Nano 4GB (Tegra X1, Maxwell, 128 CUDA cores), JetPack 4.6.1, TensorRT 8.2.1.8",
            "clocks": a.clocks, "runs": runs}
    json.dump(info, open(os.path.join(OUT, "trt_bench.json"), "w"), indent=1)
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
