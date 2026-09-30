"""Every number in the README comes from here: read results/*.json and runs/*/train.json,
write figures to assets/ and rewrite the README blocks between <!-- BEGIN:x --> / <!-- END:x -->.

  python scripts/make_figures.py
"""
import glob
import json
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from scripts.eval import wilson  # noqa: E402

MAIN = [("natural", "natural: 3000 clean expert-drive frames"),
        ("r1_ratio", "R1 ratio (800:600:400)"),
        ("uniform", "uniform (R2, 1:1:1)"),
        ("r3_ratio", "R3 ratio (2500:1000:1000)"),
        ("dagger", "DAgger: 1500 drive + 1500 on-policy")]
HIST = [("hist_r1", "R1: 1800 (800/600/400)"),
        ("hist_r2", "R2: 3000 (1000/1000/1000)"),
        ("hist_r3", "R3: 4500 (2500/1000/1000)"),
        ("hist_r3_bug", "R3 with the 2024 labels + flip")]

# Reference palette (dataviz skill, references/palette.md): chrome + categorical slots.
THEMES = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
              "grid": "#e1e0d9", "axis": "#c3c2b7", "series": ["#2a78d6", "#eb6834", "#1baf7a"]},
    "dark": {"surface": "#1a1a19", "ink": "#ffffff", "ink2": "#c3c2b7", "muted": "#898781",
             "grid": "#2c2c2a", "axis": "#383835", "series": ["#3987e5", "#d95926", "#199e70"]},
}


def load(path):
    with open(path) as f:
        return json.load(f)


def eval_of(name):
    p = os.path.join(ROOT, "results", "eval", f"{name}.json")
    return load(p) if os.path.exists(p) else None


def arm_stats(arm):
    """Pool episodes over seeds; keep per-seed rates and offline numbers."""
    seeds = []
    for tj in sorted(glob.glob(os.path.join(ROOT, "runs", f"{arm}_s[0-9]", "train.json"))):
        run = os.path.basename(os.path.dirname(tj))
        ev = eval_of(run)
        if ev is None:
            continue
        tr = load(tj)
        seeds.append({"run": run, "k": ev["summary"]["complete"], "n": ev["summary"]["episodes"],
                      "rate": ev["summary"]["rate"], "lat": ev["summary"]["mean_abs_lateral_m"],
                      "weave": ev["summary"]["weave_per_m"], "sat": ev["summary"]["steer_saturated_frac"],
                      "status": ev["summary"]["status"],
                      "val_px": tr["best"]["val_px"], "test_px": tr["best"]["test_px"],
                      "test_loss": tr["best"]["test_loss"], "val_loss": tr["best"]["val_loss"],
                      "n_frames": tr["n_frames"], "buckets": tr["buckets"]})
    if not seeds:
        return None
    k, n = sum(s["k"] for s in seeds), sum(s["n"] for s in seeds)
    lo, hi = wilson(k, n)
    status = {}
    for s in seeds:
        for key, v in s["status"].items():
            status[key] = status.get(key, 0) + v
    mean = lambda key: float(np.mean([s[key] for s in seeds if s[key] is not None]))  # noqa: E731
    return {"arm": arm, "seeds": seeds, "k": k, "n": n, "rate": k / n, "lo": lo, "hi": hi,
            "lat": mean("lat"), "weave": mean("weave"), "sat": mean("sat"), "val_px": mean("val_px"),
            "test_px": mean("test_px"), "status": status, "n_frames": seeds[0]["n_frames"]}


def pct(x):
    return f"{100 * x:.0f}%"


def ci(s):
    return f"{pct(s['rate'])} ({s['k']}/{s['n']}) [{pct(s['lo'])}–{pct(s['hi'])}]"


def failures(status):
    parts = [f"{v} {k.replace('_', ' ')}" for k, v in sorted(status.items()) if k != "complete"]
    return ", ".join(parts) if parts else "—"


def table_arms(rows):
    out = ["| Arm | Frames | Complete (pooled seeds) [Wilson 95%] | Per seed | Held-out click error | "
           "2024-style test error | Lateral error | Weave /m | Steer saturated | Failures |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for arm, label, s in rows:
        if s is None:
            out.append(f"| {label} | — | not run yet | | | | | | | |")
            continue
        per = " · ".join(f"{x['k']}/{x['n']}" for x in s["seeds"])
        out.append(f"| {label} | {s['n_frames']} | {ci(s)} | {per} | {s['val_px']:.1f} px | {s['test_px']:.1f} px | "
                   f"{100 * s['lat']:.1f} cm | {s['weave']:.3f} | {pct(s['sat'])} | {failures(s['status'])} |")
    return "\n".join(out)


def table_baselines():
    out = ["| Clicks (no learning) | Complete [Wilson 95%] | Steer saturated | Weave /m | Failures |",
           "|---|---|---|---|---|"]
    for name, label in [("expert", "perfect clicks, centred labels `(px−112)/112`"),
                        ("expert_2024", "perfect clicks, 2024 labels `(px−50)/50`"),
                        ("expert_2024_flipmean", "what a perfect 2024 net learns under the buggy flip"),
                        ("zero", "always (0, 0): floor")]:
        ev = eval_of(name)
        if ev is None:
            continue
        s = ev["summary"]
        out.append(f"| {label} | {s['complete']}/{s['episodes']} [{pct(s['wilson95'][0])}–{pct(s['wilson95'][1])}] | "
                   f"{pct(s['steer_saturated_frac'])} | {s['weave_per_m']:.3f} | {failures(s['status'])} |")
    return "\n".join(out)


def table_detector():
    p = os.path.join(ROOT, "results", "detector_probe.json")
    if not os.path.exists(p):
        return "_detector probe not run yet_"
    res = load(p)
    out = ["| Detector | Input | Channels | Stops with sign in view | Box on the sign | False stops (no sign) |",
           "|---|---|---|---|---|---|"]
    for r in res:
        tag = " ← 2024 config" if (r["detector"], r["size"], r["order"]) == ("2024_retrained", 640, "bgr") else ""
        out.append(f"| {r['detector'].replace('_', ' ')}{tag} | {r['size']} | {r['order'].upper()} | "
                   f"{pct(r['stop_rate'])} | {pct(r['hit_rate'])} | {pct(r['false_stop'])} |")
    return "\n".join(out)


def table_nano():
    files = sorted(glob.glob(os.path.join(ROOT, "results", "nano", "loop_*.json")))
    bench = os.path.join(ROOT, "results", "nano", "trt_bench.json")
    out = []
    if os.path.exists(bench):
        b = load(bench)
        out += ["| Engine | Precision | GPU compute median | p99 | Engine size |", "|---|---|---|---|---|"]
        for r in b["runs"]:
            out.append(f"| {r['model']} | {r['precision'].upper()} | {r['median_ms']:.1f} ms | {r['p99_ms']:.1f} ms | "
                       f"{r['engine_mb']:.1f} MB |")
        out.append("")
        out.append(f"_Clocks: {b['clocks']}._")
        out.append("")
    if files:
        out += ["| Loop | FPS | Preprocess p50 | Policy p50 | Detector p50 | Frame-arrival → command p50 / p95 | RAM (system) |",
                "|---|---|---|---|---|---|---|"]
        for fpath in files:
            d = load(fpath)
            det = f"{d['detector_ms']['p50']:.1f} ms" if d["detector_ms"]["p50"] else "—"
            prep = f"{d['prep_ms']['p50']:.1f} ms" if d.get("prep_ms") else "—"
            out.append(f"| {d.get('label', os.path.basename(fpath))} | {d['fps']:.1f} | {prep} | {d['policy_ms']['p50']:.1f} ms | "
                       f"{det} | {d['arrival_to_command_ms']['p50']:.1f} / {d['arrival_to_command_ms']['p95']:.1f} ms | "
                       f"{d['ram_mb']['max']} MB |")
    return "\n".join(out) if out else "_Nano not measured yet_"


# ---------- figures ----------

def _style(ax, t):
    ax.set_facecolor(t["surface"])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(t["axis"])
    ax.tick_params(colors=t["muted"], length=0, labelsize=9)
    ax.xaxis.grid(True, color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)


def fig_arms(stats, fname):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [(label, s) for _, label, s in stats if s is not None]
    if not rows:
        return
    for mode, t in THEMES.items():
        fig, ax = plt.subplots(figsize=(7.2, 0.45 * len(rows) + 1.0), facecolor=t["surface"])
        _style(ax, t)
        c = t["series"][0]
        for i, (label, s) in enumerate(rows):
            y = len(rows) - 1 - i
            ax.plot([100 * s["lo"], 100 * s["hi"]], [y, y], color=c, linewidth=2, solid_capstyle="round")
            for sd in s["seeds"]:
                ax.plot(100 * sd["rate"], y, "o", ms=4, color=t["muted"], alpha=0.8, zorder=3)
            ax.plot(100 * s["rate"], y, "o", ms=9, color=c, markeredgecolor=t["surface"], markeredgewidth=2, zorder=4)
            ax.text(101.5, y, f"{100 * s['rate']:.0f}%", va="center", fontsize=9, color=t["ink"])
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([label for label, _ in rows][::-1], color=t["ink2"], fontsize=9)
        ax.set_xlim(0, 108)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xlabel("corridors completed (%) · big dot pooled over seeds, bar Wilson 95%, small dots per seed",
                      color=t["muted"], fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(ROOT, "assets", f"{fname}_{mode}.png"), dpi=160, facecolor=t["surface"])
        plt.close(fig)


def fig_offline_vs_closed(stats, fname):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    runs = [(arm, sd) for arm, _, s in stats if s for sd in s["seeds"]]
    if not runs:
        return
    for mode, t in THEMES.items():
        fig, axes = plt.subplots(1, 2, figsize=(8, 3.2), facecolor=t["surface"], sharey=True)
        for ax, key, title in [(axes[0], "test_px", "2024-style test error\n(random 10% of the same corridors)"),
                               (axes[1], "val_px", "held-out corridors")]:
            _style(ax, t)
            ax.yaxis.grid(True, color=t["grid"], linewidth=0.8)
            xs = [sd[key] for _, sd in runs]
            ys = [100 * sd["rate"] for _, sd in runs]
            ax.plot(xs, ys, "o", ms=7, color=t["series"][0], markeredgecolor=t["surface"], markeredgewidth=1.5)
            done = set()
            for (arm, sd), x, y in zip(runs, xs, ys):
                if arm in done:
                    continue
                done.add(arm)
                ax.annotate(arm, (x, y), textcoords="offset points", xytext=(5, 4), fontsize=7, color=t["ink2"])
            ax.set_xlabel(f"click error (px) — {title}", color=t["muted"], fontsize=8)
            ax.set_xscale("log")
        axes[0].set_ylabel("closed-loop completion (%)", color=t["muted"], fontsize=8)
        axes[0].set_ylim(-5, 105)
        fig.tight_layout()
        fig.savefig(os.path.join(ROOT, "assets", f"{fname}_{mode}.png"), dpi=160, facecolor=t["surface"])
        plt.close(fig)


def rewrite_readme(blocks):
    path = os.path.join(ROOT, "README.md")
    if not os.path.exists(path):
        return
    text = open(path).read()
    for key, body in blocks.items():
        pat = re.compile(rf"(<!-- BEGIN:{key} -->\n).*?(\n<!-- END:{key} -->)", re.S)
        text = pat.sub(lambda m: m.group(1) + body + m.group(2), text)
    open(path, "w").write(text)


def main():
    os.makedirs(os.path.join(ROOT, "assets"), exist_ok=True)
    main_stats = [(a, lbl, arm_stats(a)) for a, lbl in MAIN]
    hist_stats = [(a, lbl, arm_stats(a)) for a, lbl in HIST]
    fig_arms(main_stats, "fig_arms")
    fig_arms(hist_stats, "fig_history")
    fig_offline_vs_closed(main_stats + hist_stats, "fig_offline_vs_closed")
    blocks = {"baselines": table_baselines(), "arms": table_arms(main_stats),
              "history": table_arms(hist_stats), "detector": table_detector(), "nano": table_nano()}
    rewrite_readme(blocks)
    summary = {k: [{"arm": a, **({kk: vv for kk, vv in s.items() if kk != "seeds"} if s else {})}
                   for a, _, s in v] for k, v in (("main", main_stats), ("history", hist_stats))}
    with open(os.path.join(ROOT, "results", "summary.json"), "w") as f:
        json.dump(summary, f, indent=1, default=float)
    for k, v in blocks.items():
        print(f"== {k} ==\n{v}\n")


if __name__ == "__main__":
    main()
