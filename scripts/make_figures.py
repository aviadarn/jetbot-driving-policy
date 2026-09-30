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
# hist_r2 is not listed: R2 was the uniform 1000/1000/1000 set, so with nested sampling and the
# same seed it is the `uniform` run, frame for frame (its eval JSONs are identical).
HIST = [("hist_r1", "R1: 1800 frames (800/600/400)"),
        ("uniform", "R2: 3000 frames (1000/1000/1000)"),
        ("hist_r3", "R3: 4500 frames (2500/1000/1000)")]
# Round 3 again on renderer v2 (mirror-symmetric anti-aliasing), identical frames, two label pipelines.
PAIR = [("hist_r3_v2", "R3, centred labels"),
        ("hist_r3_bug_v2", "R3, 2024 labels + always-on flip")]

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


CONDITIONS = [("", "nominal"), ("recovery", "recovery start"), ("lat3", "150 ms loop")]


def arm_stats(arm, cond=""):
    """Pool episodes over seeds for one eval condition; keep per-seed rates and offline numbers."""
    seeds = []
    for tj in sorted(glob.glob(os.path.join(ROOT, "runs", f"{arm}_s[0-9]", "train.json"))):
        run = os.path.basename(os.path.dirname(tj))
        ev = eval_of(run + (f"_{cond}" if cond else ""))
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
    return f"{s['k']}/{s['n']} [{pct(s['lo'])}–{pct(s['hi'])}]"


def failures(status):
    parts = [f"{v} {k.replace('_', ' ')}" for k, v in sorted(status.items()) if k != "complete"]
    return ", ".join(parts) if parts else "—"


def table_arms(arms):
    heads = " | ".join(f"Complete, {lbl}" for _, lbl in CONDITIONS)
    out = [f"| Arm | Frames | Held-out click error | 2024-style test error | {heads} | Weave /m (nominal) | Failures (all conditions) |",
           "|---|---|---|---|" + "---|" * len(CONDITIONS) + "---|---|"]
    for arm, label in arms:
        per = [arm_stats(arm, c) for c, _ in CONDITIONS]
        base = per[0]
        if base is None:
            out.append(f"| {label} | — | not run yet |" + " |" * (len(CONDITIONS) + 3))
            continue
        cells = [ci(p) if p else "—" for p in per]
        fails = {}
        for p in per:
            for k, v in (p["status"] if p else {}).items():
                fails[k] = fails.get(k, 0) + v
        out.append(f"| {label} | {base['n_frames']} | {base['val_px']:.1f} px | {base['test_px']:.1f} px | "
                   + " | ".join(cells) + f" | {base['weave']:.3f} | {failures(fails)} |")
    return "\n".join(out)


def table_flipcheck():
    rows = [("results/flip_check.json", "renderer v1 (pyrDown anti-aliasing)"),
            ("results/flip_check_v2.json", "renderer v2 (mirror-symmetric)")]
    out = ["| Bug-faithful net, trained on | Mean x on real frames | Mean x on the same frames mirrored | "
           "Distance to raw 2024 label | Distance to the flip-average |", "|---|---|---|---|---|"]
    n = 0
    for path, label in rows:
        path = os.path.join(ROOT, path)
        if not os.path.exists(path):
            continue
        d = load(path)
        n += 1
        out.append(f"| {label} | {d['mean_x_real']:+.2f} | {d['mean_x_mirror']:+.2f} | "
                   f"{d['real_vs_raw_2024_label_mae']:.2f} | {d['real_vs_flipmean_label_mae']:.2f} |")
    return "\n".join(out) if n else "_flip check not run yet_"


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


def _range(by_distance, thresh=0.9):
    """Farthest distance d such that every probe distance <= d localises the sign >= thresh."""
    best = 0.0
    for d, v in sorted(by_distance.items(), key=lambda kv: float(kv[0])):
        if v["hit"] < thresh:
            break
        best = float(d)
    return f"{best:.1f} m" if best else "none"


def table_detector():
    p = os.path.join(ROOT, "results", "detector_probe.json")
    if not os.path.exists(p):
        return "_detector probe not run yet_"
    out = ["| Detector | Input | Channels | Stops, sign in view | Box on the sign | Reliable range (≥90%) | "
           "False stops, no sign |", "|---|---|---|---|---|---|---|"]
    for r in load(p):
        tag = " **(2024 setting)**" if (r["detector"], r["size"], r["order"]) == ("2024_retrained", 640, "bgr") else ""
        out.append(f"| {r['detector'].replace('_', ' ')}{tag} | {r['size']} | {r['order'].upper()} | "
                   f"{pct(r['stop_rate'])} | {pct(r['hit_rate'])} | {_range(r['by_distance'])} | {pct(r['false_stop'])} |")
    real = os.path.join(ROOT, "results", "detector_real2024.json")
    if os.path.exists(real):
        out += ["", "On real frames from the 2024 car, none of which contains a stop sign "
                "(`scripts/probe_real_frames.py`):", "",
                "| Detector | Channels | Frames that would have stopped the car | Median box size | Median confidence |",
                "|---|---|---|---|---|"]
        for r in load(real):
            area = f"{100 * r['median_box_area_frac']:.0f}% of the frame" if r["median_box_area_frac"] else "—"
            conf = f"{r['median_conf']:.2f}" if r["median_conf"] else "—"
            out.append(f"| {r['detector'].replace('_', ' ')} | {r['order'].upper()} | {r['stopped']}/{r['frames']} | {area} | {conf} |")
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
        out += ["| Loop | FPS | Preprocess p50 | Policy p50 | Detector p50 | Frame-arrival → command p50 / p95 | "
                "Detector frames that stopped the car | RAM (system) |", "|---|---|---|---|---|---|---|---|"]
        for fpath in files:
            d = load(fpath)
            det = f"{d['detector_ms']['p50']:.1f} ms" if d["detector_ms"]["p50"] else "—"
            prep = f"{d['prep_ms']['p50']:.1f} ms" if d.get("prep_ms") else "—"
            stops = pct(d["stop_frac_of_detector_frames"]) if d.get("detector") else "—"
            out.append(f"| {d.get('label', os.path.basename(fpath))} | {d['fps']:.1f} | {prep} | {d['policy_ms']['p50']:.1f} ms | "
                       f"{det} | {d['arrival_to_command_ms']['p50']:.1f} / {d['arrival_to_command_ms']['p95']:.1f} ms | "
                       f"{stops} | {d['ram_mb']['max']} MB |")
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


def fig_arms(arms, fname):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    grid = [(label, [arm_stats(arm, c) for c, _ in CONDITIONS]) for arm, label in arms]
    grid = [(label, per) for label, per in grid if per[0] is not None]
    if not grid:
        return
    for mode, t in THEMES.items():
        fig, axes = plt.subplots(1, len(CONDITIONS), figsize=(9.6, 0.42 * len(grid) + 1.3),
                                 facecolor=t["surface"], sharey=True)
        c = t["series"][0]
        for j, (ax, (_, cond_label)) in enumerate(zip(axes, CONDITIONS)):
            _style(ax, t)
            for i, (label, per) in enumerate(grid):
                s = per[j]
                y = len(grid) - 1 - i
                if s is None:
                    continue
                ax.plot([100 * s["lo"], 100 * s["hi"]], [y, y], color=c, linewidth=2, solid_capstyle="round")
                for sd in s["seeds"]:
                    ax.plot(100 * sd["rate"], y, "o", ms=3.5, color=t["muted"], alpha=0.8, zorder=3)
                ax.plot(100 * s["rate"], y, "o", ms=8, color=c, markeredgecolor=t["surface"], markeredgewidth=2, zorder=4)
                ax.text(100 * s["rate"], y + 0.32, f"{100 * s['rate']:.0f}%", ha="center", fontsize=8, color=t["ink"])
            ax.set_xlim(-4, 104)
            ax.set_xticks([0, 50, 100])
            ax.set_title(cond_label, color=t["ink"], fontsize=10, loc="left")
            ax.set_ylim(-0.6, len(grid) - 0.2)
        axes[0].set_yticks(range(len(grid)))
        axes[0].set_yticklabels([label for label, _ in grid][::-1], color=t["ink2"], fontsize=9)
        fig.supxlabel("corridors completed (%): big dot pooled over seeds, bar = Wilson 95%, small dots = seeds",
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
    fig_arms(MAIN, "fig_arms")
    fig_arms(HIST, "fig_history")
    fig_arms(PAIR, "fig_pair")
    fig_offline_vs_closed(main_stats + hist_stats, "fig_offline_vs_closed")
    blocks = {"baselines": table_baselines(), "arms": table_arms(MAIN), "history": table_arms(HIST),
              "pair": table_arms(PAIR), "flipcheck": table_flipcheck(), "detector": table_detector(),
              "nano": table_nano()}
    rewrite_readme(blocks)
    summary = {k: [{"arm": a, **({kk: vv for kk, vv in s.items() if kk != "seeds"} if s else {})}
                   for a, _, s in v] for k, v in (("main", main_stats), ("history", hist_stats))}
    with open(os.path.join(ROOT, "results", "summary.json"), "w") as f:
        json.dump(summary, f, indent=1, default=float)
    for k, v in blocks.items():
        print(f"== {k} ==\n{v}\n")


if __name__ == "__main__":
    main()
