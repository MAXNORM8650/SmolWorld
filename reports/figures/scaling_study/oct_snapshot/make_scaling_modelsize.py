#!/usr/bin/env python3
"""Model-size scaling figure for the worldlab WM ladder (N4–N9), figures4papers house style.
Panel A: per-stream val_video CE vs tokens-seen, one curve per model size (scaling trend).
Panel B: val_video CE vs model size N at a common iso-token budget (fair capacity comparison).
Reads /tmp/scaling_data.json (pulled from wandb). Outputs png+pdf @300 dpi.
NOTE: runs are mid-training at different D -> this is a CURRENT-SNAPSHOT, not a converged law.
"""
import json, math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PALETTE = {"blue_main": "#0F4D92", "violet": "#9A4D8E", "red_strong": "#B64342", "neutral": "#CFCECE"}

def apply_publication_style(font_size=16, axes_linewidth=2.2):
    plt.rcParams.update({
        "font.family": ["DejaVu Sans", "Helvetica", "Arial", "sans-serif"],
        "font.size": font_size, "axes.linewidth": axes_linewidth,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42,
        "xtick.direction": "out", "ytick.direction": "out",
        "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.8,
    })

def finalize(fig, base, formats=("png", "pdf"), dpi=300):
    import os; os.makedirs(os.path.dirname(base), exist_ok=True)
    for f in formats:
        fig.savefig(f"{base}.{f}", dpi=dpi, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return [f"{base}.{f}" for f in formats]

d = json.load(open("/tmp/scaling_data.json"))
rungs = sorted(d, key=lambda k: d[k]["N"])                      # N4..N9 by size
Ns = np.array([d[k]["N"] for k in rungs], float)
# size-ordered color ramp (small=light blue -> big=violet)
cmap = plt.cm.viridis
colors = [cmap(x) for x in np.linspace(0.05, 0.9, len(rungs))]

apply_publication_style()
fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

# --- Panel A: val_video vs tokens, one curve per size ---
axA = axes[0]
for k, c in zip(rungs, colors):
    t = np.array(d[k]["tokens"], float) / 1e9
    vv = np.array(d[k]["val_video"], float)
    axA.plot(t, vv, "-o", color=c, lw=2.2, ms=4,
             label=f"{k} · {d[k]['N']/1e9:.2f}B" if d[k]['N'] >= 1e9 else f"{k} · {d[k]['N']/1e6:.0f}M")
axA.set_xscale("log")
axA.set_xlabel("tokens seen (B)")
axA.set_ylabel("val video CE (nats/token)")
axA.set_title("(A) Training trajectories by model size", fontsize=15)
axA.legend(title="rung · params", fontsize=11, title_fontsize=11, ncol=1, loc="upper right")

# --- Panel B: iso-token slice: val_video @ common budget vs N ---
axB = axes[1]
common = min(max(d[k]["tokens"]) for k in rungs)
iso = []
for k in rungs:
    t = np.array(d[k]["tokens"], float); vv = np.array(d[k]["val_video"], float)
    idx = int(np.max(np.where(t <= common + 1)[0])) if np.any(t <= common + 1) else 0
    iso.append(vv[idx])
iso = np.array(iso)
axB.plot(Ns/1e9, iso, "-", color=PALETTE["neutral"], lw=1.6, zorder=1)
for x, y, c, k in zip(Ns/1e9, iso, colors, rungs):
    axB.scatter([x], [y], s=120, color=c, zorder=3, edgecolor="white", linewidth=1.2)
    axB.annotate(k, (x, y), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=11)
axB.set_xscale("log")
axB.set_xlabel("non-embedding params N (B)")
axB.set_ylabel(f"val video CE @ {common/1e9:.2f} B tokens")
axB.set_title("(B) Impact of model size (iso-token)", fontsize=15)

fig.suptitle("worldlab WM scaling — Cosmos-DV, 4-suite LIBERO-depth  (current snapshot, mid-training)",
             fontsize=13, y=1.005)
out = finalize(fig, "/media/mbzuaiser/SSD1/Komal/Documents/WM/worldlab/benchmarking/results/local/scaling_model_size")
print("saved:", out, "| iso-token budget:", f"{common/1e9:.2f}B")
print("iso-token val_video by rung:", {k: round(float(v), 3) for k, v in zip(rungs, iso)})
