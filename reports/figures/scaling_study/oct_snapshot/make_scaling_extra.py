#!/usr/bin/env python3
"""Extra scaling figures (figures4papers house style), from /tmp/scaling_data.json (wandb pull):
  1) scaling_val_psnr      — val PSNR vs tokens (curves) + iso-token vs N
  2) scaling_isoflop       — val_video vs compute (6ND, log-log) + iso-FLOP slices (loss vs N at fixed C)
  3) scaling_depth_ratio   — val_video (iso-token) vs depth (layers) and vs width:depth ratio
All CURRENT-SNAPSHOT (mid-training, rungs at different D); iso-token budget pinned by the least-trained rung.
"""
import json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = "/media/mbzuaiser/SSD1/Komal/Documents/WM/worldlab/benchmarking/results/local"
PALETTE = {"blue_main": "#0F4D92", "violet": "#9A4D8E", "red_strong": "#B64342", "neutral": "#CFCECE", "highlight": "#FFD700"}
def style(fs=16, lw=2.2):
    plt.rcParams.update({"font.family": ["DejaVu Sans", "sans-serif"], "font.size": fs, "axes.linewidth": lw,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.8})
def save(fig, base):
    os.makedirs(OUT, exist_ok=True)
    for f in ("png", "pdf"): fig.savefig(f"{OUT}/{base}.{f}", dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig); return f"{OUT}/{base}.png"

d = json.load(open("/tmp/scaling_data.json"))
rungs = sorted(d, key=lambda k: d[k]["N"])
N = {k: d[k]["N"] for k in rungs}
cmap = plt.cm.viridis
col = {k: cmap(x) for k, x in zip(rungs, np.linspace(0.05, 0.9, len(rungs)))}
def arr(k, key): return np.array([v for v in d[k][key]], dtype=float)
def clean(k, key):  # (tokens, metric) with non-None metric
    t = arr(k, "tokens"); m = np.array([np.nan if v is None else v for v in d[k][key]], float)
    ok = ~np.isnan(m); return t[ok], m[ok]
lab = lambda k: f"{k}·{N[k]/1e9:.2f}B" if N[k] >= 1e9 else f"{k}·{N[k]/1e6:.0f}M"
common = min(arr(k, "tokens").max() for k in rungs)   # iso-token budget
def iso(k, key):
    t, m = clean(k, key)
    sl = t <= common + 1
    return m[sl][-1] if sl.any() else m[0]

style()
# ---------- 1) val_psnr ----------
fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
for k in rungs:
    t, p = clean(k, "val_psnr"); ax[0].plot(t/1e9, p, "-o", color=col[k], lw=2.2, ms=4, label=lab(k))
ax[0].set_xscale("log"); ax[0].set_xlabel("tokens seen (B)"); ax[0].set_ylabel("val PSNR (dB)")
ax[0].set_title("(A) Reconstruction PSNR by model size", fontsize=15); ax[0].legend(fontsize=10, title="rung·N", title_fontsize=10)
xs = np.array([N[k] for k in rungs])/1e9; ys = np.array([iso(k, "val_psnr") for k in rungs])
ax[1].plot(xs, ys, "-", color=PALETTE["neutral"], lw=1.6, zorder=1)
for k in rungs:
    ax[1].scatter([N[k]/1e9], [iso(k, "val_psnr")], s=120, color=col[k], edgecolor="white", lw=1.2, zorder=3)
    ax[1].annotate(k, (N[k]/1e9, iso(k, "val_psnr")), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=11)
ax[1].set_xscale("log"); ax[1].set_xlabel("non-embedding params N (B)"); ax[1].set_ylabel(f"val PSNR @ {common/1e9:.2f}B tok (dB)")
ax[1].set_title("(B) PSNR vs model size (iso-token)", fontsize=15)
fig.suptitle("worldlab WM scaling — val PSNR (current snapshot)", fontsize=13, y=1.005)
print("saved", save(fig, "scaling_val_psnr"))

# ---------- 2) IsoFLOP / compute ----------
fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
# (A) loss vs compute envelope
for k in rungs:
    t, v = clean(k, "val_video"); flops = 6*N[k]*t
    ax[0].plot(flops, v, "-o", color=col[k], lw=2.2, ms=4, label=lab(k))
ax[0].set_xscale("log"); ax[0].set_xlabel("compute  C = 6·N·D  (FLOPs)"); ax[0].set_ylabel("val video CE (nats/token)")
ax[0].set_title("(A) Loss vs compute", fontsize=15); ax[0].legend(fontsize=10, title="rung·N", title_fontsize=10)
# (B) iso-FLOP slices: for feasible C, val_video vs N
# feasible C range = [max over rungs of min-FLOPs, min over rungs of max-FLOPs]
fmins, fmaxs = [], []
for k in rungs:
    t, v = clean(k, "val_video"); fl = 6*N[k]*t; fmins.append(fl.min()); fmaxs.append(fl.max())
lo, hi = max(fmins), min(fmaxs)
Cs = np.array([0.5 * (lo + hi)]) if hi > lo else np.array([hi])   # only a thin common-C band feasible this early -> one slice
ccolors = [PALETTE["blue_main"], PALETTE["red_strong"], PALETTE["violet"]]
for C, cc in zip(Cs, ccolors):
    xs2, ys2 = [], []
    for k in rungs:
        t, v = clean(k, "val_video"); fl = 6*N[k]*t
        if fl.min() <= C <= fl.max():
            o = np.argsort(fl); ys2.append(float(np.interp(C, fl[o], v[o]))); xs2.append(N[k]/1e9)
    if len(xs2) >= 2:
        ax[1].plot(xs2, ys2, "-o", color=cc, lw=2.4, ms=8, label=f"C={C:.0e} FLOPs")
        imin = int(np.argmin(ys2)); ax[1].scatter([xs2[imin]], [ys2[imin]], s=180, facecolor="none", edgecolor=PALETTE["highlight"], lw=2.5, zorder=5)
ax[1].set_xscale("log"); ax[1].set_xlabel("non-embedding params N (B)"); ax[1].set_ylabel("val video CE (nats/token)")
ax[1].set_title("(B) IsoFLOP slices (min = compute-optimal N)", fontsize=15); ax[1].legend(fontsize=11)
fig.suptitle(f"worldlab WM scaling — IsoFLOP  (feasible C≈[{lo:.1e},{hi:.1e}] at this snapshot)", fontsize=12, y=1.005)
print("saved", save(fig, "scaling_isoflop"))

# ---------- 3) depth & ratio ----------
fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
depth = {k: d[k]["layers"] for k in rungs}; ratio = {k: d[k]["hidden"]/d[k]["layers"] for k in rungs}
vv = {k: iso(k, "val_video") for k in rungs}
for ax_i, key, xlabel, title in [(ax[0], depth, "depth (num layers)", "(A) val loss vs depth"),
                                  (ax[1], ratio, "width : depth ratio (hidden/layers)", "(B) val loss vs width:depth ratio")]:
    xs3 = [key[k] for k in rungs]; ys3 = [vv[k] for k in rungs]
    o = np.argsort(xs3); ax_i.plot(np.array(xs3)[o], np.array(ys3)[o], "-", color=PALETTE["neutral"], lw=1.6, zorder=1)
    for k in rungs:
        ax_i.scatter([key[k]], [vv[k]], s=120, color=col[k], edgecolor="white", lw=1.2, zorder=3)
        ax_i.annotate(k, (key[k], vv[k]), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=11)
    ax_i.set_xlabel(xlabel); ax_i.set_ylabel(f"val video CE @ {common/1e9:.2f}B tok"); ax_i.set_title(title, fontsize=15)
fig.suptitle("worldlab WM scaling — shape (depth & width:depth ratio), iso-token  [NOTE: co-varies with N in this ladder]", fontsize=11.5, y=1.005)
print("saved", save(fig, "scaling_depth_ratio"))
print("iso-token budget:", f"{common/1e9:.2f}B | depth", depth, "| ratio", {k: round(ratio[k],1) for k in rungs})
