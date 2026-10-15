#!/usr/bin/env python3
"""Chinchilla IsoFLOP curve for the worldlab dcp 4-suite ladder: val loss vs compute C=6*N*D.
Reads /tmp/scaling_raw.txt (lines: 'VAL <rung> <step> <val_video>' and 'TOK <rung> <step> <Btok>').
Panel A: per-size val-loss trajectories vs C (log-x), size-ramped colors.
Panel B: compute-optimal frontier = lower envelope of all trajectories (the Chinchilla loss(C) curve).
NOTE: current snapshot -- rungs at different D; frontier is partial (big rungs still early)."""
import re, os, math
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = "/media/mbzuaiser/SSD1/Komal/Documents/WM/worldlab/benchmarking/results/local"
PALETTE = {"blue_main": "#0F4D92", "violet": "#9A4D8E", "red_strong": "#B64342", "neutral": "#CFCECE", "highlight": "#FFD700"}
# ladder: rung -> (hidden, layers)
LAD = {"S1":(128,6),"S2":(192,8),"N1":(256,8),"S4":(320,10),"S5":(384,12),"N2":(512,10),"N3":(768,12),
       "N4":(1024,16),"N5":(1536,24),"N6":(2048,24),"N7":(2560,30),"N8":(3072,32),"N9":(4096,30)}
import os as _os
KEEP = _os.environ.get("RUNGS", "").split(",") if _os.environ.get("RUNGS") else list(LAD.keys())
LAD = {r:v for r,v in LAD.items() if r in KEEP}
ORDER = list(LAD.keys())
Nparam = {r:12*h*h*L for r,(h,L) in LAD.items()}   # non-embedding params
TAG = _os.environ.get("TAG", "")

def style(fs=15, lw=2.0):
    plt.rcParams.update({"font.family":["DejaVu Sans","sans-serif"],"font.size":fs,"axes.linewidth":lw,
        "axes.spines.top":False,"axes.spines.right":False,"legend.frameon":False,
        "pdf.fonttype":42,"ps.fonttype":42,"axes.grid":True,"grid.alpha":0.25,"grid.linewidth":0.8})

# ---- parse ----
val = {r:{} for r in LAD}     # rung -> {step: val_video}
tok = {r:{} for r in LAD}     # rung -> {step: tokens(absolute)}
for ln in open("/tmp/scaling_raw.txt"):
    p = ln.split()
    if len(p)!=4: continue
    kind,r,step,v = p[0],p[1],int(p[2]),float(p[3])
    if r not in LAD: continue
    if kind=="VAL": val[r][step]=v
    elif kind=="TOK": tok[r][step]=v*1e9     # Btok -> tokens

# tokens/step is constant across rungs (same data, same global batch 16). Derive from the richest TOK signal.
tps = []
for r in LAD:
    for s,t in tok[r].items():
        if s>0 and t>0: tps.append(t/s)
TOK_PER_STEP = float(np.median(tps)) if tps else 240000.0
print(f"[chinchilla] tokens/step ~ {TOK_PER_STEP:,.0f}  (from {len(tps)} obs)")

# ---- build trajectories: (C, loss) per rung, sorted by C ----
traj = {}
for r in ORDER:
    if not val[r]: continue
    steps = sorted(val[r])
    D = np.array([s*TOK_PER_STEP for s in steps])        # tokens seen
    L = np.array([val[r][s] for s in steps])
    C = 6.0*Nparam[r]*D
    if _os.environ.get("CAP20N"):                     # Chinchilla 20N study: keep only points up to D <= 20*N (+2% tol)
        m = D <= 20.0*Nparam[r]*1.02
        C, L, D = C[m], L[m], D[m]
    if len(D) == 0:
        continue
    traj[r] = (C, L, D)

rungs = [r for r in ORDER if r in traj]
cmap = plt.cm.viridis
col = {r:cmap(x) for r,x in zip(rungs, np.linspace(0.05,0.9,len(rungs)))}
lab = lambda r: f"{r}·{Nparam[r]/1e9:.2f}B" if Nparam[r]>=1e9 else f"{r}·{Nparam[r]/1e6:.0f}M"

style()
fig, ax = plt.subplots(1,2, figsize=(13,5.2))

# Panel A: per-size loss vs compute
axA = ax[0]
for r in rungs:
    C,L,_ = traj[r]
    axA.plot(C, L, "-o", color=col[r], lw=2.0, ms=3.5, label=lab(r))
axA.set_xscale("log")
axA.set_xlabel("compute  C = 6·N·D  (FLOPs)"); axA.set_ylabel("val video CE (nats/token)")
axA.set_title("(A) Loss vs compute, per model size", fontsize=14)
axA.legend(fontsize=8.5, ncol=2, title="rung·N", title_fontsize=9)

# Panel B: compute-optimal frontier (lower envelope across all rungs)
axB = ax[1]
allC = np.concatenate([traj[r][0] for r in rungs]); allL = np.concatenate([traj[r][1] for r in rungs])
o = np.argsort(allC); allC, allL = allC[o], allL[o]
# running minimum loss as compute grows = compute-optimal envelope
env = np.minimum.accumulate(allL)
axB.scatter(allC, allL, s=10, color=PALETTE["neutral"], alpha=0.5, zorder=1, label="all (size,D) points")
axB.plot(allC, env, "-", color=PALETTE["red_strong"], lw=2.6, zorder=3, label="compute-optimal frontier")
# fit L = E + a*C^-b on the envelope (log-log of (L-E))
try:
    E = max(0.0, allL.min()-0.5)
    m = env - E > 1e-3
    b,loga = np.polyfit(np.log(allC[m]), np.log(env[m]-E), 1)
    axB.plot(allC, E+np.exp(loga)*allC**b, "--", color=PALETTE["blue_main"], lw=1.8,
             label=f"fit  L≈{E:.2f}+{np.exp(loga):.1e}·C^{b:.3f}")
except Exception as e:
    print("fit skipped:", e)
axB.set_xscale("log"); axB.set_xlabel("compute  C = 6·N·D  (FLOPs)"); axB.set_ylabel("val video CE (nats/token)")
axB.set_title("(B) Chinchilla compute-optimal frontier", fontsize=14)
axB.legend(fontsize=9.5)

fig.suptitle("worldlab WM — Chinchilla IsoFLOP (dcp 4-suite LIBERO-depth, current snapshot)", fontsize=12.5, y=1.004)
os.makedirs(OUT, exist_ok=True)
base = f"chinchilla_flops{('_'+TAG) if TAG else ''}"
for ext in ("png","pdf"): fig.savefig(f"{OUT}/{base}.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.05)
print("saved", f"{OUT}/{base}.png")
print("rungs:", {r:(f"{Nparam[r]/1e6:.0f}M", len(val[r]), f"Cmax={traj[r][0].max():.1e}") for r in rungs})
