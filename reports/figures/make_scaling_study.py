"""Figures + numbers for benchmarking/SCALING_STUDY.md, built only from the wandb snapshot on disk.
Usage: /nvme-data/flexpi-eval-venv/bin/python benchmarking/figures/make_scaling_study.py
In:    benchmarking/results/scaling_study/wandb_snapshot_2026-10-15.json   (all 'wl_scale*' runs, val history)
Out:   benchmarking/figures/scaling_study/*.png|pdf, benchmarking/results/scaling_study/summary.json
Stitching: a resumed run restarts from its last checkpoint, so for every step the value from the most recently
created fragment wins. Each point keeps the LR schedule (cfg.steps) of the fragment that logged it.
"""
import json, math, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, "scaling_study"); os.makedirs(OUT, exist_ok=True)
RES = os.path.join(ROOT, "results", "scaling_study")
SNAP = json.load(open(os.path.join(RES, "wandb_snapshot_2026-10-15.json")))["runs"]

INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF})
TOK_PER_STEP = 148576                 # logged tokens_seen / step, identical for every run (global batch 16 x 9286 tok)
SEQ = TOK_PER_STEP // 16


def style(ax, grid_axis="both"):
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.grid(True, axis=grid_axis, color=GRID, lw=0.8); ax.set_axisbelow(True)


def save(fig, name):
    fig.tight_layout()
    for ext in ("png", "pdf"): fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=200)
    plt.close(fig); print("wrote", name)


def lr_at(step, peak, warm, total):
    if step < warm: return peak * (step + 1) / max(1, warm)
    return 0.5 * peak * (1 + math.cos(math.pi * min(1.0, (step - warm) / max(1, total - warm))))


# ---- stitch fragments into one curve per rung --------------------------------------------------------------------
def rung_of(name):
    tag = name.replace("wl_scale1g_", "1g:").replace("wl_scale_", "").replace("_dcp_4suite_amd", "")
    return tag


rungs = {}
for rid, r in sorted(SNAP.items(), key=lambda kv: kv[1]["created"]):
    if r["name"].startswith("wl_scale1g_") and r["steps"] < 10000 or not r["val"]:
        continue                                          # short "20N" runs are separate experiments (own table)
    k = rung_of(r["name"])
    R = rungs.setdefault(k, dict(hidden=r["hidden"], layers=r["layers"], lr=r["lr"], pts={}, frags=[]))
    R["frags"].append(dict(id=rid, created=r["created"], steps=r["steps"], warmup=r["warmup"], state=r["state"],
                           n_val=len(r["val"]), last=max([v["_step"] for v in r["val"]], default=None)))
    for v in r["val"]:                                    # later fragment overwrites
        R["pts"][v["_step"]] = dict(L=v["val_video"], psnr=v.get("val_psnr"), sched=r["steps"], warm=r["warmup"], frag=rid)
for k, R in rungs.items():
    R["N"] = float(12 * R["hidden"] ** 2 * R["layers"])   # non-embedding params (launcher convention)
    R["steps"] = np.array(sorted(R["pts"]))
    R["L"] = np.array([R["pts"][s]["L"] for s in R["steps"]])
    R["D"] = R["steps"].astype(float) * TOK_PER_STEP          # float: 6*N*D overflows int64
    R["lr_frac"] = np.array([lr_at(s, 1.0, R["pts"][s]["warm"], R["pts"][s]["sched"]) for s in R["steps"]])

MAIN = ["N4", "N5", "N6", "N7", "N8", "N9"]
SMALL = [k for k in ("1g:S1", "1g:S2", "1g:N1", "1g:S4", "1g:S5", "1g:N2", "1g:N3") if k in rungs]
CMAP = plt.cm.viridis
ORDER = sorted(SMALL + MAIN, key=lambda k: 12 * rungs[k]["hidden"] ** 2 * rungs[k]["layers"])
col = {k: CMAP(x) for k, x in zip(ORDER, np.linspace(0.0, 0.92, len(ORDER)))}
fmtN = lambda n: f"{n/1e9:.2f}B" if n >= 1e9 else f"{n/1e6:.0f}M" if n >= 1e7 else f"{n/1e6:.1f}M"

# ---- the "20N" short runs: what budget they really got ----------------------------------------------------------
short = []
for rid, r in SNAP.items():
    if r["name"].startswith("wl_scale1g_") and r["steps"] < 10000 and r["val"]:
        N = 12 * r["hidden"] ** 2 * r["layers"]
        short.append(dict(rung=rung_of(r["name"]), id=rid, steps=r["steps"], warmup=r["warmup"],
                          planned_tok_per_step=round(20 * N / r["steps"]), D_over_N=r["steps"] * TOK_PER_STEP / N,
                          last_step=max(v["_step"] for v in r["val"]), L_last=r["val"][-1]["val_video"]))

# ---- validation noise: spread of the last 5 evals of the annealed N4/N5/N6 runs (LR < 2% of peak) ---------------
tail = {k: rungs[k]["L"][(rungs[k]["steps"] >= 34000) & (rungs[k]["steps"] <= 36000)] for k in ("N4", "N5", "N6")}
noise_sd = float(np.mean([t.std(ddof=1) for t in tail.values()]))
end_mean = {k: (float(t.mean()), float(t.std(ddof=1)), len(t)) for k, t in tail.items()}


# ---- power law at a fixed token budget: L = E + A * N^-alpha ------------------------------------------------------
def fit_pl(N, L):
    best = None
    for a in np.linspace(0.02, 1.5, 593):
        X = np.stack([np.ones_like(N), N ** -a], 1)
        (E, A), *_ = np.linalg.lstsq(X, L, rcond=None)
        if A <= 0: continue
        sse = float(((X @ [E, A] - L) ** 2).sum())
        if best is None or sse < best[0]: best = (sse, a, E, A)
    return dict(alpha=best[1], E=best[2], A=best[3], rmse=math.sqrt(best[0] / len(N)))


def slice_at(step, keys):
    xs, ys, ks = [], [], []
    for k in keys:
        R = rungs.get(k)
        if R is None: continue
        i = np.where(R["steps"] == step)[0]
        if len(i): xs.append(R["N"]); ys.append(float(R["L"][i[0]])); ks.append(k)
    return np.array(xs, float), np.array(ys), ks


S10 = 10000
N10, L10, K10 = slice_at(S10, SMALL + MAIN)
fit10 = fit_pl(N10, L10)
rng = np.random.default_rng(0); boots = []
for _ in range(2000):
    idx = rng.integers(0, len(N10), len(N10))
    if len(set(idx)) < 4: continue
    try: boots.append(fit_pl(N10[idx], L10[idx])["alpha"])
    except TypeError: pass
fit10["alpha_ci90"] = [float(np.percentile(boots, 5)), float(np.percentile(boots, 95))]
small_mask = np.array([k.startswith("1g:") for k in K10])
fit10_small = fit_pl(N10[small_mask], L10[small_mask])          # fit on <=31M, predict the 0.2-2.4B rungs
pred_big = {k: fit10_small["E"] + fit10_small["A"] * n ** -fit10_small["alpha"]
            for k, n, s in zip(K10, N10, small_mask) if not s}

# ---- compute: 6ND (non-embedding, launcher convention) and an attention-aware estimate ----------------------------
def compute(R, D):
    c6 = 6 * R["N"] * D
    attn = 6 * R["layers"] * SEQ * R["hidden"] * D          # Kaplan 2020: + 2*n_layer*n_ctx*d per token fwd, x3 fwd+bwd
    return c6, c6 + attn


summary = dict(
    tok_per_step=TOK_PER_STEP, seq_len=SEQ, val_noise_sd=noise_sd, end_of_schedule_mean=end_mean,
    short_runs=sorted(short, key=lambda x: x["D_over_N"]),
    slice_10k=dict(tokens=S10 * TOK_PER_STEP, points={k: float(l) for k, l in zip(K10, L10)}, fit_all=fit10,
                   fit_small_only=fit10_small, pred_big_from_small={k: float(v) for k, v in pred_big.items()}),
    slices={str(s): {k: float(l) for k, l in zip(*slice_at(s, SMALL + MAIN)[1:][::-1])} for s in (3500, 5000, 10000, 20000, 36000)},
    rungs={k: dict(hidden=R["hidden"], layers=R["layers"], N=R["N"], lr=R["lr"], last_step=int(R["steps"][-1]),
                   tokens=int(R["D"][-1]), L_last=float(R["L"][-1]), L_min=float(R["L"].min()),
                   C_6ND=float(compute(R, R["D"][-1])[0]), C_attn=float(compute(R, R["D"][-1])[1]),
                   attn_share=float(1 - compute(R, 1)[0] / compute(R, 1)[1]),
                   frags=R["frags"]) for k, R in rungs.items()})
json.dump(summary, open(os.path.join(RES, "summary.json"), "w"), indent=1)
print(json.dumps({k: summary[k] for k in ("val_noise_sd", "end_of_schedule_mean", "slice_10k", "slices")}, indent=1))

# ---- fig1: training curves of the main ladder --------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.2, 3.8))
for k in MAIN:
    R = rungs[k]; s = R["steps"] / 1000
    ax.plot(s, R["L"], color=col[k], lw=1.6, label=f"{k}  {fmtN(R['N'])}")
    ax.text(s[-1] + 0.6, R["L"][-1], k, color=INK, fontsize=7.5, va="center")
for k, st in (("N5", 36500), ("N7", 17000)):                # LR schedule was lengthened on resume -> LR jumps up
    R = rungs[k]; i = int(np.where(R["steps"] == st)[0][0])
    ax.annotate("schedule\nlengthened", (st / 1000, R["L"][i]), xytext=(st / 1000 + 2.5, R["L"][i] + 1.0),
                fontsize=7, color=INK2, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
ax.set_ylim(3.0, 8.0); ax.set_xlim(0, 68)
ax.set_xlabel("training step (k)   [1k steps = 0.149B tokens]"); ax.set_ylabel("val video CE (nats/token)")
ax.legend(frameon=False, ncol=2, fontsize=7.5, loc="upper right"); style(ax)
save(fig, "fig1_training_curves")

# ---- fig2: iso-token slices --------------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.2, 3.9))
SL = ((3500, "#c9c8c3", "0.52B tok (3.5k steps)"), (10000, "#2a78d6", "1.49B tok (10k)"),
      (20000, "#1baf7a", "2.97B tok (20k)"), (36000, "#eb6834", "5.35B tok (36k, annealed)"))
for st, c, lab in SL:
    x, y, ks = slice_at(st, SMALL + MAIN)
    o = np.argsort(x); ax.plot(x[o], y[o], "-o", color=c, lw=1.6, ms=5, label=lab, markeredgecolor=SURF, markeredgewidth=1)
    if st == 20000 and "N7" in ks:
        j = ks.index("N7"); ax.annotate("N7: LR re-raised", (x[j], y[j]), xytext=(6, 6), textcoords="offset points", fontsize=7, color=INK2)
xx = np.logspace(np.log10(N10.min()), np.log10(N10.max()), 100)
ax.plot(xx, fit10["E"] + fit10["A"] * xx ** -fit10["alpha"], "--", color="#2a78d6", lw=1,
        label=f"fit @1.49B: E={fit10['E']:.2f}, α={fit10['alpha']:.2f}")
ax.set_xscale("log"); ax.set_xlabel("non-embedding parameters N"); ax.set_ylabel("val video CE (nats/token)")
ax.legend(frameon=False, fontsize=7.5, loc="lower left"); style(ax)
save(fig, "fig2_iso_token_slices")

# ---- fig3: loss vs compute + empirical frontier ------------------------------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(9.6, 3.6), sharey=True)
for ax, which, title in ((axs[0], 0, "C = 6·N·D (non-embedding)"), (axs[1], 1, "C = 6·N·D + attention (seq 9286)")):
    allC, allL = [], []
    for k in SMALL + MAIN:
        R = rungs.get(k)
        if R is None or len(R["steps"]) == 0: continue
        m = R["steps"] >= 1000                               # drop warm-up points
        if not m.any(): continue
        C = compute(R, R["D"][m])[which]
        ax.plot(C, R["L"][m], "-", color=col.get(k, MUTED), lw=1.2, marker="o" if k.startswith("1g:") else None, ms=3)
        allC += list(C); allL += list(R["L"][m])
    allK = [k for k in SMALL + MAIN if k in rungs for _ in range(int((rungs[k]["steps"] >= 1000).sum()))]
    o = np.argsort(allC); env = np.minimum.accumulate(np.array(allL)[o])
    owner, prev = [], None                                   # which rung sets the envelope, in compute order
    for i, j in enumerate(o):
        if allL[j] <= env[i] and allK[j] != prev: owner.append((allK[j], float(allC[j]))); prev = allK[j]
    summary.setdefault("envelope_owner", {})[["6ND", "6ND+attn"][which]] = owner
    ax.plot(np.array(allC)[o], env, color=INK, lw=1.8, label="lower envelope")
    ax.set_xscale("log"); ax.set_xlabel(f"training compute (FLOPs)\n{title}"); style(ax)
axs[0].set_ylabel("val video CE (nats/token)"); axs[0].set_ylim(3.0, 7.0)
handles = [plt.Line2D([], [], color=col[k], lw=1.6, label=k.replace("1g:", "")) for k in ORDER]
axs[1].legend(handles=handles + [plt.Line2D([], [], color=INK, lw=1.8, label="envelope")], frameon=False, fontsize=7, ncol=2, loc="upper right")
save(fig, "fig3_loss_vs_compute")
json.dump(summary, open(os.path.join(RES, "summary.json"), "w"), indent=1)

# ---- fig4: end-of-schedule (annealed) N4/N5/N6 with noise bars ----------------------------------------------------
fig, ax = plt.subplots(figsize=(4.8, 3.2))
for k in ("N4", "N5", "N6"):
    m, sd, n = end_mean[k]; x = rungs[k]["N"]
    ax.errorbar([x], [m], yerr=[sd], fmt="o", color=col[k], ms=7, capsize=4, lw=1.4)
    ax.text(x * 1.08, m + 0.03, f"{k}  {m:.3f}", fontsize=8, color=INK)
ax.set_xscale("log"); ax.set_xlim(1.4e8, 2.2e9)
ax.set_xlabel("non-embedding parameters N"); ax.set_ylabel("val video CE, mean of last 5 evals")
ax.set_title("Same 5.35B tokens, full cosine schedule", fontsize=9, color=INK, loc="left"); style(ax)
save(fig, "fig4_annealed_endpoints")
