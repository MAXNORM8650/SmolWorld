"""Figures for benchmarking/VQ_JEPA_TOKENIZER.md, built only from result files on disk.
Usage: /nvme-data/flexpi-eval-venv/bin/python benchmarking/figures/make_vq_jepa.py
Out:   benchmarking/figures/vq_jepa/*.png|pdf (+ copies of the qualitative panels written by the runs)
"""
import json, os, shutil
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "vq_jepa"); os.makedirs(OUT, exist_ok=True)
R = "/nvme-data/Komal/wm/vq_jepa"
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF})


def style(ax, grid_axis="both"):
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.grid(True, axis=grid_axis, color=GRID, lw=0.8); ax.set_axisbelow(True)


def save(fig, name):
    fig.tight_layout()
    for ext in ("png", "pdf"): fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=200)
    plt.close(fig); print("wrote", name)


def log(run):
    return [json.loads(l) for l in open(f"{R}/{run}/log.jsonl")]


# 1. tokenizer comparison (held-out LIBERO, same 60 videos / 1666 frames) -------------------------------------------
pca = json.load(open(f"{R}/pca_vq_on_cache_val.json"))
rows = [("PCA-VQ k=3  (256 tok x 2 codes, 64+64)", pca["pca_vq_k3"]),
        ("PCA-VQ k=8  (256 tok x 2 codes, 64+64)", pca["pca_vq_k8"]),
        ("FSQ 8x8  (64 tok x 1 code, 1000)", log("fsq/g8")[-1]),
        ("VQ-BeT 8x8  (64 tok x 2 codes, 256+256)", log("rvq_v2/g8")[-1]),
        ("FSQ 16x16  (256 tok x 1 code, 1000)", log("fsq/g16")[-1]),
        ("VQ-BeT 16x16  (256 tok x 2 codes, 256+256)", log("rvq_v2/g16")[-1])]
fig, ax = plt.subplots(figsize=(7.2, 3.6))
y = range(len(rows)); h = 0.36
for k, (key, col, lab) in enumerate((("r2_all", C1, "all patches"), ("r2_fg", C2, "foreground patches"))):
    vals = [100 * r[key] for _, r in rows]
    yy = [i + (k - 0.5) * (h + 0.04) for i in y]
    ax.barh(yy, vals, height=h, color=col, label=lab)
    for v, yi in zip(vals, yy):
        ax.text(v + 0.8, yi, f"{v:.1f}", va="center", fontsize=7.5, color=INK)
ax.set_yticks(list(y)); ax.set_yticklabels([n for n, _ in rows]); ax.invert_yaxis()
ax.set_xlim(0, 100); ax.set_xlabel("V-JEPA feature variance kept by the tokens (%)")
ax.legend(frameon=False, loc="upper right"); style(ax, "x")
save(fig, "fig1_tokenizer_comparison")

# 2. training curves (foreground variance kept) ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.4, 3.4))
for run, col, lab in (("rvq_v2/g16", C1, "VQ-BeT 16x16"), ("fsq/g16", C2, "FSQ 16x16"),
                      ("rvq_v2/g8", C3, "VQ-BeT 8x8"), ("fsq/g8", C4, "FSQ 8x8")):
    L = [d for d in log(run) if d["step"] > 0]
    xs, ys = [d["step"] / 1000 for d in L], [100 * d["r2_fg"] for d in L]
    ax.plot(xs, ys, color=col, lw=2, marker="o", ms=3.5, label=lab)
    ax.text(xs[-1] + 0.3, ys[-1], f"{lab} {ys[-1]:.1f}", va="center", fontsize=7.5, color=INK)
ax.set_xlim(0, 26); ax.set_xlabel("training step (k)"); ax.set_ylabel("foreground variance kept (%)")
ax.legend(frameon=False, loc="lower right"); style(ax)
save(fig, "fig2_training_curves")

# 3. VQ-BeT level-1 codebook: restart bug vs fix ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.4, 3.0))
for run, col, lab in (("rvq/g8", C2, "before fix (restart shrinks codes)"), ("rvq_v2/g8", C1, "after fix")):
    L = [d for d in log(run) if d["step"] > 0]
    ax.plot([d["step"] / 1000 for d in L], [d["perplexity"][0] for d in L], color=col, lw=2, marker="o", ms=3.5, label=lab)
ax.axhline(256, color=MUTED, lw=1, ls="--"); ax.text(11, 249, "codebook size 256", fontsize=7.5, color=INK2, va="top")
ax.set_ylim(0, 270); ax.set_xlabel("training step (k)"); ax.set_ylabel("level-1 code perplexity (val)")
ax.set_title("VQ-BeT 8x8, level-1 (primary) codebook", fontsize=9, color=INK, loc="left")
ax.legend(frameon=False, loc="center right"); style(ax)
save(fig, "fig3_codebook_fix")

# qualitative panels written by the runs -------------------------------------------------------------------------------
for src, dst in ((f"{R}/rvq_v2/g16/viz.png", "fig4_heldout_frames_vqbet16.png"),
                 (f"{R}/rvq_v2/g16/libero_spatial_ep414_frames.png", "fig5_robot_episode_ep414.png"),
                 (f"{R}/rvq_v2/g16/fucking_dog_pca_frames.png", "fig6_out_of_domain_video.png"),
                 (f"{R}/strict/viz_k8.png", "figA_pca_vq_k8.png"),
                 (f"{R}/strict/pc1_check.png", "figB_first_component.png"),
                 (f"{R}/strict/bg_probe_viz.png", "figC_motion_background_probe.png")):
    shutil.copy(src, os.path.join(OUT, dst)); print("copied", dst)
