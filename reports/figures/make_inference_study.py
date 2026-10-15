"""Figures for benchmarking/INFERENCE_STUDY.md, built only from result files on disk.
Usage: /nvme-data/flexpi-eval-venv/bin/python benchmarking/figures/make_inference_study.py
Out:   benchmarking/figures/inference_study/*.png|pdf
"""
import glob, json, os
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); BM = os.path.dirname(HERE)
RES = os.path.join(BM, "results", "local"); OUT = os.path.join(HERE, "inference_study"); os.makedirs(OUT, exist_ok=True)
TREE = "/nvme-data/Komal/wm/ckpts/ms_eval_ckpts/AUTO_sr_scan/tts_tree_100k/tree.json"
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


def steady(cfg):
    lines = open(os.path.join(RES, "item2", "inloop", "INLOOP.md")).read().splitlines()
    row = next(l for l in lines if l.startswith(f"| {cfg} |"))
    return float(row.split("|")[4])


# 1. SR vs 400-action time (operating points) ---------------------------------------------------------------
SCAN = "/nvme-data/Komal/wm/ckpts/ms_eval_ckpts/AUTO_sr_scan"; TT = f"{SCAN}/tts_tree_100k"
SUITES4 = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
tree = json.load(open(TREE))["nodes"]
ms_root, ms_kvc = steady("trt_deploy"), steady("trt_kvc")


def steady_dir(d):
    s_ = json.load(open(os.path.join(d, "short.json")))["timing"]; l_ = json.load(open(os.path.join(d, "long.json")))["timing"]
    return (l_["model_ms_per_decision"] * l_["n_decisions"] - s_["model_ms_per_decision"] * s_["n_decisions"]) / \
           (l_["n_decisions"] - s_["n_decisions"])


def sr_parts(node_dir):
    k = n = 0
    for s_ in SUITES4:
        for p_ in glob.glob(os.path.join(node_dir, s_, "part*.json")):
            for v in json.load(open(p_))["per_task"].values(): k += v["successes"]; n += v["total_episodes"]
    return k / n


def grounded(bk):
    k = n = 0; tms = 0.0; nd = 0
    for f in glob.glob(f"{SCAN}/4suite_grounded_4suite_20ep/{bk}/*/*_sr.json"):
        d = json.load(open(f)); t = d["timing"]
        for v in d["per_task"].values(): k += v["successes"]; n += v["total_episodes"]
        tms += t["model_ms_per_decision"] * t["n_decisions"]; nd += t["n_decisions"]
    return k / n, tms / nd


ms_sdpa = steady_dir(f"{TT}/sdpa_kvc_compile/inloop"); ms_43 = steady_dir(f"{TT}/tf443/inloop")
g_flex, g_trt = grounded("flex"), grounded("trt")
# (label, ms/decision, E, SR, status) status: accepted | rejected | pending | tts
pts = [("Root TRT deploy, E8", ms_root, 8, tree["trt_deploy-E8"]["sr_avg"], "accepted"),
       ("TRT + KV cache, E8", ms_kvc, 8, tree["trt_deploy-kvc-E8"]["sr_avg"], "accepted"),
       ("TRT + KV cache, E12", ms_kvc, 12, tree["trt_deploy-kvc-E12"]["sr_avg"], "accepted"),
       ("SDPA + compile + KV cache, E12 (best)", ms_sdpa, 12, tree["trt_deploy-kvc-E12"]["sr_avg"], "accepted"),
       ("KV cache, E10", ms_kvc, 10, tree["trt_deploy-kvc-E10"]["sr_avg"], "rejected"),
       ("TRT compiled (tf 4.43), E8", ms_43, 8, sr_parts(f"{TT}/tf443/tf443-trt_deploy-E8"), "pending"),
       ("TRT compiled (tf 4.43), E12", ms_43, 12, sr_parts(f"{TT}/tf443/tf443-trt_deploy-E12"), "pending"),
       ("Grounded MPC (flex), E8", g_flex[1], 8, g_flex[0], "tts"),
       ("Grounded MPC (TRT), E8", g_trt[1], 8, g_trt[0], "tts")]
fig, ax = plt.subplots(figsize=(6.4, 3.9)); style(ax)
floor = 95.25 - 2
ax.axhline(floor, color=MUTED, lw=1, ls=(0, (4, 3)))
ax.text(130, floor - 0.1, "SR floor 93.25", color=MUTED, fontsize=8, ha="right", va="top")
for st, col, mk, lab in (("accepted", C1, "o", "accepted (greedy)"), ("pending", C4, "D", "SR pending replicate"),
                         ("rejected", C2, "s", "rejected"), ("tts", C3, "^", "grounded MPC (imagination)")):
    sel = [p for p in pts if p[4] == st]
    ax.scatter([400 / p[2] * p[1] / 1000 for p in sel], [100 * p[3] for p in sel], s=60, color=col, marker=mk,
               edgecolor=SURF, linewidth=2, zorder=3, label=lab)
offs = {"TRT + KV cache, E12": (6, -14), "SDPA + compile + KV cache, E12 (best)": (6, 7),
        "TRT + KV cache, E8": (7, -13), "TRT compiled (tf 4.43), E12": (6, -14), "TRT compiled (tf 4.43), E8": (7, 6)}
for name, ms, e, sr, st in pts:
    dx, dy = offs.get(name, (7, 5))
    ax.annotate(name, (400 / e * ms / 1000, 100 * sr), xytext=(dx, dy), textcoords="offset points", fontsize=7,
                color=INK2, ha="right" if dx < 0 else "left")
ax.set_xscale("log"); ax.set_xticks([15, 20, 30, 50, 100]); ax.set_xticklabels(["15", "20", "30", "50", "100"])
ax.set_xlabel("400-action model time (s, log), in-loop; lower is better")
ax.set_ylabel("LIBERO 4-suite SR (%)")
ax.set_xlim(12, 140); ax.set_ylim(91.7, 95.6)
ax.legend(frameon=False, fontsize=7, loc="upper left", labelcolor=INK2)
save(fig, "fig1_sr_vs_400action")

# 2. Latency of every measured configuration ----------------------------------------------------------------
rows = []
for f in glob.glob(os.path.join(RES, "e2e_sweep", "*.json")):
    d = json.load(open(f)); p = d.get("p50_ms")
    if p: rows.append((os.path.basename(f)[:-5], p))
rows.sort(key=lambda r: r[1])
def group(c):
    if "single" in c: return "single-forward (fails SR)", C2
    if c.startswith("pd"): return "parallel decode pd2/pd3", C3
    if "kvc" in c: return "KV cache", C1
    return "kernel / compile / graphs only", C4
fig, ax = plt.subplots(figsize=(6.4, 7.2)); style(ax, "x")
base = dict(rows)["g_trt_deploy"]
for i, (c, p) in enumerate(rows):
    g, col = group(c)
    ax.barh(i, p, height=0.72, color=col, label=g)
    ax.text(p + 12, i, f"{p:.0f}", va="center", fontsize=7, color=INK2)
ax.axvline(base, color=MUTED, lw=1, ls=(0, (4, 3)))
ax.text(base + 8, -0.9, "root 544 ms", color=MUTED, fontsize=8)
ax.set_yticks(range(len(rows))); ax.set_yticklabels([c for c, _ in rows], fontsize=7)
ax.invert_yaxis(); ax.set_xlabel("p50 ms per decision (profiler, uncontended A6000)")
h, l = ax.get_legend_handles_labels(); seen = {}
for hh, ll in zip(h, l): seen.setdefault(ll, hh)
order = ["KV cache", "kernel / compile / graphs only", "parallel decode pd2/pd3", "single-forward (fails SR)"]
ax.legend([seen[o] for o in order if o in seen], [o for o in order if o in seen], frameon=False, fontsize=8,
          loc="upper right", labelcolor=INK2)
ax.set_ylim(len(rows) - 0.4, -1.6)
save(fig, "fig2_latency_all_configs")

# 3. Decision breakdown --------------------------------------------------------------------------------------
bd = {c: json.load(open(os.path.join(RES, "item2", "breakdown", f"{c}.json")))["real"] for c in ("trt_deploy", "trt_kvc")}
parts = [("Cosmos + DINO encode", ["encode_window (Cosmos+DINO)"], C4),
         ("full-sequence forward x2", ["predictor forward (full seq)"], C2),
         ("KV prefill (past frames)", ["kv prefill (past frames)"], C1),
         ("last-frame forward x2", ["kv forward_active (last frame)"], C3)]
fig, ax = plt.subplots(figsize=(6.0, 2.3)); style(ax, "x")
labels = {"trt_deploy": "Root (TRT deploy)", "trt_kvc": "TRT + KV cache"}
for j, c in enumerate(("trt_deploy", "trt_kvc")):
    st = bd[c]["stages"]; left = 0
    for name, keys, col in parts:
        v = sum(st[k]["ms_per_decision_p50"] for k in keys if k in st)
        if v <= 0: continue
        ax.barh(j, v, left=left, height=0.55, color=col, edgecolor=SURF, linewidth=2, label=name)
        if v > 45: ax.text(left + v / 2, j, f"{v:.0f}", ha="center", va="center", fontsize=8, color=INK)
        left += v
    ax.text(left + 8, j, f"{bd[c]['total_ms_p50']:.0f} ms", va="center", fontsize=8, color=INK2)
ax.set_yticks([0, 1]); ax.set_yticklabels([labels["trt_deploy"], labels["trt_kvc"]]); ax.invert_yaxis()
ax.set_xlabel("ms per decision (real LIBERO windows, p50)"); ax.set_xlim(0, 640)
h, l = ax.get_legend_handles_labels(); seen = {}
for hh, ll in zip(h, l): seen.setdefault(ll, hh)
ax.legend(seen.values(), seen.keys(), frameon=False, fontsize=7, ncol=2, loc="upper center",
          bbox_to_anchor=(0.5, -0.38), labelcolor=INK2)
save(fig, "fig3_decision_breakdown")

# 4. Batch scaling -------------------------------------------------------------------------------------------
bs = json.load(open(os.path.join(RES, "inference_study", "batch_scaling.json")))["rows"]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.0, 2.5)); style(a1); style(a2)
B = [r["B"] for r in bs]
a1.plot(B, [r["ms_per_sample"] for r in bs], color=C1, lw=2, marker="o", ms=6, mec=SURF, mew=2)
a1.set_xscale("log", base=2); a1.set_xticks(B); a1.set_xticklabels(B)
a1.set_xlabel("batch size B"); a1.set_ylabel("ms per sample"); a1.set_ylim(0, 850)
a1.set_title("per-sample decision time", fontsize=9, color=INK)
a2.plot(B, [r["peak_gb"] for r in bs], color=C1, lw=2, marker="o", ms=6, mec=SURF, mew=2)
a2.set_xscale("log", base=2); a2.set_xticks(B); a2.set_xticklabels(B)
a2.set_xlabel("batch size B"); a2.set_ylabel("peak GPU memory (GB)"); a2.set_ylim(0, 45)
a2.set_title("memory", fontsize=9, color=INK)
save(fig, "fig4_batch_scaling")

# 5. Paired spatial replicate (per task) ---------------------------------------------------------------------
fails = {}
for n in ("trt_deploy-E8", "trt_deploy-kvc-E12"):
    f = defaultdict(int)
    for line in open(os.path.join(RES, "inference_study", f"rep_{n}.txt")):
        t, _, o = line.split()
        f[int(t)] += o != "OK"
    fails[n] = [f[t] for t in range(10)]
fig, ax = plt.subplots(figsize=(5.6, 2.6)); style(ax, "y"); w = 0.38
for k, (n, col, lab) in enumerate((("trt_deploy-E8", C1, "Root, E=8"), ("trt_deploy-kvc-E12", C2, "KV cache, E=12"))):
    xs = [t + (k - 0.5) * w for t in range(10)]
    ax.bar(xs, fails[n], width=w - 0.04, color=col, label=lab)
ax.set_xticks(range(10)); ax.set_xticklabels([f"t{t}" for t in range(10)])
ax.set_xlabel("libero_spatial task"); ax.set_ylabel("failures / 20 episodes")
ax.legend(frameon=False, fontsize=8, labelcolor=INK2)
save(fig, "fig5_spatial_replicate")


# 6. Decisive tasks: reactive vs blend vs grounded vs MTP-grounded ------------------------------------------
TASKS = [("libero_spatial", 4), ("libero_spatial", 9), ("libero_goal", 6), ("libero_goal", 9),
         ("libero_10", 3), ("libero_10", 8), ("libero_10", 9)]
def task_sr(f, t):
    if not os.path.exists(f): return None
    v = json.load(open(f))["per_task"].get(f"task{t}")
    return 100 * v["successes"] / v["total_episodes"] if v else None
modes = [("reactive greedy (ref)", C1, "o", lambda s_, t: task_sr(f"{SCAN}/4suite_chunk16_warm_exec8_100k/exec8/{s_}/{s_}_sr.json", t)),
         ("blend alpha=0.5", C4, "D", lambda s_, t: task_sr(f"{SCAN}/alpha_blend_sweep/a0.5/{s_}_sr.json", t)),
         ("grounded MPC (TRT)", C3, "^", lambda s_, t: task_sr(f"{SCAN}/4suite_grounded_4suite_20ep/trt/{s_}/{s_}_t{t}_sr.json", t)),
         ("MTP-grounded (TRT)", C2, "s", lambda s_, t: task_sr(f"{SCAN}/mtp_grounded_keytasks/{s_}_t{t}_sr.json", t))]
fig, ax = plt.subplots(figsize=(6.4, 3.0)); style(ax, "y")
for j, (lab, col, mk, fn) in enumerate(modes):
    xs, ys = [], []
    for i, (s_, t) in enumerate(TASKS):
        v = fn(s_, t)
        if v is not None: xs.append(i + (j - 1.5) * 0.14); ys.append(v)
    ax.scatter(xs, ys, s=46, color=col, marker=mk, edgecolor=SURF, linewidth=1.5, zorder=3, label=lab)
ax.set_xticks(range(len(TASKS))); ax.set_xticklabels([f"{s_.replace('libero_', '')} t{t}" for s_, t in TASKS], fontsize=8)
ax.set_ylabel("task SR (%)"); ax.set_ylim(0, 105)
ax.legend(frameon=False, fontsize=7, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.16), labelcolor=INK2)
save(fig, "fig6_decisive_tasks")
