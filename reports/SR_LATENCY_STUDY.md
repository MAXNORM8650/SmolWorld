# SR vs latency study (decode iterations x inference stack)

Goal: one consistent ablation of **closed-loop success vs per-decision latency** for our model, at the
same checkpoint, recipe and hardware as the existing benchmark, so the numbers drop straight into the
latency/compute comparison and the paper's accuracy-vs-compute section.

## Fixed across every cell
- ckpt `eval_ckpts/pulled/wl_ms_dcp_4suite_chunk16_warm_ciai_100k.pt` (h1024 L16, ~758M, chunk 16)
- recipe = `AUTO_sr_scan/4suite_chunk16_warm_exec8_100k/run.sh`: EXEC=8, policy_ctx=0, hold=1,
  gripper rescale_flip+binarize, num_steps_wait=10, action_cfg=1.0, action_temp=0, env seed 0
- LIBERO 4 suites x 10 tasks x 10 trials = 400 episodes per decode cell
- venv `/nvme-data/flexpi-eval-venv` (torch 2.6 + TRT 10.9) for both SR and latency; local RTX A6000

## Two measurements, kept separate
1. **SR (closed loop)** — `run/sr_latency_study.sh`. 4 GPUs, one suite per GPU (same as the reference).
   Also logs in-loop `model_ms_per_decision` (secondary: 4 concurrent sims, includes warmup).
2. **Latency (uncontended)** — `run/latency_grid.sh`. One idle GPU, no sim, `profile/profile_policy.py`
   (same profiler and schema as the baseline table), warmup 10, 30 timed iters, p50/p95, peak VRAM.
   This is the number on the x-axis.

## Cells (launched 2026-10-13 08:17, queue over GPUs 0-3; `sr_latency_study_100k_exec8/`)
| SR cell | decode | stack | latency cell |
|---|---|---|---|
| g1 | greedy, action_n_iter=1 | trt + compile(predictor) + graphs + seq_cache | g_trt_deploy (reused SR) |
| pd1 / pd2 / pd4 / pd8 | parallel decode, choice_temp 4.5, action_temp 0 | trt + seq_cache | pd{k}_trt (pd8 spatial/object/goal reused) |
| g1s | greedy + `WL_ACTION_SINGLE_FWD=1` | trt deploy | g_trt_single |
| g1c | greedy + `INFER_CACHE=1` | trt + seq_cache | g_trt_infercache |
| pd4s | pd4 + `WL_ACTION_SINGLE_FWD=1` | trt + seq_cache | pd4_trt_single |
| g1f | greedy | flex + graphs + seq_cache | g_flex_deploy |
| diag_pd8f | pd8, `CUDA_LAUNCH_BLOCKING=1`, libero_goal only | flex + graphs | none (crash reproduction, exact trace) |

Latency-only extras: g_sdpa, pd{k}_sdpa, pd{k}_flex, pd{k}_trt_cache. The grid and collate run automatically
when the SR queue drains.

## Decode facts and fixes (verified on the real ckpt, real LIBERO val window)
- The action stream is **Ft = 3 tokens** (one per token-frame); only the last is executed. Parallel decode
  commits >= 1 token per iteration, so **at most 3 decode forwards can matter**. The old code ran all k
  forwards (+1 final): pd8 = 9 forwards, 5 of them no-ops.
- **Fix: early exit** in `parallel_decode.maskgit_parallel_decode` once every gen token is committed.
  Verified bit-identical ids and hidden vs the original for k = 1,2,4,8 at temp 0 and 1; forwards
  pd4 5 -> 4, pd8 9 -> 4. pd4 and pd8 now cost the same; they differ only in schedule/choice temperature.
- **Fix: pd + KV cache.** `INFER_CACHE` previously routed to `maskgit_fill_cached`, which ignored pd.
  It now dispatches to `maskgit_parallel_decode_cached` (same per-step update). Verified: ids identical
  to uncached pd on the same gen mask, hidden diff 0.010 (bf16). The cached path generates only the
  **last** frame's action token (past actions stay masked), so pd iterations are irrelevant there and
  pd + cache == greedy + cache at action_temp 0. The cache is therefore *not* exact w.r.t. the uncached
  decode (which fills past action tokens and reads the offset from that hidden); its SR is measured in g1c.
- **Fix: pd + single-forward lever.** `skip_final` used to disable pd silently. pd now returns the hidden
  of its last decode forward (the same approximation the greedy lever makes). Verified: one fewer forward,
  ids unchanged. Also honoured on both cached paths.
- `infer_cache.verify()` returns ok=false (max logit diff 1.25) on this window under bf16 autocast; its
  1e-3 tolerance is not a working gate as configured.
- **flex + CUDA graphs crash: root cause found, fix pending closed-loop confirmation.** With
  `CUDA_LAUNCH_BLOCKING=1` (`diag_pd8f/libero_goal/crash_before_fix.log`) the fault is inside the *replay* of an
  already-captured graph (`inference_runtime.py:176`) at the task4->5 boundary after 4 layout captures. The flex
  BlockMask cache (`forward.py`) evicts independently of the graph cache, so a revisited layout replays a graph
  over freed BlockMask memory. (The first offline test missed it: the first-compiled mask stays pinned by the
  compiled flex kernel.) Fix: dropping the CUDA graphs whenever the flex cache evicts (logged as
  `[runtime] flex mask cache evicted; dropping N CUDA graph(s)`). Confirmation = `diag_pd8f` re-run (re-queued
  first) completing all 10 goal tasks, and `g1f` (greedy, same stack) completing 4 suites.

## Sanity checks to read off the results
- **pd1 vs g1:** with action_temp=0 the categorical sample is effectively argmax, so pd1 should land
  close to g1 in SR. A large gap means the pd path differs from greedy in more than the schedule.
- **Variance:** pd re-masking uses unseeded Gumbel noise, so pd cells are stochastic. Read differences
  smaller than the Wilson CI (about ±3 pts on 400 episodes near 90%) as ties. If a pd cell is within
  CI of g1, re-run it once (new output dir) before claiming a difference.
- **Latency scaling:** pd_k latency should be roughly affine in k (k full-sequence forwards). Deviations
  point at warmup or per-iteration overhead.

## Run
```bash
# 1) closed loop (waits for 4 free GPUs; ~4-5 h, dominated by pd8/pd4 on libero_10)
bash benchmarking/run/sr_latency_study.sh
# 2) latency grid on an idle GPU (~30-60 min)
GPU=3 bash benchmarking/run/latency_grid.sh
# 3) table + figures
/nvme-data/flexpi-eval-venv/bin/python benchmarking/sr_latency_collate.py
```
Outputs: `results/local/sr_latency/{SR_LATENCY.md, sr_latency.csv, sr_vs_latency.pdf, latency_by_backend.pdf}`.
Run 1 and 2 at different times: the latency grid must not share the box with the SR sims.

## Caveats for the paper
- Baseline VLA latencies in `COMPARISON.md` were profiled in `.venv-libero` (torch 2.7.1 then, now 2.11, no
  TRT). Our numbers here use torch 2.6. State the per-model software stack in the table.
- The old `results/local/ours.json` profiled a different checkpoint (`h1024L16_ciai_step100000`, chunk 4).
  Replace "ours" rows with `sr_latency/g_*.json` from this study.
- `profile/worldlab_decision_profile.py` now also sets `ms_flex` / `_compile_forward` / `_seq_layout_cache`
  exactly as the eval does; earlier profiler runs did not, so they are not directly comparable.
