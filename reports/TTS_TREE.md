# TTS speedup tree (incremental, SR-gated)

Goal: the lowest **episode wall-clock** at high SR for `wl_ms_dcp_4suite_chunk16_warm_ciai_100k`, found by adding
one inference lever at a time. Replaces the full grid in `SR_LATENCY_STUDY.md` (paused 2026-10-13).
Code: `tts_tree/tts_tree.py` (orchestrator), `tts_tree/equiv.py` (action equivalence).
Output: `/nvme-data/Komal/wm/ckpts/ms_eval_ckpts/AUTO_sr_scan/tts_tree_100k/{TREE.md, tree.json, TREE.log}`.

## Acceptance rule (fixed before any result)
- SR floor anchored to the root reference (.97/1.00/.94/.90 = 95.25): 4-suite avg >= root - 2 pts and no
  suite below root - 5 pts. Anchored to the root, not the parent, so small losses cannot accumulate.
- Accepted only if measured wall-clock (s/episode) is also lower than its base node.
- Wall-clock is reported per episode (objective) and per env step (failed episodes run to the step limit, so
  per-episode time alone mixes SR into speed).

## Search
```
ROOT trt+compile+graphs, greedy n=1, E=8     (re-measured under the tree's pinned conditions)
L1 kernel        flex+graphs | sdpa+compile     action-equivalent on 200 real windows -> SR inherited, no sim
L2 forward count +single-fwd | +KV cache        SR-gated; if both pass alone, their combination is tried
L3 decisions     E=12                           SR-gated; if it fails -> repair: E=12 + pd2 | E=12 + pd3
```
Each level expands from the fastest accepted node so far. Per child:
1. Latency (profiler, uncontended, GPU 3): predicted wall-clock = base wall - base model time + child model
   time (decisions scale with 8/E). Pruned if the predicted gain < 3 %.
2. Kernel-only levers: executed-chunk actions compared to the base on 200 fixed val windows; all within 0.02
   -> identical trajectories, SR inherited, wall-clock predicted.
3. Otherwise SR stage A (libero_10 + libero_goal, 200 episodes); fail -> rejected without stage B.
   Stage B (spatial + object) completes the 4-suite numbers.

Dropped as levers (no sequence-length saving or incompatible): `policy_ctx` keeps T token-frames;
sliding-window attention needs the dense SDPA mask (not trt/flex). Parallel decode is never cheaper than greedy,
so it only appears as a repair lever.

## Conditions
- SR slots: GPUs 0,1,2, one sim per GPU, each pinned to 16 dedicated cores (`taskset`), OMP/MKL 8 threads.
- Latency + equivalence: GPU 3, serialized.
- Same recipe as the reference eval otherwise (EXEC per node, policy_ctx 0, hold 1, gripper rescale_flip +
  binarize, num_steps_wait 10, 10 tasks x 10 trials per suite).
- Resumable: re-running `tts_tree.py` re-reads every finished latency/equivalence/SR file.

## Caveats
- L1 flex+graphs depends on the forward.py flex-eviction fix; its confirmation run is
  `sr_latency_study_100k_exec8/diag_pd8f`.
- SR is paired with the root at the task level only (the eval does not log per-trial outcomes).
- Concurrency varies between 1 and 3 sims; pinned cores limit, but do not remove, wall-clock interference.

## Result (2026-10-13)
Model `wl_ms_dcp_4suite_chunk16_warm_ciai_100k`; SR = closed-loop LIBERO 4 suites x 100 episodes; latency = uncontended
in-loop steady ms/decision (warmup cancelled by differencing two run lengths, `results/local/item2/INLOOP.md`).

| Operating point | SR (4-suite) | ms/decision | 400 actions | Speedup |
|---|---|---|---|---|
| Root: TRT + compile + graphs, E=8 | 94.3 | 543 | 27.2 s | 1.00x |
| KV cache (INFER_CACHE=1), E=8 | 93.5 | 462 | 23.1 s | 1.18x |
| **KV cache, E=12 (best)** | **93.8** (spatial pooled 271/300) | 462 | **15.4 s** | **1.77x** |

Rejected: single-forward (goal 0.70), KV cache + E=10 (92.0; E10 < E12 is within run-to-run noise of ~1.2 pts on
400 episodes). KV cache + E=12 was first rejected on spatial (0.89, guard 0.92); a paired 200-episode spatial
replicate (`tts_tree_100k/replicates/`) gave root 189 vs E12 182 (McNemar p=0.14), pooled spatial 90.3% vs 95.3%
(-4.9, within the 5-pt guard), and it was accepted by the user. Its losses concentrate on spatial task 4 (top
drawer, occluded) and task 0: committing 12 actions without re-observing hurts most under occlusion.
Kernel notes: CUDA graphs and compile are no-ops on the KV-cache path (identical actions and timing); flex must not
be combined with the KV cache (3x slower). Latency regression vs the 2026-09-25 reference (447 -> 543 ms for the
same flags) is unexplained; transformer compile on TRT is not the cause (it crashes under both scopes today).
