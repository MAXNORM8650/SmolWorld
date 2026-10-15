# SmolWorld deployment package

Closed-loop inference for SmolWorld (`wl_ms_dcp_4suite_chunk16_warm_ciai_100k`, 758M params): one call per control
step, `observation -> 7-dim action`, with the inference configuration selected in
`reports/INFERENCE_STUDY.md`.

```
configs/    profiles.yaml (fast / safe / reference + eval-only grounded MPC), env.sh (paths, venv, PYTHONPATH)
model/      smolworld_100k.pt and cosmos_tokenizer/ (symlinks), camera_intrinsics_libero.json, MODEL_CARD.md
inputs/     INPUT_SPEC.md (observation contract, real-robot checklist), example_episode.npz (30 real LIBERO steps)
inference/  smolworld_policy.py: SmolWorldPolicy.reset(instruction) / .step(obs) -> (action, info)
scripts/    smoke_test.py, run_libero.py, eval_libero.sh, benchmark_latency.sh, make_example_input.py
outputs/    README.md (output contract), run directories, validation runs
baselines/  BASELINE_SETTINGS.md (exact settings and results of every baseline model)
reports/    study reports + figures (inference study, SR-latency, TTS tree, scaling study, VQ-JEPA tokenizer); see reports/README.md
```

## Quick start

```bash
cd /media/mbzuaiser/SSD1/Komal/Documents/WM/smolworld_deploy
source configs/env.sh
export CUDA_VISIBLE_DEVICES=1 MUJOCO_EGL_DEVICE_ID=1 EGL_DEVICE_ID=1      # an idle GPU

$SW_PY scripts/smoke_test.py --profile fast                              # no simulator: example input -> actions, latency
$SW_PY scripts/run_libero.py --profile fast --suite libero_goal --tasks 0,6 --trials 10 --run demo   # closed loop
PROFILE=fast SUITES="libero_goal" GPU=1 bash scripts/eval_libero.sh       # official harness (DRY_RUN=1 prints flags)
PROFILE=fast GPU=3 bash scripts/benchmark_latency.sh                      # uncontended latency
```

## Python API

```python
from inference import SmolWorldPolicy
policy = SmolWorldPolicy("fast")                 # loads the model once (~1 min) and applies the profile
policy.reset("open the middle drawer of the cabinet")
while not done:
    obs = {"rgb": [third_person_rgb, wrist_rgb],  # (H,W,3) uint8 each
           "depth_mm": [third_person_depth, wrist_depth],   # (H,W) uint16 mm, aligned to rgb
           "proprio": eef_pos_axisangle_gripper,  # (8,)
           "rgb_rot180": False}                   # True only for raw LIBERO env renders
    action, info = policy.step(obs)               # (7,) [dx,dy,dz,droll,dpitch,dyaw,gripper]
    robot.apply(action)
```
Call `step()` on **every** control step (the model conditions on a 9-step history); a new 16-action chunk is
predicted only when the queue of `exec_steps` actions is empty (`info.decided`). Contracts: `inputs/INPUT_SPEC.md`,
`outputs/README.md`.

## Profiles

| Profile | Stack | E | SR (LIBERO 4 suites) | ms / decision | 400 actions |
|---|---|---|---|---|---|
| **fast** (default) | SDPA + compile + KV cache | 12 | 93.8 | 443 | 14.8 s |
| safe | SDPA + compile + KV cache | 8 | 93.5 | 443 | 22.1 s |
| reference | TRT + compile + CUDA graphs | 8 | 94.3 | 543 | 27.2 s |
| max_sr_grounded_mpc (eval only) | grounded MPC, flex | 8 | 95.0 | 1717 | 85.9 s |

`fast` re-observes every 12 actions: it loses a few episodes on occluded targets (spatial 90.3 vs 95.3). Use `safe`
when the scene has occlusion or precise insertion.

## Validation (this package vs the official eval)

The closed loop is deterministic given the actions, so the package must reproduce the official eval episode by
episode. `safe` profile, `libero_goal`, compared with the official run of the same configuration
(`AUTO_sr_scan/tts_tree_100k/sdpa_kvc_compile`):

| Task | Package | Official eval |
|---|---|---|
| 0 | 10/10 `1111111111` | 10/10 `1111111111` |
| 1 | 10/10 `1111111111` | 10/10 `1111111111` |
| 6 (mixed outcomes) | 6/10 `1101001011` | 6/10 `1101001011` |

30/30 identical episode outcomes; 441-455 ms per decision in the loop. Runs: `outputs/validate_safe_goal_t01/`,
`outputs/validate_safe_goal_t6/`. Smoke test (`fast`): 436 ms steady per decision.

## Requirements and provenance

- GPU with >= 8 GB free (peak ~6.7 GB); validated on RTX A6000.
- Python env `/nvme-data/flexpi-eval-venv` (torch 2.6.0+cu124, transformers 5.17.0, TensorRT 10.9, MuJoCo 3.3.0,
  robosuite 1.4.0, PyYAML). Paths are set in `configs/env.sh`.
- Code: imports the model and decoding code from the `worldlab` checkout at `SW_WM_ROOT` (plus `flex-pi` for the
  LIBERO helpers and `Lumos`). That code is uncommitted in git (last commit 2026-08-06) and includes the 2026-10-13
  changes to `forward.py`, `generate.py`, `infer_cache.py` and `parallel_decode.py` described in INFERENCE_STUDY.md.
  For a portable copy, snapshot those trees together with this folder.
- The model was trained on LIBERO simulation only; see the real-robot checklist in `inputs/INPUT_SPEC.md`.
