# Baseline model settings

Settings each baseline was run with, so comparisons against SmolWorld stay reproducible. Sources:
`worldlab/artifacts/lerobot_baselines/{manifest.json, summary.csv, runs/<model>/command.json}`,
`worldlab/benchmarking/BASELINE_EVAL_COMMANDS.md`, `flex-pi/configs/sim_libero.yaml`,
`flex-pi/run_flexpi_deployed_latency.sh`, `worldlab/benchmarking/results/local/{COMPARISON_FULL.md, END_TO_END_LATENCY.md}`.

## Protocols differ: do not mix SR across them

| | LeRobot baselines | flex-pi | SmolWorld (this package) |
|---|---|---|---|
| Suites | LIBERO-Goal only | 4 suites | 4 suites |
| Episodes | 10 tasks x 5 (init states 0-4) | per its harness | 10 tasks x 10 per suite (init states 0-9) |
| Episode limit | 300 steps | its harness | 400 (spatial/object/goal), 700 (long) |
| Settle steps | 10 | 30 (`num_steps_wait`) | 10 |
| Seed | 1000 | its harness | env seed 0 |
| Env | LeRobot 0.5.2, torch 2.11, MuJoCo 3.3.2 (isolated), transformers 5.12 | own venv | MuJoCo 3.3.0, torch 2.6, transformers 5.17 |
| Timing | fresh-chunk latency in real rollouts, first 10 chunks excluded | `model_ms_per_decision` | in-loop steady ms/decision |

## LeRobot baselines (LIBERO-Goal, RTX A6000)

Common command (`scripts/eval/eval_lerobot_baseline_audited.py`):
```
--env.type=libero --env.task=libero_goal --env.episode_length=300 --env.max_parallel_tasks=1
--eval.batch_size=1 --eval.n_episodes=5 --seed=1000 --policy.device=cuda
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 MUJOCO_GL=egl
```
Model-specific overrides appended to it (from `manifest.json`); the expanded command of every attempt is in
`runs/<model>/command.json`.

| Model | Checkpoint @ revision | Model-specific overrides | Executed actions / chunk | Goal SR | p50 ms / chunk | Params (M) |
|---|---|---|---|---|---|---|
| XVLA | `lerobot/xvla-libero` @ `12e8783e` | `--env.control_mode=absolute` | 30 | 98.0 | 107 | 879 |
| LaWAM | `jialei02/lawam-libero-sft-lerobot` @ `f3d7b131` | none | 8 | 96.0 | 165 | 2555 |
| Pi0+FAST | `lerobot/pi0fast-libero` @ `840f4b50` | none | 10 | 90.0 | 495 | 3450 |
| Pi0 | `lerobot/pi0_libero_finetuned_v044` @ `45dcc8fc` | `--policy.n_action_steps=10` (published LIBERO recipe) | 10 | 82.0 | 88 | 4028 |
| Pi0.5 | `lerobot/pi05_libero_finetuned_v044` @ `8e174154` | `--policy.n_action_steps=10` | 10 | failed / retry pending | - | 4143 |
| Fast-WAM | `ZibinDong/fastwam_libero_uncond_2cam224` @ `a784645f` | `--policy.torch_dtype=bfloat16 --policy.n_action_steps=10 --policy.tokenizer_model_id=<local umt5-xxl>` | 10 | 0.0 (BF16 A6000 profile; published recipe is FP32 on H20, SR not claimed) | 488 | 6021 |
| LingBot-VA | `lerobot/lingbot_va_libero_long` @ `73e0be23` | `--env.observation_height=128 --env.observation_width=128` (Long checkpoint, Goal transfer) | - | partial | 4498 | 5089 |
| GR00T N1.7 | `nvidia/gr00t17-lerobot-libero_goal-640` @ `436c57c0` | `--policy.embodiment_tag=libero_sim` | - | failed / retry pending | - | - |
| MolmoAct2 | `allenai/MolmoAct2-LIBERO-LeRobot` @ `f0c5a956` | `--policy.model_dtype=bfloat16 --policy.use_amp=true --policy.enable_inference_cuda_graph=true --policy.per_episode_seed=true --policy.eval_seed=1000 --env.camera_name_mapping={"agentview_image":"image","robot0_eye_in_hand_image":"wrist_image"}` | - | failed / retry pending | - | - |
| VLA-JEPA | `lerobot/VLA-JEPA-LIBERO` @ `735d9f69` | none | - | failed / retry pending | - | - |
| EVO1 | `zuoxingdong/evo1_libero` @ `515921f4` | `--policy.vlm_dtype=bfloat16 --policy.use_amp=true --policy.use_flash_attn=false --policy.enable_gradient_checkpointing=false --policy.n_action_steps=14 --policy.postprocess_action_dim=7 --policy.binarize_gripper=true --env.observation_height=448 --env.observation_width=448 --policy.gripper_threshold=0.0 --policy.gripper_below_threshold_value=-1.0 --policy.gripper_above_threshold_value=1.0` | - | failed / retry pending | - | - |
| SmolVLA | `HuggingFaceVLA/smolvla_libero` @ `6721902b` | none | - | failed / retry pending | - | - |
| EO-1 | `lerobot/eo1-base` @ `bba306bb` | latency only (`scripts/eval/benchmark_eo1_base.py`); no verified Goal checkpoint | - | not reported | - | - |
| WALL-OSS | `ases200q2/lerobot_libero_wallx_diffusion_20260321_1003` @ `f5d24b79` | `--env.observation_height=360 --env.observation_width=360` (trained on LIBERO-Object; Goal transfer only) | - | pending | - | - |

Status as recorded in `summary.csv`; "failed / retry pending" models have no valid SR yet.

## flex-pi (WAN 2.2-initialized world-action model, 6.04B)

Deployed action-control configuration (`run_flexpi_deployed_latency.sh`, `configs/sim_libero.yaml`):
```
python experiments/libero/eval_libero_batch.py task=libero_unified_flex_2cam224_32d_rotvec_1e-4 \
  ckpt=<flexpi_repro>/weights/flexpi-libero/checkpoints/weights/step_010860.pt \
  EVALUATION.dataset_stats_path=<flexpi_repro>/weights/flexpi-libero/dataset_stats.json \
  +EVALUATION.camera_intrinsics_path=<data>/<suite>_no_noops_lerobot/meta/camera_intrinsics.json \
  EVALUATION.num_inference_steps=4 EVALUATION.torch_compile=true EVALUATION.dynamic_step_skip=true \
  +EVALUATION.infer_present_video=true +EVALUATION.infer_present_dino=true +EVALUATION.infer_present_pointmap=true \
  +EVALUATION.infer_joint_video=false +EVALUATION.infer_joint_dino=false +EVALUATION.infer_joint_pointmap=false
```
`replan_steps` 10, `num_steps_wait` 30, `text_cfg_scale` 1.0 (`configs/sim_libero.yaml`). Measured 375 ms/decision
(action-only, 4 flow steps) -> 15.0 s per 400 actions. SR from the paper table: 98.4 (4-suite);
our-harness Goal-only check: 100.0.

## SmolWorld settings to compare against

Use `configs/profiles.yaml`: `fast` (E=12) or `safe` (E=8) for the latency-optimized configuration, `reference` for
the original deploy stack. For a latency comparison report each model's executed actions per decision: 400-action
time = 400 / executed actions x ms per decision (`worldlab/benchmarking/results/local/END_TO_END_LATENCY.md`).
