# Output contract

## Per control step: `act, info = policy.step(obs)`

`act`: float32 `(7,)` = `[dx, dy, dz, droll, dpitch, dyaw, gripper]`

| Dims | Meaning | Range |
|---|---|---|
| 0-2 | end-effector translation delta | normalized [-1, 1]; LIBERO's OSC_POSE maps +-1 to +-0.05 m per control step |
| 3-5 | end-effector rotation delta (axis-angle) | normalized [-1, 1]; +-1 maps to +-0.5 rad per control step |
| 6 | gripper | exactly -1 (open) or +1 (close) after `rescale_flip` + binarize |

`info` (`StepInfo`): `decided` (a new chunk was predicted this step), `decision_ms` (synchronized model time),
`chunk` (the full post-processed `(16, 7)` chunk), `queued` (actions left before the next decision).

Controller used in training/eval: robosuite `OSC_POSE`, delta control, kp 150, damping ratio 1,
`input_max` 1, `output_max` [0.05, 0.05, 0.05, 0.5, 0.5, 0.5] (LIBERO default, `env_wrapper.py`).

The model predicts a 16-action chunk per decision; the profile's `exec_steps` (E) of them are executed, then the
policy re-observes. The first decision includes one-time compile warmup (~2-15 s); later decisions take
~440 ms (`fast`/`safe`) or ~540 ms (`reference`) on an RTX A6000.

## Files written by `scripts/run_libero.py --run <name>`

`outputs/<name>/`:

| File | Content |
|---|---|
| `config.json` | profile and run arguments |
| `episodes.jsonl` | one line per episode: suite, task, trial, instruction, success, steps, decisions, decision_ms_mean (excl. first), decision_ms_first, wall_s |
| `summary.json` | episodes, successes, sr, decision_ms_mean_excl_first, s_per_400_actions |
| `<suite>_t<task>_ep<trial>.npz` | with `--save_actions`: actions (steps x 7), decision_steps, decision_ms |

`validate_safe_goal_t01/` is the validation run of this package against the official eval (see `README.md`).
