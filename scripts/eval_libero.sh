#!/bin/bash
# Official closed-loop LIBERO eval for a profile, through worldlab's eval script (the harness behind every SR in
# INFERENCE_STUDY.md). Use this for publishable numbers; scripts/run_libero.py is the deployment-loop equivalent.
# Usage: PROFILE=fast SUITES="libero_goal libero_10" GPU=1 NT=10 bash scripts/eval_libero.sh   (DRY_RUN=1: print flags only)
#        PROFILE=max_sr_grounded_mpc ...   (eval-only grounded MPC)
set -uo pipefail
source "$(cd "$(dirname "$0")/.." && pwd)/configs/env.sh"
PROFILE=${PROFILE:-fast}; SUITES=${SUITES-"libero_spatial libero_object libero_goal libero_10"}; GPU=${GPU:-0}; NT=${NT:-10}
OUT=$SW_ROOT/outputs/eval_${PROFILE}_$(date +%Y%m%d_%H%M%S); mkdir -p "$OUT"
read -r FLAGS ENVV < <($SW_PY - "$PROFILE" <<'PY'
import sys, yaml, os
cfg = yaml.safe_load(open(os.path.join(os.environ["SW_ROOT"], "configs", "profiles.yaml"))); name = sys.argv[1]
if name in cfg.get("eval_only", {}):
    print(cfg["eval_only"][name]["flags"].replace(" ", "\x1f"), "-"); sys.exit()
p = cfg["profiles"][name]
f = [f"--attention_backend {p['backend']}"]
if p["compile"]: f.append(f"--compile --compile_scope {p.get('compile_scope', 'predictor')}")
if p["cuda_graphs"]: f.append("--cuda_graphs")
if p["seq_cache"]: f.append("--seq_cache")
if p["backend"] == "flex": f.append("--flex_attn")
f.append(f"--replan_steps {p['exec_steps']} --exec_steps {p['exec_steps']} --hold {p['hold']}")
f.append(f"--gripper_mode {p['gripper_mode']}" + (" --binarize_gripper" if p["binarize_gripper"] else ""))
print(" ".join(f).replace(" ", "\x1f"), "INFER_CACHE=1" if p["kv_cache"] else "-")
PY
)
FLAGS=${FLAGS//$'\x1f'/ }; [ "$ENVV" = "-" ] && ENVV=""
case "$FLAGS" in *--gripper_mode*) ;; *) FLAGS="$FLAGS --gripper_mode rescale_flip --binarize_gripper";; esac
echo "profile=$PROFILE env=[$ENVV] flags=[$FLAGS] -> $OUT" | tee "$OUT/CONFIG.txt"
[ "${DRY_RUN:-0}" = "1" ] && { rm -rf "$OUT"; exit 0; }
cd "$SW_WM_ROOT/worldlab"
for S in $SUITES; do
  env $ENVV CUDA_VISIBLE_DEVICES=$GPU MUJOCO_EGL_DEVICE_ID=$GPU EGL_DEVICE_ID=$GPU \
  $SW_PY scripts/eval/eval_libero_action_worldlab.py --ckpt "$SW_CKPT" --task_suite "$S" --num_trials $NT \
    --dataset_dir "${S}_no_noops_lerobot" --num_steps_wait 10 --policy_ctx 0 --action_cfg 1.0 --action_n_iter 1 --action_temp 0 \
    $FLAGS --enc_jit "$SW_COSMOS/encoder.jit" --dec_jit "$SW_COSMOS/decoder.jit" --data_path "$SW_LIBERO_DATA" \
    --out "$OUT/${S}_sr.json" > "$OUT/${S}.log" 2>&1
  $SW_PY -c "import json;d=json.load(open('$OUT/${S}_sr.json'));print('$S', round(d['overall_sr'],3), d['timing']['model_ms_per_decision'],'ms/decision')" 2>/dev/null | tee -a "$OUT/SUMMARY.txt" || echo "$S FAILED (see $OUT/${S}.log)"
done
