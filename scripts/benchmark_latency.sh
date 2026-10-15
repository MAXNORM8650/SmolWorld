#!/bin/bash
# Uncontended per-decision latency of a profile (profiler: 10 warmup + 30 timed decisions, p50/p95/peak VRAM).
# Same profiler as worldlab/benchmarking (baseline comparison table). Run on an idle GPU.
# Usage: PROFILE=fast GPU=3 bash scripts/benchmark_latency.sh
set -uo pipefail
source "$(cd "$(dirname "$0")/.." && pwd)/configs/env.sh"
PROFILE=${PROFILE:-fast}; GPU=${GPU:-0}; OUT=$SW_ROOT/outputs/latency_${PROFILE}.json
eval "$($SW_PY - "$PROFILE" <<'PY'
import sys, yaml, os
p = yaml.safe_load(open(os.path.join(os.environ["SW_ROOT"], "configs", "profiles.yaml")))["profiles"][sys.argv[1]]
print(f"export WL_DEPLOY_BACKEND={p['backend']} WL_COMPILE={int(p['compile'])} WL_CUDA_GRAPHS={int(p['cuda_graphs'])} "
      f"WL_SEQ_CACHE={int(p['seq_cache'])} INFER_CACHE={int(p['kv_cache'])} WL_COMPILE_SCOPE={p.get('compile_scope', 'predictor')} "
      f"EXEC={p['exec_steps']}")
PY
)"
used=$(nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits)
[ "$used" -ge 2000 ] && { echo "GPU $GPU is busy (${used} MiB); latency would be contaminated"; exit 1; }
cd "$SW_WM_ROOT/worldlab"
CUDA_VISIBLE_DEVICES=$GPU $SW_PY benchmarking/profile/profile_policy.py --name "smolworld_$PROFILE" --worldlab_ckpt "$SW_CKPT" \
  --cosmos "$SW_COSMOS" --warmup 10 --iters 30 --out "$OUT" > "${OUT%.json}.log" 2>&1
$SW_PY -c "import json;d=json.load(open('$OUT'));p=d['p50_ms'];print(f'$PROFILE: p50 {p:.0f} ms/decision, p95 {d[\"p95_ms\"]:.0f}, peak {d[\"peak_vram_gb\"]:.1f} GB -> 400 actions at E=$EXEC: {400/$EXEC*p/1000:.1f} s')"
