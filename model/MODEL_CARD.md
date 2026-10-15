# SmolWorld 100k (wl_ms_dcp_4suite_chunk16_warm_ciai_100k)

| | |
|---|---|
| Files | `smolworld_100k.pt` (7.0 GB, symlink to `worldlab/eval_ckpts/pulled/`), `cosmos_tokenizer/` (Cosmos-Tokenizer-DV4x8x8 encoder/decoder JIT, 404 MB, symlink) |
| Parameters | 757.5M total: predictor 331.5M (from scratch), action head/decoder 292.3M, frozen Cosmos codec + DINO 133.7M |
| Predictor | Chameleon-style transformer, hidden 1024, 16 layers, block-causal multimodal RoPE, MaskGIT action decoding |
| Training data | LIBERO spatial / object / goal / 10 (no-op filtered, LeRobot format, MuJoCo 3.3.2 renders with depth), 100k steps |
| Inputs | T = 9 frames (frameskip 1), 2 cameras RGB 256x256, 2 pointmaps 128x128 from depth, 8-dim proprio, language |
| Output | 16-action chunk x 7 dims per decision (VQ-BeT code + continuous offset), executed E actions at a time |
| Control | robosuite OSC_POSE delta actions, gripper {-1 open, +1 close} after post-processing |
| Software | `/nvme-data/flexpi-eval-venv`: torch 2.6.0+cu124, transformers 5.17.0, TensorRT 10.9, MuJoCo 3.3.0, robosuite 1.4.0 |

## Measured performance (LIBERO, RTX A6000)

| Profile | SR (4 suites, 400 ep) | ms / decision | 400 actions |
|---|---|---|---|
| `fast` (SDPA + compile + KV cache, E=12) | 93.8 | 443 | 14.8 s |
| `safe` (SDPA + compile + KV cache, E=8) | 93.5 | 443 | 22.1 s |
| `reference` (TRT + compile + CUDA graphs, E=8) | 94.3 | 543 | 27.2 s |
| grounded MPC (eval only) | 95.0 (800 ep) | 1717 | 85.9 s |

Per suite (`fast`): spatial 90.3 (pooled 300 ep), object 99, goal 96, long 90. Spatial losses concentrate on occluded
targets (bowl in a drawer). Full study: `worldlab/benchmarking/INFERENCE_STUDY.md`.

## Known limitations

- Simulation-only training data; no real-robot evaluation.
- With `transformers` >= 5 the transformer is not compiled on TRT (`reference` profile ~20% slower than with 4.43).
- A 400-episode SR has a 95% CI of about +-2.3 points; differences smaller than that between profiles are not resolved.
- `flex` attention + CUDA graphs crashes in long closed-loop runs (task 4->5 boundary); no profile uses it.
