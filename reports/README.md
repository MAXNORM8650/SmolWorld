# Reports

Copies of the study reports from `worldlab/benchmarking/` (snapshot 2026-10-15). Figures are in `figures/` with the
same relative paths as the originals, so the images render here; `figures/make_*.py` regenerate them from the result
files on disk (paths inside the scripts point at the original `worldlab/benchmarking` and `/nvme-data` locations).

| Report | What it covers |
|---|---|
| [INFERENCE_STUDY.md](INFERENCE_STUDY.md) | Speed vs success rate over every inference configuration; source of the `fast` / `safe` / `reference` profiles |
| [SR_LATENCY_STUDY.md](SR_LATENCY_STUDY.md) | Closed-loop SR vs per-decision latency (decode iterations x inference stack) |
| [TTS_TREE.md](TTS_TREE.md) | Incremental, SR-gated speedup tree for episode wall-clock |
| [SCALING_STUDY.md](SCALING_STUDY.md) | Val loss vs model size (1.2M–6B), tokens and compute; compute-optimal frontier; gaps to close |
| [VQ_JEPA_TOKENIZER.md](VQ_JEPA_TOKENIZER.md) | Discrete VQ-BeT-style patch tokens from frozen V-JEPA 2.1 (research, not used by the deployed model) |
