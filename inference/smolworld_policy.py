"""SmolWorld closed-loop policy: reset(instruction) -> step(obs) -> one 7-dim action per control step.

Reproduces the control logic of worldlab/scripts/eval/eval_libero_action_worldlab.py (the configuration whose SR is
reported in INFERENCE_STUDY.md) for the greedy profiles in configs/profiles.yaml:
  * every control step is appended to a history of (T-1)*FS+1 frames; a decision subsamples T frames from it
    (pad-repeat at episode start)                                                   [eval: build_batch, policy_ctx=0]
  * RGB -> optional rot180 -> bilinear resize to 256 -> [-1,1]                      [eval: _u8_rot_resize]
  * depth (uint16 mm) -> nearest resize to 128 -> pointmap with per-axis-scaled K   [eval: _depth_to_pointmap, _load_K]
  * proprio = eef_pos(3) + eef axis-angle(3) + gripper qpos(2)                      [eval: obs_frame]
  * one decision = generate_actions (greedy, n_iter 1, cfg 1.0, bf16 autocast) -> last frame's 16x7 chunk
  * gripper rescale_flip ({0,1} -> {+1,-1}) + binarize; execute the first exec_steps actions, each `hold` times
Input/Output contract: inputs/INPUT_SPEC.md, outputs/README.md.
"""
from __future__ import annotations

import json
import os
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAMS = ("cam_high", "cam_left_wrist")                  # model camera order: agentview / third-person, wrist


def load_profile(name: str | None = None, path: str | None = None) -> dict:
    cfg = yaml.safe_load(open(path or os.path.join(ROOT, "configs", "profiles.yaml")))
    name = name or cfg["default"]
    if name not in cfg["profiles"]:
        raise KeyError(f"profile {name!r} not in {sorted(cfg['profiles'])} (eval-only modes: {sorted(cfg.get('eval_only', {}))})")
    return {"name": name, **cfg["profiles"][name]}


@dataclass
class StepInfo:
    decided: bool                      # a new chunk was predicted at this step
    decision_ms: float | None          # synchronized model time of that decision
    chunk: np.ndarray | None           # (16, 7) post-processed chunk of that decision
    queued: int                        # actions still queued after this step
    extra: dict = field(default_factory=dict)


class SmolWorldPolicy:
    def __init__(self, profile: str | dict | None = None, ckpt: str | None = None, cosmos_dir: str | None = None,
                 intrinsics: str | dict | None = None, data_path: str | None = None, device: str = "cuda"):
        from eval_multistream import load_model                                     # worldlab/scripts/eval
        from worldlab.components.multistream import generate as GEN
        from worldlab.components.multistream.inference_runtime import configure
        from worldlab.data.pointmap import pointmap_from_depth
        self._GEN, self._pointmap_from_depth = GEN, pointmap_from_depth
        self.p = profile if isinstance(profile, dict) else load_profile(profile)
        self.dev = device
        ckpt = ckpt or os.environ.get("SW_CKPT", os.path.join(ROOT, "model", "smolworld_100k.pt"))
        cosmos_dir = cosmos_dir or os.environ.get("SW_COSMOS", os.path.join(ROOT, "model", "cosmos_tokenizer"))
        ov = {"enc_jit": os.path.join(cosmos_dir, "encoder.jit"), "dec_jit": os.path.join(cosmos_dir, "decoder.jit")}
        if data_path or os.environ.get("SW_LIBERO_DATA"):
            ov["data_path"] = data_path or os.environ["SW_LIBERO_DATA"]
        m, cfg = load_model(ckpt, device, ov)
        m.eval()
        # runtime levers, same attributes and order as the eval script
        if self.p["kv_cache"]:
            m._infer_cache = True
        if self.p["seq_cache"]:
            m._seq_layout_cache = True
        if self.p["backend"] == "flex":
            m.predictor.ms_flex = True
        if self.p["compile"]:
            m.predictor._compile_forward = True
        configure(m.predictor, m, backend=self.p["backend"], compile_model=bool(self.p["compile"]), compile_mode="default",
                  cuda_graphs=bool(self.p["cuda_graphs"]), cache_layout=bool(self.p["seq_cache"]),
                  compile_scope=self.p.get("compile_scope", "predictor"), strict_cfg=1)
        self.m, self.cfg = m, cfg
        ex = cfg.data.extra
        self.T = int(ex.get("video_len", 9)); self.FS = int(cfg.data.frameskip)
        self.rgb_size = int(getattr(cfg.data, "img_size", 256))
        phw = ex.get("pointmap_hw", None)
        self.pm_hw = (int(phw[0]), int(phw[1])) if phw else (int(ex.get("pointmap_img", 128)),) * 2
        self.depth_fill = str(ex.get("depth_fill", "zero"))
        self.chunk = int(ex.get("action_chunk") or ex.get("action_video_freq_ratio") or self.FS)
        self.adim = int(cfg.data.action_dim)
        self.use_depth = bool(ex.get("depth_keys"))
        if list(cfg.data.cam_keys)[0].split(".")[-1] != CAMS[0]:
            raise ValueError(f"unexpected camera order in checkpoint: {list(cfg.data.cam_keys)}")
        intr = intrinsics or os.environ.get("SW_INTRINSICS", os.path.join(ROOT, "model", "camera_intrinsics_libero.json"))
        self.intrinsics = intr if isinstance(intr, dict) else json.load(open(intr))
        self.K = [self._scaled_K(self.intrinsics[c]) for c in CAMS] if self.use_depth else None
        self._ac = torch.autocast("cuda", dtype=torch.bfloat16)
        self.instruction = None
        self.reset("")

    # ------------------------------------------------------------------------------------------------------------
    def _scaled_K(self, c: dict) -> torch.Tensor:
        sw = self.pm_hw[1] / float(c["width"]); sh = self.pm_hw[0] / float(c.get("height", c["width"]))
        return torch.tensor([[c["fx"] * sw, 0, c["cx"] * sw], [0, c["fy"] * sh, c["cy"] * sh], [0, 0, 1]], dtype=torch.float32)

    def _rgb(self, img_hwc: np.ndarray, rot180: bool) -> torch.Tensor:
        x = torch.from_numpy(np.ascontiguousarray(img_hwc)).permute(2, 0, 1).float() / 255.0
        if rot180:
            x = torch.flip(x, dims=[1, 2])
        if x.shape[-1] != self.rgb_size or x.shape[-2] != self.rgb_size:
            x = F.interpolate(x[None], size=(self.rgb_size, self.rgb_size), mode="bilinear", align_corners=False)[0]
        return x * 2 - 1

    def _pmap(self, depth_mm: np.ndarray, K: torch.Tensor) -> torch.Tensor:
        d = torch.from_numpy(np.ascontiguousarray(depth_mm)).float()[None, None]
        d = F.interpolate(d, size=self.pm_hw, mode="nearest")
        return self._pointmap_from_depth(d, K[None], depth_fill=self.depth_fill)[0]

    def _frame(self, obs: dict) -> dict:
        rot = bool(obs.get("rgb_rot180", False))
        fr = {"rgb": [self._rgb(obs["rgb"][i], rot) for i in range(2)], "instr": self.instruction}
        if self.use_depth:
            fr["pmap"] = [self._pmap(obs["depth_mm"][i], self.K[i]) for i in range(2)]
        p = np.asarray(obs["proprio"], dtype=np.float32)
        if p.shape != (8,):
            raise ValueError(f"proprio must be (8,) = eef_pos(3)+axis_angle(3)+gripper_qpos(2), got {p.shape}")
        fr["proprio"] = p
        return fr

    def _batch(self) -> dict:
        hl = list(self.hist)
        idx = np.linspace(0, len(hl) - 1, self.T).round().astype(int)
        fr = [hl[i] for i in idx]
        b = {"video": torch.stack([torch.stack(f["rgb"], 0) for f in fr], 0)[None].to(self.dev),
             "actions": torch.zeros(1, self.T, self.chunk * self.adim, device=self.dev),
             "instruction": [hl[-1]["instr"]]}
        if self.use_depth:
            b["pointmap"] = torch.stack([torch.stack(f["pmap"], 0) for f in fr], 0)[None].to(self.dev)
        b["proprio"] = torch.stack([torch.from_numpy(f["proprio"]) for f in fr], 0)[None].float().to(self.dev)
        return b

    def _postprocess(self, chunk: np.ndarray) -> np.ndarray:
        gm = self.p.get("gripper_mode", "rescale_flip")
        if gm == "rescale_flip":
            chunk[:, -1] = 1.0 - 2.0 * chunk[:, -1]
        elif gm == "rescale":
            chunk[:, -1] = 2.0 * chunk[:, -1] - 1.0
        elif gm == "flip":
            chunk[:, -1] *= -1.0
        if self.p.get("binarize_gripper", True):
            chunk[:, -1] = np.where(chunk[:, -1] >= 0, 1.0, -1.0)
        return chunk

    # ------------------------------------------------------------------------------------------------------------
    def reset(self, instruction: str) -> None:
        """Start an episode. Call after the robot has settled (the eval waits 10 no-op steps first)."""
        self.instruction = instruction
        self.hist = deque(maxlen=(self.T - 1) * self.FS + 1)
        self.pending: list[np.ndarray] = []

    @torch.no_grad()
    def decide(self) -> tuple[np.ndarray, float]:
        """Predict a chunk from the current history; returns (post-processed (16,7) chunk, synchronized ms)."""
        batch = self._batch()
        torch.cuda.synchronize(); t0 = time.perf_counter()
        with self._ac:
            acts, _ = self._GEN.generate_actions(self.m, batch, n_iter=1, temp=0.0, greedy=True, cfg_scale=1.0)
        chunk = acts[0, -1].float().cpu().numpy().reshape(self.chunk, self.adim)
        torch.cuda.synchronize()
        return self._postprocess(chunk), 1000.0 * (time.perf_counter() - t0)

    def step(self, obs: dict) -> tuple[np.ndarray, StepInfo]:
        """Feed this control step's observation, get the action to execute now."""
        self.hist.append(self._frame(obs))
        decided, ms, chunk = False, None, None
        if not self.pending:
            chunk, ms = self.decide()
            ex = chunk[:max(1, int(self.p["exec_steps"]))]
            h = int(self.p.get("hold", 1))
            self.pending = [a for a in ex for _ in range(h)] if h > 1 else list(ex)
            decided = True
        act = self.pending.pop(0)
        return act.astype(np.float32), StepInfo(decided, ms, chunk, len(self.pending))
