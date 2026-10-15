"""Smoke test without a simulator: load the policy, feed inputs/example_episode.npz step by step, check outputs.
Usage: python scripts/smoke_test.py [--profile fast]"""
import argparse, os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from inference import SmolWorldPolicy

ap = argparse.ArgumentParser(); ap.add_argument("--profile", default=None); a = ap.parse_args()
ex = np.load(os.path.join(ROOT, "inputs", "example_episode.npz"))
pol = SmolWorldPolicy(a.profile)
pol.reset(str(ex["instruction"]))
ms = []
for t in range(len(ex["proprio"])):
    obs = {"rgb": list(ex["rgb"][t]), "depth_mm": list(ex["depth_mm"][t]), "proprio": ex["proprio"][t],
           "rgb_rot180": bool(ex["rgb_rot180"])}
    act, info = pol.step(obs)
    assert act.shape == (7,) and np.isfinite(act).all(), act
    assert abs(act[-1]) == 1.0, "gripper must be binarized to +/-1"
    if info.decided:
        ms.append(info.decision_ms)
        assert info.chunk.shape == (pol.chunk, pol.adim)
    print(f"t={t:2d} decided={info.decided!s:5} queued={info.queued:2d} action={np.round(act, 3).tolist()}")
print(f"\nOK profile={pol.p['name']} decisions={len(ms)} first(warmup)={ms[0]:.0f} ms"
      + (f" steady={np.mean(ms[1:]):.0f} ms" if len(ms) > 1 else ""))
