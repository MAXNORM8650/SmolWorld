"""Record a real LIBERO observation sequence in the deployment input format -> inputs/example_episode.npz.
Settles 10 no-op steps on libero_goal task 0 (init state 0), then records `--steps` observations while holding
still (no-op actions). Each record is exactly what SmolWorldPolicy.step() consumes (see inputs/INPUT_SPEC.md)."""
import argparse, os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from scripts.run_libero import libero_obs, SETTLE

ap = argparse.ArgumentParser(); ap.add_argument("--steps", type=int, default=12); a = ap.parse_args()
import libero_utils as LU
from libero.libero import benchmark as libero_benchmark
bench = libero_benchmark.get_benchmark_dict()["libero_goal"](); task = bench.get_task(0)
env, desc = LU.get_libero_env(task, 256, 0, camera_depths=True)
env.reset(); obs = env.set_init_state(bench.get_task_init_states(0)[0])
for _ in range(SETTLE):
    obs, _, _, _ = env.step(LU.get_libero_dummy_action())
rgb, dep, pro = [], [], []
for _ in range(a.steps):
    o = libero_obs(obs, env, LU)
    rgb.append(np.stack(o["rgb"])); dep.append(np.stack(o["depth_mm"])); pro.append(o["proprio"])
    obs, _, _, _ = env.step(LU.get_libero_dummy_action())
out = os.path.join(ROOT, "inputs", "example_episode.npz")
np.savez_compressed(out, rgb=np.stack(rgb), depth_mm=np.stack(dep), proprio=np.stack(pro), instruction=desc,
                    rgb_rot180=True, source="libero_goal task 0, init state 0, after 10 settle steps")
print(f"saved {out}: rgb {np.stack(rgb).shape} {np.stack(rgb).dtype}, depth_mm {np.stack(dep).shape} {np.stack(dep).dtype}, "
      f"proprio {np.stack(pro).shape}, instruction={desc!r}")
