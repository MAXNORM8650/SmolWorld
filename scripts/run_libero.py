"""Closed-loop LIBERO episodes driven by SmolWorldPolicy (the same loop a real robot runs: observe -> step -> execute).

Mirrors the official eval's episode protocol: LIBERO init state `trial % n_init`, env seed 0, 10 no-op settle steps,
max steps 400 (spatial/object/goal) or 700 (libero_10). Writes outputs/<run>/{episodes.jsonl, summary.json} and, with
--save_actions, one .npz per episode (actions, decision steps, decision ms).
Usage: python scripts/run_libero.py --profile fast --suite libero_goal --tasks 0,1 --trials 10 --run my_run
"""
import argparse, json, os, sys, time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
MAX_STEPS = {"libero_spatial": 400, "libero_object": 400, "libero_goal": 400, "libero_10": 700, "libero_90": 700}
SETTLE = 10


def libero_obs(obs, env, LU):
    dep = LU.get_libero_per_cam_depth_uint16_mm(obs, env)             # already rot180-aligned with dataset RGB
    dk = list(dep.keys())
    return {"rgb": [obs["agentview_image"], obs["robot0_eye_in_hand_image"]], "rgb_rot180": True,   # raw env renders
            "depth_mm": [dep[dk[0]], dep[dk[1]]],
            "proprio": np.concatenate([obs["robot0_eef_pos"], LU.quat2axisangle(obs["robot0_eef_quat"]),
                                       obs["robot0_gripper_qpos"]]).astype(np.float32)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default=None); ap.add_argument("--suite", default="libero_goal")
    ap.add_argument("--tasks", default="0"); ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--run", default=None); ap.add_argument("--save_actions", action="store_true")
    a = ap.parse_args()
    import libero_utils as LU
    from libero.libero import benchmark as libero_benchmark
    from inference import SmolWorldPolicy
    pol = SmolWorldPolicy(a.profile)
    run = a.run or f"{pol.p['name']}_{a.suite}_{time.strftime('%Y%m%d_%H%M%S')}"
    out = os.path.join(ROOT, "outputs", run); os.makedirs(out, exist_ok=True)
    json.dump({"profile": pol.p, "suite": a.suite, "tasks": a.tasks, "trials": a.trials}, open(os.path.join(out, "config.json"), "w"), indent=2)
    bench = libero_benchmark.get_benchmark_dict()[a.suite]()
    max_steps = MAX_STEPS.get(a.suite, 400)
    epf = open(os.path.join(out, "episodes.jsonl"), "a")
    tot_s = tot_n = 0; all_ms = []
    for tid in [int(x) for x in a.tasks.split(",")]:
        task = bench.get_task(tid)
        env, desc = LU.get_libero_env(task, pol.rgb_size, 0, camera_depths=pol.use_depth)
        inits = bench.get_task_init_states(tid)
        for trial in range(a.trials):
            env.reset(); obs = env.set_init_state(inits[trial % len(inits)])
            pol.reset(desc)
            t, done, ms, acts, dsteps = 0, False, [], [], []
            t0 = time.perf_counter()
            while t < max_steps + SETTLE and not done:
                if t < SETTLE:
                    obs, _, done, _ = env.step(LU.get_libero_dummy_action()); t += 1; continue
                act, info = pol.step(libero_obs(obs, env, LU))
                if info.decided:
                    ms.append(info.decision_ms); dsteps.append(t)
                acts.append(act)
                obs, _, done, _ = env.step(act); t += 1
            rec = {"suite": a.suite, "task": tid, "trial": trial, "instruction": desc, "success": bool(done),
                   "steps": t - SETTLE, "decisions": len(ms), "decision_ms_mean": float(np.mean(ms[1:] or ms)),
                   "decision_ms_first": float(ms[0]) if ms else None, "wall_s": round(time.perf_counter() - t0, 2)}
            epf.write(json.dumps(rec) + "\n"); epf.flush()
            all_ms += ms[1:]; tot_s += int(done); tot_n += 1
            if a.save_actions:
                np.savez_compressed(os.path.join(out, f"{a.suite}_t{tid}_ep{trial}.npz"), actions=np.stack(acts),
                                    decision_steps=np.array(dsteps), decision_ms=np.array(ms))
            print(f"[{a.suite} t{tid} trial {trial + 1}/{a.trials}] {'OK' if done else 'fail'}  steps={rec['steps']}  "
                  f"decisions={rec['decisions']}  {rec['decision_ms_mean']:.0f} ms/decision", flush=True)
    summ = {"run": run, "profile": pol.p["name"], "suite": a.suite, "episodes": tot_n, "successes": tot_s,
            "sr": tot_s / max(tot_n, 1), "decision_ms_mean_excl_first": float(np.mean(all_ms)) if all_ms else None,
            "s_per_400_actions": (400 / pol.p["exec_steps"] * float(np.mean(all_ms)) / 1000) if all_ms else None}
    json.dump(summ, open(os.path.join(out, "summary.json"), "w"), indent=2); print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
