# Input contract

`SmolWorldPolicy.step(obs)` takes one observation per **control step** (every step, also while queued actions
are executing: the model conditions on a 9-step history). `reset(instruction)` starts an episode.

| Key | Shape / type | Meaning | Training convention |
|---|---|---|---|
| `rgb` | list of 2 arrays `(H, W, 3)` uint8 | `[third-person, wrist]` cameras | LIBERO `agentview` and `robot0_eye_in_hand`, resized to 256x256 internally (bilinear) |
| `rgb_rot180` | bool (optional, default False) | rotate RGB by 180 deg before use | `True` for raw LIBERO env renders (the training data is stored rotated); `False` for images already in dataset orientation |
| `depth_mm` | list of 2 arrays `(H, W)` uint16, millimetres | metric depth aligned pixel-for-pixel with `rgb` **after** rotation | resized to 128x128 (nearest) and unprojected to a pointmap with the camera intrinsics; valid range 0.01-2.0 m |
| `proprio` | `(8,)` float32 | `[eef_x, eef_y, eef_z, axis-angle(3), gripper_qpos(2)]` | robosuite `robot0_eef_pos`, `quat2axisangle(robot0_eef_quat)`, `robot0_gripper_qpos` |

Camera intrinsics: `model/camera_intrinsics_libero.json` (identical for all four LIBERO suites):

| Camera key | fx = fy | cx, cy | width x height |
|---|---|---|---|
| `cam_high` (agentview) | 618.04 | 256, 256 | 512 x 512 |
| `cam_left_wrist` (eye-in-hand) | 333.63 | 256, 256 | 512 x 512 |

Pass your own JSON (same keys) as `SmolWorldPolicy(intrinsics=...)` for other cameras; K is rescaled per axis to the
128x128 pointmap grid.

## Example

`example_episode.npz` (made by `scripts/make_example_input.py`): 30 real LIBERO observations, libero_goal task 0
("open the middle drawer of the cabinet"), init state 0, after 10 settle steps.

```python
ex = np.load("inputs/example_episode.npz")
obs = {"rgb": list(ex["rgb"][t]), "depth_mm": list(ex["depth_mm"][t]), "proprio": ex["proprio"][t], "rgb_rot180": True}
```
Arrays: `rgb (30, 2, 256, 256, 3) uint8`, `depth_mm (30, 2, 256, 256) uint16`, `proprio (30, 8) float32`.

## Real-robot checklist

The model was trained only on LIBERO simulation (Franka Panda, robosuite OSC_POSE delta control, two fixed cameras).
Before running on hardware:
1. **Camera placement and intrinsics**: the third-person and wrist views must resemble LIBERO's framing; write their
   intrinsics in the JSON format above.
2. **Depth**: metric, millimetres, aligned to RGB, same orientation as RGB.
3. **Proprio frame and units**: metres and axis-angle radians in the robot base frame used by robosuite;
   gripper qpos as two finger joint positions.
4. **Action interface**: see `outputs/README.md`; actions are normalized OSC_POSE deltas, not joint targets.
5. **Settle**: the eval runs 10 no-op steps after reset before the first `step()`.
6. **Expect a sim-to-real gap**: no real-robot data was used in training; validate on the target setup first.
