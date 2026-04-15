# RBY1 Scripted Grasp Sanity Summary

This documents the CuRobo-free scripted sanity path for the RBY1 salt-shaker
pick task. The goal is to separate simulator/scene pickability issues from the
learned MolmoBot policy.

## What Was Added

- `MolmoBotRBY1ScriptedGraspSanityEvalConfig`
- `RBY1ScriptedGraspSanityPolicy`
- `scripts/run_rby1_privileged_grasp_sanity.py`

Despite the historical script name, the current implementation does not import
or require CuRobo. It replays fixed RBY1 joint waypoints extracted from the best
previous frozen-policy trajectory, applies small scripted offsets, scores smoke
runs, and then full-runs the best candidate.

## Key Fix

RBY1 gripper commands must use actuator-scale commands in this eval path:

- open: `-100.0`
- close: `100.0`

Using native-looking joint values such as `-0.05` and `0.0` records a command
but does not actually close the gripper through the `JointPosController` path.
After switching to `-100.0` / `100.0`, the measured left gripper distance closes
from about `0.100 m` to about `0.0003 m`.

## How To Run

Focused run for a known variant:

```bash
.venv/bin/python scripts/run_rby1_privileged_grasp_sanity.py \
  --variant local_j5_plus_080 \
  --smoke_horizon 110 \
  --task_horizon 300
```

The runner writes under:

```text
eval_output/rby1_scripted_grasp_sweep_ctrl100_local_arm/
```

Use `--resume` to avoid rerunning completed variant/stage pairs in the same
output root.

## Latest Result

Best local-arm candidate:

- variant: `local_j5_plus_080`
- full H5:
  `eval_output/rby1_scripted_grasp_sweep_ctrl100_local_arm/local_j5_plus_080/scripted/full/MolmoBotRBY1ScriptedGraspSanityEvalConfig/20260415_154806/house_0/trajectories_batch_1_of_1.h5`
- exo video:
  `eval_output/rby1_scripted_grasp_sweep_ctrl100_local_arm/local_j5_plus_080/scripted/full/MolmoBotRBY1ScriptedGraspSanityEvalConfig/20260415_154806/house_0/episode_00000000_exo_camera_1_batch_1_of_1.mp4`
- report:
  `eval_output/rby1_scripted_grasp_sweep_ctrl100_local_arm/scripted_grasp_sanity_report.md`

Metrics:

- `physical_grasp_success=False`
- object delta: `0.0`
- max z lift: `0.0`
- left gripper closes: `left_grip_min_m=0.0003`
- closest left finger midpoint: `0.0632 m`
- no finger contact and no held state

Measured miss vector at closest midpoint:

```text
finger_midpoint - object = [-0.006 m, +0.062 m, +0.009 m]
```

The `local_j5_plus_080` offset improved height alignment but did not fix the
dominant lateral miss. The current evidence points to a persistent grasp
geometry/alignment issue rather than a gripper actuation issue.
