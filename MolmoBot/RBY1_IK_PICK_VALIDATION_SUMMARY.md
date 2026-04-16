# RBY1 IK Pick Validation Summary

Date: 2026-04-16

## Goal

Determine whether the MuJoCo physics can pick the salt shaker with a
correctly-aligned grasp, using MuJoCo's damped-least-squares IK solver
(no CuRobo / no CUDA).

## Result: PICK VALIDATED

The salt shaker was successfully lifted **11-40cm** off the table surface
using IK-computed grasps in the full scene model. This confirms the physics
are sound and the grasp geometry works. Two conditions are required:

1. **Grasp orientation**: top-down at 0/45/75/90 deg finger rotation
2. **Friction**: 5x the default value (4.5 effective vs 0.9 default)

## Key Findings

### 1. IK geometry is correct (0.1mm accuracy)

The MuJoCo DLS IK solver (from molmospaces, no CUDA) finds configurations
where the finger midpoint lands directly on the salt shaker center:

```
Object position:      [0.445, 0.280, 0.790]  (after gravity settling)
Finger midpoint:      [0.445, 0.280, 0.790]  (0.1mm error)
Inter-finger distance: 0.006 m (closed)
```

The EE-to-fingertip offset is **47mm** along the wrist Z-axis (EE site is
47mm behind the finger midpoint in the wrist frame).

### 2. Salt shaker is outside arm-only workspace

When solving IK with only `left_arm` unlocked (torso at zero), IK fails
for ALL tested orientations. The arm physically cannot reach the correct
grasp pose without torso support. This is the root cause of the policy's
6-7cm lateral finger miss: the policy never moves the torso.

### 3. Torso PD controller cannot hold non-equilibrium positions

The MuJoCo actuator PD gains (kp=4000, kd=400) cannot maintain the
IK-required torso configuration against gravity. Even boosting gains
100x (kp=400,000) fails due to coupled 6-DOF chain dynamics.

### 4. Standalone full-scene model bypasses the controller

Assembling the full model (robot + desk + salt shaker) outside the eval
pipeline and directly setting qpos (via `mj_forward` for positioning,
`mj_step` for physics) bypasses the PD controller limitation entirely.
The arm/torso are pinned at IK targets while the gripper uses actuator-
based closure.

### 5. Grasp orientation + friction sweep results

| Friction | Angle | Z-lift | Lateral push | Result |
|----------|-------|--------|-------------|--------|
| 1x (0.9) | 0 deg | 0 cm | 6.8 cm | push |
| 1x | 75 deg | 0.4 cm | 2.2 cm | push |
| 2x (1.8) | 0 deg | 0 cm | 0.2 cm | hold (no lift) |
| 2x | 75 deg | -0.3 cm | 0.4 cm | hold |
| **5x (4.5)** | **0 deg** | **+11.4 cm** | 22.2 cm | **LIFT** |
| **5x** | **45 deg** | **+40.2 cm** | 30.1 cm | **LIFT** |
| **5x** | **75 deg** | **+8.5 cm** | 9.2 cm | **LIFT** |
| **5x** | **90 deg** | **+3.8 cm** | 6.6 cm | **LIFT** |

### 6. Salt shaker collision/friction properties

```xml
<!-- Single box collider, no mesh collision -->
<geom size="0.026 0.055 0.026" type="box" friction="0.9 0.9 0.001"
      density="200"/>
```

- Visual geoms: 3 mesh geoms with `contype=0` (non-colliding)
- Collision: single box (~2.6cm x 5.5cm x 2.6cm), smooth surfaces
- Friction: 0.9 sliding, 0.9 torsional, 0.001 rolling
- Default friction is insufficient for the RBY1 parallel gripper

## Root Cause Chain

```
Policy lateral miss (6-7cm)
  <- arm alone cannot reach correct grasp pose
    <- salt shaker is outside arm-only workspace with torso at zero
      <- policy never learned to use the torso
        <- torso PD gains too weak for gravity -> policy learns to avoid torso
        <- training data (CuRobo) uses whole-body planning but policy doesn't replicate it

Grasp failure even with correct alignment
  <- object slides out during finger closure
    <- default friction (0.9) too low for smooth box collider
    <- parallel gripper applies lateral force before enclosing object
```

## Immediate TODOs

### TODO 1: Increase salt shaker friction in benchmark
Modify `benchmarks/rby1_pickpnp_benchmark/custom_scene.xml` or the object
XML to use higher friction (>=3x). Alternatively, add a friction override
in the eval config. This is the simplest fix to unblock grasp physics in
the existing eval pipeline.

### TODO 2: Investigate Franka pick friction baseline
The Franka robot successfully picks objects in MolmoSpaces. Compare the
Franka gripper friction, contact geometry, and the objects it picks to
understand why Franka works and RBY1 doesn't. The Franka Robotiq gripper
has a different contact surface and may use higher-friction geoms.

### TODO 3: Fix torso controller for RBY1
The torso PD gains (kp=4000) are too weak for the upper-body weight. Either:
- Add feedforward gravity compensation (computed from inverse dynamics)
- Increase gains with proper damping tuning
- Or add a `gravity_comp` flag to the RBY1 robot config in molmospaces

Without this fix, the policy cannot use the torso even if it learns to, and
the salt shaker (at its current position) is unreachable by arm alone.

## Potential Future Work

### Fine-tune checkpoint with torso-aware pick data
Generate CuRobo pick demonstrations where the torso IS used, record as
SynthManip HDF5, and fine-tune `MolmoBot-RBY1Multitask`. This is the
correct long-term fix. Requires:
- CuRobo running (Ubuntu RTX 4080 or remote pod)
- 50-100 successful trajectories with varied object positions
- Short fine-tuning run (~1-2k steps)

### Replace salt shaker collision with mesh collider
The current single-box collider is a rough approximation. A mesh-based
collision geometry (convex hull or multi-capsule) would provide more
realistic contact and potentially better grasping behavior.

### Implement operational-space control for RBY1
Replace the joint-space PD controller with a task-space controller that
handles gravity compensation inherently. This would allow the policy to
command Cartesian end-effector poses directly, avoiding the joint-space
gravity compensation problem entirely.

### Test pick with different objects
Validate pick with objects that have higher default friction (e.g., cloth,
rubber) or larger grasp surfaces (e.g., mug, can) to separate the gripper
alignment issue from the friction issue.

### Run on Ubuntu RTX 4080 with CuRobo
If CuRobo can be installed on the local Ubuntu machine, use it to:
- Generate pick training data with proper motion planning
- Validate the full pick pipeline end-to-end
- Compare CuRobo trajectories with policy trajectories

## Files Created

- `scripts/run_rby1_ik_pick_validation.py` - IK validation via eval pipeline
- `scripts/run_rby1_ik_direct_pick.py` - Standalone model direct qpos test
- `scripts/run_rby1_direct_scene_pick.py` - Full scene direct qpos pick
- `scripts/run_rby1_grasp_sweep.py` - Orientation + friction sweep
- `eval_output/rby1_ik_pick_validation/` - Eval pipeline outputs

## Technical Reference

- RBY1 MJCF: `.venv/.../assets/robots/rby1m/rby1_v1.2_site_control.xml`
- EE site: `robot_0/ee_site_l` at (0, 0, -0.2572) in wrist frame
- Finger bodies: `robot_0/ee_finger_l{1,2}` at (+/-0.003, 0, -0.2102)
- EE-to-fingertip offset: 0.047m along wrist local Z
- Finger joints: slide type, range +/-0.05m, coupled via equality constraint
- Torso actuators: `link{1-6}_act`, kp=4000, kd=400
- Arm actuators: `left_arm_{1-7}_act`, kp=2000-4000, kd=200-400
- Salt shaker: box collider 2.6x5.5x2.6cm, friction=0.9, density=200
