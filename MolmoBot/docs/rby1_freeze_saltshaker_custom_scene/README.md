# RBY1 Frozen-Base Salt Shaker Pick Test

## Goal

Get the RBY1 mobile manipulator to successfully pick up a salt shaker in
simulation using the `MolmoBot-RBY1Multitask` checkpoint, without retraining.

## Current Status

**Physics validated, policy blocked by torso workspace limitation.**

The IK validation proved the simulator CAN pick the salt shaker (lifted 6-40cm)
with correct finger alignment and 5x friction. The learned policy cannot
replicate this because the salt shaker is outside the arm-only reachable
workspace, and the policy never uses the torso.

## Investigation Timeline

### Phase 1: Gripper & Scene Debugging (38 commits)
Systematically ruled out scene setup, gripper polarity, camera views, object/
robot pose sweeps, gripper timing variants, and point prompts. Best result:
TCP within 7mm of object but finger midpoint 6-7cm off laterally.

- [Debug Plan](RBY1_PICKPNP_DEBUG_PLAN.md)
- [Execution Summary](RBY1_PICKPNP_EXECUTION_SUMMARY.md)
- [Midpoint Sweep](RBY1_PICKPNP_MIDPOINT_SWEEP_SUMMARY.md)
- [View Ablation](RBY1_PICKPNP_VIEW_ABLATION_SUMMARY.md)
- [Scripted Grasp Sanity](RBY1_SCRIPTED_GRASP_SANITY_SUMMARY.md)

### Phase 2: IK Pick Validation
Proved the physics work by computing correct grasp poses via MuJoCo IK
(no CuRobo / no CUDA) and executing with direct joint control.

- [IK Validation Summary](RBY1_IK_PICK_VALIDATION_SUMMARY.md)

Key findings:
- IK geometry: 0.1mm finger-to-object accuracy
- Salt shaker is outside arm-only workspace (torso at zero)
- Torso PD controller cannot hold required positions against gravity
- 5x friction required for stable grasp (now applied automatically)

### Selected Videos

- `videos/non_fixed_base_20260415_003409_exo_camera.mp4` - base active, no pick
- `videos/fixed_base_20260415_010245_exo_camera.mp4` - frozen base, no pick

## How to Run

### IK Physics Validation (no GPU required)

Tests whether the simulator physics can pick the object, independent of the
learned policy:

```bash
# Full-scene direct-qpos pick (assembles robot + desk + salt shaker)
.venv/bin/python scripts/run_rby1_direct_scene_pick.py

# Orientation + friction sweep
.venv/bin/python scripts/run_rby1_grasp_sweep.py
```

### Policy Eval with Friction Fix (requires CUDA GPU)

Verifies the 5x friction boost helps the real policy:

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export JAX_PLATFORMS=cpu

# Download checkpoint if needed
python -c "from huggingface_hub import snapshot_download; \
  snapshot_download('allenai/MolmoBot-RBY1Multitask', \
  local_dir='ckpts/molmobot/MolmoBot-RBY1Multitask')"

# Run frozen-base eval
python launch_scripts/run_eval.py \
  --checkpoint_path ckpts/molmobot/MolmoBot-RBY1Multitask \
  --benchmark_path benchmarks/rby1_pickpnp_benchmark \
  --eval_config_cls olmo.eval.configure_molmo_spaces:MolmoBotRBY1PickPnPFrozenBaseEvalConfig \
  --task_horizon 400 \
  --output_dir eval_output/rby1_friction_verify \
  --num_workers 1

# Summarize
python scripts/summarize_rby1_eval.py eval_output/rby1_friction_verify
```

**What to check:** Compare `obj_delta_m` and `left_finger_midpoint_obj_dist_min_m`
against pre-friction runs. The friction fix addresses the object-slides-out issue
but does NOT fix the 6-7cm lateral finger miss from the torso workspace limitation.

## Remaining Blockers

1. **Torso workspace** — salt shaker is outside arm-only IK workspace. Policy
   keeps torso at zero, so the arm cannot reach the correct grasp pose.
2. **Torso controller** — MuJoCo torso PD gains (kp=4000) cannot maintain
   non-equilibrium positions against gravity, even at 100x boost.
3. **Friction** — FIXED. 5x boost applied automatically via
   `MolmoBotRBY1PickPnPPolicyConfig.friction_multiplier`.

## Next Steps

1. Verify friction fix on GPU (command above)
2. Fix torso gravity compensation in molmospaces
3. Investigate why policy doesn't command the torso
4. Fine-tune checkpoint with torso-aware CuRobo demonstrations

## Related Files

| File | Purpose |
|------|---------|
| `scripts/run_rby1_direct_scene_pick.py` | Full-scene direct-qpos pick |
| `scripts/run_rby1_grasp_sweep.py` | Orientation + friction sweep |
| `scripts/run_rby1_ik_pick_validation.py` | IK validation via eval pipeline |
| `scripts/run_rby1_ik_direct_pick.py` | Standalone model IK test |
| `scripts/run_rby1_freeze_test.sh` | RunPod freeze test runner |
| `scripts/run_rby1_adaptive_variants.py` | Geometry sweep runner |
| `scripts/run_rby1_privileged_grasp_sanity.py` | Scripted grasp tests |
| `scripts/run_rby1_midpoint_sweep.py` | Finger midpoint ranking |
| `scripts/run_rby1_view_ablation.py` | Camera view experiments |
| `scripts/summarize_rby1_eval.py` | Trajectory analysis |
| `benchmarks/rby1_pickpnp_benchmark/` | Salt shaker benchmark scene |
| `olmo/eval/configure_molmo_spaces.py` | All RBY1 eval configs |
| `olmo/eval/rby1_debug_pick_task.py` | Debug task with contact recording |
