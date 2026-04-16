# RBY1 Frozen-Base Salt Shaker Pick Test

## Goal

Get the RBY1 mobile manipulator to successfully pick up a salt shaker in
simulation using the `MolmoBot-RBY1Multitask` checkpoint.

## Current Status: Retraining Required

All inference-time fixes have been exhausted. The physics are validated
(IK grasp lifts the object 6-40cm), but the learned policy cannot replicate
this due to a fundamental workspace limitation.

**Root cause:** The policy learned arm-only manipulation with torso at zero.
The salt shaker is outside the arm-only workspace. Overriding the torso height
at eval time doesn't help because the policy's arm trajectory is coupled to
the torso-at-zero configuration.

## Experiment Results Summary

### Phase 1: Gripper & Scene Debugging (38 commits)

Systematically ruled out scene setup, gripper polarity, camera views, object/
robot pose sweeps, gripper timing variants, and point prompts.

| Experiment | Result | Conclusion |
|-----------|--------|------------|
| Gripper polarity/clamping | Fixed | Was using wrong command convention |
| Close hysteresis (24 steps) | No improvement | Gripper timing not the issue |
| Camera view reordering (5 variants) | All worse | Checkpoint sensitive to training-time order |
| Object/robot pose sweep (20+ variants) | Best midpoint 7.1cm | Improved but still no grasp |
| Point prompts / door-style conditioning | Worse (5cm→10cm) | Pick task doesn't benefit |
| Contact-hold / contact-gate | No grasp | Poor contact geometry |
| Scripted grasp replay (policy waypoints) | No grasp | Replayed misaligned waypoints |

**Best result:** TCP within 7mm, finger midpoint 6-7cm off laterally.

- [Debug Plan](RBY1_PICKPNP_DEBUG_PLAN.md)
- [Execution Summary](RBY1_PICKPNP_EXECUTION_SUMMARY.md)
- [Midpoint Sweep](RBY1_PICKPNP_MIDPOINT_SWEEP_SUMMARY.md)
- [View Ablation](RBY1_PICKPNP_VIEW_ABLATION_SUMMARY.md)
- [Scripted Grasp Sanity](RBY1_SCRIPTED_GRASP_SANITY_SUMMARY.md)

### Phase 2: IK Pick Validation (no CuRobo / no CUDA)

Proved the physics work by computing correct grasp poses via MuJoCo DLS IK
and executing with direct joint control, bypassing the learned policy.

| Test | Result |
|------|--------|
| IK finger alignment (standalone model) | **0.1mm** accuracy |
| Arm-only IK (torso locked at zero) | **FAIL** - object out of workspace |
| Arm-only IK (torso height=0.50-0.60) | **OK** - object reachable |
| IK via eval pipeline (PD controller) | Torso drifts under gravity |
| PD gain boost (10x, 100x) | Still drifts, coupled chain dynamics |
| Direct qpos in full scene (1x friction) | Object pushed 6.8cm, no lift |
| Direct qpos in full scene (5x friction) | **Object lifted 6-40cm** |

**Key findings:**
- EE-to-fingertip offset: 47mm along wrist Z-axis
- Salt shaker collision: single box 2.6x5.5x2.6cm, default friction 0.9
- Gravity compensation (`gravcomp=True`) already enabled in eval pipeline
- Torso "height" mode: maps 1D scalar to 6D joints (h, -2h, h, 0, 0, 0)

- [IK Validation Summary](RBY1_IK_PICK_VALIDATION_SUMMARY.md)

### Phase 3: Policy Eval with Friction Fix (GPU, RTX 4080)

Tested the 5x friction boost with the real policy on the frozen-base benchmark.

| Metric | Pre-friction | With 5x friction |
|--------|-------------|-------------------|
| Best TCP distance | 0.0076m | 0.028m |
| Best midpoint distance | 0.063m | 0.073m |
| Object moved | 0.0 | 0.0 |
| Finger contact | rare, one-sided | traj_0 has contact |
| Success | 0/6 | 0/6 |

**Conclusion:** Friction alone doesn't help because the fingers never reach
the object (6-7cm lateral miss). Friction only matters when fingers contact
the object, which requires solving the workspace issue first.

### Phase 4: Fixed Torso Height Override (GPU, RTX 4080)

Tested overriding the policy's zero torso output with height=0.50 (the
IK-validated reachable height).

| Metric | Torso=0 (default) | Torso=0.50 (fixed) |
|--------|-------------------|---------------------|
| Best TCP distance | 0.028m | 0.065m (worse) |
| Best midpoint distance | 0.073m | **0.072m** (same) |
| Object moved | 0.0 | 0.0 |
| Success | 0/6 | 0/6 |

**Conclusion:** The policy's arm trajectory is adapted to torso-at-zero.
Changing the torso height changes the arm's base frame, but the policy doesn't
adjust its arm commands accordingly. The arm trajectory and torso must be
learned together - a fixed torso override is not sufficient.

## What Was Ruled Out

| Hypothesis | Tested | Result |
|-----------|--------|--------|
| Wrong gripper polarity | Fixed clamp convention | Was an issue, now fixed |
| Gripper closes too early/late | Hysteresis, contact-trigger | Not the issue |
| Wrong camera order | 5 view ablations | Current order is best |
| Object too far / wrong pose | 20+ pose sweeps | Slightly helps, not enough |
| Point prompts needed | Enabled for pick | Makes it worse |
| Simulator can't pick object | IK + direct qpos | Physics work at 5x friction |
| Default friction too low | 5x friction boost | Fixed, but policy still misses |
| Object unreachable | IK with torso height sweep | Reachable at h=0.50-0.60 |
| Just need torso height override | Fixed h=0.50 eval | Policy arm doesn't adapt |
| PD controller too weak | Gains boosted 100x | Coupled chain defeats PD |

## Conclusion

The task requires **coordinated torso + arm motion** that the current
checkpoint has not learned for pick tasks. The checkpoint was trained with
CuRobo-generated demonstrations that include torso movement, but the policy
learned to keep the torso at zero and use arm-only trajectories.

## Next Steps (require retraining)

1. **Fine-tune with torso-aware pick demonstrations**
   - Generate 50-100 CuRobo pick trajectories with explicit torso movement
   - Fine-tune `MolmoBot-RBY1Multitask` on the new data
   - Mix with existing training data to avoid catastrophic forgetting

2. **Investigate training data torso statistics**
   - Check if the CuRobo pick training data actually has torso variation
   - If torso is always near zero in training, the policy correctly learned
     to ignore it

3. **Try 6D torso action instead of 1D height**
   - The height mode constrains the torso to a 1D manifold
   - Full 6D joint control may allow richer torso configurations
   - Requires retraining with new action spec

## How to Run

### IK Physics Validation (no GPU required)

```bash
# Full-scene direct-qpos pick (assembles robot + desk + salt shaker)
.venv/bin/python scripts/run_rby1_direct_scene_pick.py

# Orientation + friction sweep
.venv/bin/python scripts/run_rby1_grasp_sweep.py
```

### Policy Eval (requires CUDA GPU)

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export JAX_PLATFORMS=cpu

CKPT=/mnt/data/ckpts/molmobot/MolmoBot-RBY1Multitask  # adjust path

# Frozen-base with friction fix (baseline)
python launch_scripts/run_eval.py \
  --checkpoint_path $CKPT \
  --benchmark_path benchmarks/rby1_pickpnp_benchmark \
  --eval_config_cls olmo.eval.configure_molmo_spaces:MolmoBotRBY1PickPnPFrozenBaseEvalConfig \
  --task_horizon 400 --output_dir eval_output/rby1_friction_verify --num_workers 1

# Fixed torso height=0.50
python launch_scripts/run_eval.py \
  --checkpoint_path $CKPT \
  --benchmark_path benchmarks/rby1_pickpnp_benchmark \
  --eval_config_cls olmo.eval.configure_molmo_spaces:MolmoBotRBY1PickPnPFixedTorso050FrozenBaseEvalConfig \
  --task_horizon 400 --output_dir eval_output/rby1_fixed_torso_050 --num_workers 1

# Summarize
python scripts/summarize_rby1_eval.py eval_output/rby1_friction_verify
python scripts/summarize_rby1_eval.py eval_output/rby1_fixed_torso_050
```

### Selected Videos

- `videos/non_fixed_base_20260415_003409_exo_camera.mp4` - base active
- `videos/fixed_base_20260415_010245_exo_camera.mp4` - frozen base

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

## Eval Config Reference

| Config | Description |
|--------|------------|
| `MolmoBotRBY1PickPnPEvalConfig` | Default, base active |
| `MolmoBotRBY1PickPnPFrozenBaseEvalConfig` | Frozen base (5x friction) |
| `MolmoBotRBY1PickPnPFixedTorsoFrozenBaseEvalConfig` | Fixed torso h=0.2 |
| `MolmoBotRBY1PickPnPFixedTorso050FrozenBaseEvalConfig` | Fixed torso h=0.5 |
| `MolmoBotRBY1ScriptedGraspSanityEvalConfig` | Scripted waypoint replay |
