# RBY1 Pick/PnP Execution Summary

Date: 2026-04-15

## Scope

We debugged the RBY1 multitask pick/pick-and-place eval on the custom
salt-shaker scene. The main target was to determine whether failures were due
to simulator setup, object placement, gripper command mapping, robot geometry,
or policy inputs.

## Implemented Changes

- Added a custom RBY1 pick/pnp benchmark scene using the salt shaker object.
- Added an exocentric camera for debugging video output.
- Set the pick/pnp gripper path to clamp model gripper outputs into the RBY1
  simulator command convention.
- Added RBY1 pick/pnp trajectory summarization, including TCP-object distance,
  contact, touch/hold, object motion, and output video paths.
- Added `scripts/run_rby1_adaptive_variants.py` to run one-episode geometry
  variants with smoke checks, 400-step full evals, resumable reports, bounded
  follow-up generation, and failed-setting tracking.

## Execution Results

Adaptive run output root:

```text
eval_output/rby1_adaptive_geometry_search_400
```

Report files:

```text
eval_output/rby1_adaptive_geometry_search_400/adaptive_report.md
eval_output/rby1_adaptive_geometry_search_400/adaptive_report.json
```

The full geometry sweep completed. No variant succeeded and no variant moved
the salt shaker:

```text
best object motion: 0.0 m
best task success: false
```

The best completed result remained the frozen baseline:

```text
variant: baseline_best_pose
min left TCP-object distance: 0.007628 m
object movement: 0.0 m
contact: true
success: false
```

Qualitative video/trajectory inspection showed:

- RBY1 loads correctly.
- The salt shaker is present, visible, on the table, and physically reachable.
- The robot attempts the pick.
- The gripper passes very close to the object.
- Contact happens after the closest point and is one-sided, mostly left-finger
  side contact.
- There is no stable two-finger pinch, no held state, and no object lift.

## Geometry Variants Tried

The run covered baseline-local z shifts, lateral and approach object shifts,
robot yaw/base offsets, object yaw changes, table yaw changes, default/base
active confirmation, and contact-hold confirmation.

Notable outcomes:

- `baseline_best_pose_contact_z_plus0.010`: close but worse than baseline,
  no object motion.
- `baseline_best_pose_contact_z_minus0.005`: much worse than baseline,
  no object motion.
- `baseline_best_pose_default_confirm`: much worse, suggesting unfreezing the
  mobile base/torso does not fix this pose.
- `baseline_best_pose_contact_hold`: matched baseline, still no object motion.
- `object_y_plus_015`: kept the gripper open during the near pass but worsened
  lateral alignment.
- `object_x_minus_015`: made the encounter earlier but more side-offset, with
  the gripper close timing late relative to contact.
- `robot_y_plus_030_toward_tcp_0.5`: secondary close result, but still worse
  than baseline and no object motion.

## Current Interpretation

The failure is unlikely to be caused by:

- Missing object or bad scene setup.
- Unpickable salt shaker placement.
- `/dev/dri` rendering warnings.
- Simple gripper polarity.
- Simple table/robot/object geometry.

The current failure mode is best described as a policy final-alignment problem:
the hand reaches near the object, but the visual/action trajectory does not
center the object inside the effective pinch path.

## Next Investigation

The next high-leverage investigation is policy input view selection and target
visibility.

The checkpoint config uses:

```text
model.mm_preprocessor.image.max_images: 5
```

The active RBY1 pick/pnp policy currently feeds these three camera views:

```text
wrist_camera_r
head_camera
wrist_camera_l
```

`exo_camera_1` is recorded for debugging but is not fed to the policy.

Questions to answer next:

- Which views are actually passed into `agent.get_action_chunk` at inference?
- How large and centered is the salt shaker in each policy input view around
  the key approach window?
- Is the left-arm pick being diluted by a right-wrist-first camera ordering?
- Does a left-focused camera set improve final alignment?

Recommended next runs:

```text
current:        wrist_camera_r, head_camera, wrist_camera_l
left-focused:   head_camera, wrist_camera_l
left-first:     wrist_camera_l, head_camera
left-only diag: wrist_camera_l
```

Do not feed `exo_camera_1` into the policy as the first fix. It is valuable for
debugging videos, but it is not part of the current RBY1 training camera preset.
