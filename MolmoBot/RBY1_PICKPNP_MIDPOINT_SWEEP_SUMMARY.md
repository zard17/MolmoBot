# RBY1 Pick/PnP Midpoint Sweep Summary

Date: 2026-04-15

## Goal

The previous RBY1 salt-shaker runs showed that the policy can move the left TCP
near the object while the actual pinch center remains far away. This sweep
therefore ranks by the left finger midpoint, not by TCP distance.

Output root:

```text
eval_output/rby1_midpoint_sweep
```

Runner report:

```text
eval_output/rby1_midpoint_sweep/midpoint_sweep_report.md
eval_output/rby1_midpoint_sweep/midpoint_sweep_report.json
```

## Implementation

Added midpoint-focused metrics to:

```text
scripts/summarize_rby1_eval.py
```

Added midpoint sweep runner:

```text
scripts/run_rby1_midpoint_sweep.py
```

The runner:

- creates one-episode custom benchmarks per variant,
- runs smoke before full eval,
- defaults to 400 steps because useful RBY1 approaches often happen after step
  300,
- appends completed variants to a JSON/Markdown report,
- skips completed variants on `--resume`,
- scores primarily by object movement/held/touch and then by left finger
  midpoint distance.

## Results

All completed variants still failed to move or lift the salt shaker. The sweep
did improve the true pinch-center distance.

| Rank by midpoint | Variant | Object pose change | Object moved | Left midpoint min | Left TCP min | Finger contact |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | `mid_object_x_minus_075_y_plus_010` | `x -0.075 m`, `y +0.010 m` | no | `0.0705 m` | `0.0832 m` | yes |
| 2 | `mid_object_x_minus_075` | `x -0.075 m` | no | `0.0710 m` | `0.0480 m` | yes |
| 3 | `mid_object_x_minus_075_contact_hold` | `x -0.075 m`, contact-hold config | no | `0.0710 m` | `0.0480 m` | yes |
| 4 | `mid_object_x_minus_090` | `x -0.090 m` | no | `0.0734 m` | `0.0739 m` | yes |
| 5 | `mid_object_x_minus_075_y_minus_010` | `x -0.075 m`, `y -0.010 m` | no | `0.0762 m` | `0.0573 m` | yes |
| 6 | `mid_object_x_minus_075_z_plus_030` | `x -0.075 m`, `z +0.030 m` | no | `0.0769 m` | `0.0482 m` | yes |
| 7 | `mid_object_x_minus_075_table_yaw_plus_7p5deg` | `x -0.075 m`, table yaw `+7.5 deg` | no | `0.0787 m` | `0.0190 m` | yes |
| 8 | `mid_object_x_minus_060` | `x -0.060 m` | no | `0.0795 m` | `0.0549 m` | yes |
| 9 | `mid_object_x_minus_075_z_plus_015` | `x -0.075 m`, `z +0.015 m` | no | `0.0809 m` | `0.0585 m` | yes |
| 10 | `mid_object_x_minus_045` | `x -0.045 m` | no | `0.0842 m` | `0.0538 m` | yes |
| 11 | `mid_object_x_minus_030` | `x -0.030 m` | no | `0.0852 m` | `0.0404 m` | yes |
| 12 | `mid_table_yaw_plus_7p5deg` | table yaw `+7.5 deg` | no | `0.0908 m` | `0.0115 m` | yes |
| 13 | `mid_table_yaw_plus_10deg` | table yaw `+10 deg` | no | `0.0920 m` | `0.0286 m` | yes |
| 14 | `mid_object_z_plus_030` | `z +0.030 m` | no | `0.0961 m` | `0.0269 m` | yes |
| 15 | `mid_table_yaw_plus_5deg` | table yaw `+5 deg` | no | `0.1183 m` | `0.0809 m` | no |
| 16 | `mid_baseline_reference` | none | no | `0.1203 m` | `0.0670 m` | no |

## Takeaways

The x-minus object shift is the strongest non-model intervention found so far.
Moving the object about `7.5 cm` closer along x reduced the pinch-center miss
from roughly `9.5-12 cm` to about `7.1 cm`. Adding `y +0.010 m` gives the best
recorded midpoint value, but the improvement is only about `0.5 mm` and still
does not produce a true grasp or object displacement.

This confirms the failure is still primarily tool-frame/pinch-center alignment,
not policy input view selection. The robot can make one-finger contact, but the
object is not entering a stable two-finger grasp.

## Next Steps

Recommended next tests:

1. Run a scripted or teleop-style privileged grasp sanity check. That will
   separate policy alignment failure from simulator/object/gripper physics
   failure.
2. If the privileged grasp works, test a small family around the weak new lead:
   `x -0.075 m`, `y +0.010 m`, with smaller lateral deltas such as
   `y +0.005 m` and `y +0.015 m`.
3. Avoid repeating these failed branches unless the policy or gripper control
   changes: z-lift, table-yaw combination, and contact-hold did not improve the
   outcome.
