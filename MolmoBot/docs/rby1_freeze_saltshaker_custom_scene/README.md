# RBY1 Salt-Shaker Freeze-Test Videos

These are the best qualitative fixed-base and non-fixed-base videos from the
custom salt-shaker scene as of 2026-04-16. None of these runs successfully picks
or lifts the object; they are selected as the clearest comparison clips.

## Selected Clips

### Non-Fixed Base

- Exo camera: `videos/non_fixed_base_20260415_003409_exo_camera.mp4`
- Head camera: `videos/non_fixed_base_20260415_003409_head_camera.mp4`
- Source run:
  `eval_output/rby1_freeze_test/default/MolmoBotRBY1PickPnPEvalConfig/20260415_003409/house_0`
- Reason selected: full 401-step run with the largest observed mobile-base
  motion among the checked non-fixed-base candidates.
- Metrics:
  - `success_any=False`
  - `base_end_delta=0.5900`
  - `base_xy_span=0.7393`
  - `obj_z_delta=0.0`

### Fixed Base

- Exo camera: `videos/fixed_base_20260415_010245_exo_camera.mp4`
- Head camera: `videos/fixed_base_20260415_010245_head_camera.mp4`
- Source run:
  `eval_output/rby1_freeze_test/frozen_base/MolmoBotRBY1PickPnPFrozenBaseEvalConfig/20260415_010245/house_0`
- Reason selected: full 401-step run with complete trajectory/log/video outputs
  and a clean exocentric view for comparison against the non-fixed-base run.
- Metrics:
  - `success_any=False`
  - `success_rate=0.00%`
  - `obj_z_delta=0.0`

## Notes

The exocentric clips are preferred for side-by-side qualitative review. The head
camera clips are included for policy-view context.
