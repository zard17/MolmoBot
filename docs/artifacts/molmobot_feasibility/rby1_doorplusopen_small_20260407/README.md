# RBY1 Door Small-Slice Artifacts

Artifacts from the reusable 3-episode RBY1 door validation run on 2026-04-07.

Result:
- Eval config: `MolmoBotRBY1DoorPlusOpenEvalConfig`
- Normalized benchmark slice: first 3 episodes from released `door_opening_benchmark`
- Success rate: `1/3` (`33.3%`)
- Output dir at run time: `/tmp/molmobot_rby1_doorplusopen_small_20260407/MolmoBotRBY1DoorPlusOpenEvalConfig/20260407_011148`

Files:
- `rby1_doorplusopen_success_exo.mp4`: exo/follower clip from the successful episode.
- `rby1_doorplusopen_failure_exo.mp4`: exo/follower clip from a failed episode.
- `running_log.log`: full evaluation log.
- `feasibility_manifest.json`: workflow manifest for the reusable runner invocation.
