# RBY1 PnP Blocker Artifacts

Artifacts from the reusable 3-episode RBY1 pick-and-place validation attempt on 2026-04-07.

Result:
- Eval config: `MolmoBotRBY1PickPnPEvalConfig`
- Normalized benchmark slice: first 3 episodes from released `pnp_benchmark`
- Executed episodes: `0`
- Blocking failure: `Expected max_place_receptacle_pos_displacement=0.15, got 0.1`
- Output dir at run time: `/tmp/molmobot_rby1_pnp_small_20260407/MolmoBotRBY1PickPnPEvalConfig/20260407_013852`

Files:
- `running_log.log`: task-sampling failure log.
- `feasibility_manifest.json`: workflow manifest for the reusable runner invocation.
