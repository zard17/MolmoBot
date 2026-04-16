# RBY1 Pick-and-Place Debug Plan

## Goal

Make the RBY1 salt-shaker pick task work without updating or retraining the model checkpoint.

Current checkpoint:

- `ckpts/molmobot/MolmoBot-RBY1Multitask`
- Frozen-base evaluation uses the same checkpoint and only suppresses base motion.

Current best result:

- Run: `eval_output/rby1_freeze_test/frozen_base/MolmoBotRBY1PickPnPFrozenBaseEvalConfig/20260415_021959`
- Best trajectory: `traj_5`
- Object pose: `[0.445, 0.28, 0.805]`
- Minimum TCP-object distance: `0.0097 m`
- Gripper command at closest point: closed, `+100.0`
- Result: no touch, no held state, no object movement, no success.

This means the model can bring the robot close to the salt shaker, but the simulator/control/contact path is still not producing a grasp.

Latest diagnostic run:

- Run: `eval_output/rby1_freeze_test/frozen_base_debug/MolmoBotRBY1PickPnPFrozenBaseEvalConfig/20260415_042257`
- The debug task now records `left_tcp_pose`, `right_tcp_pose`, and object contact diagnostics in `task_info`.
- The reported generic TCP is the left TCP for this run.
- Right TCP stays far from the object, so this is not a right-hand attempt.
- At the closest step, the salt shaker contacts only the desk.
- Around steps 290-291, one left finger contacts the salt shaker, but the gripper starts reopening immediately after that contact.
- Next active fix: increase close-command hysteresis so the left gripper remains closed after first contact.

Door/open comparison:

- Door/open and pick/PnP share the same RBY1 multitask checkpoint family and the same clamped gripper polarity.
- Door/open runs with object point prompts, and door+open also uses a conditioning image.
- The checkpoint training args show point prompts were enabled for door/open datasets and disabled for RBY1 pick/PnP datasets.
- The diagnostic hypothesis is that door/open works partly because the model receives stronger target localization, and because a constrained door handle can be opened with hook/contact behavior that would not lift a free salt shaker.
- Diagnostic experiments: run separate prompted pick/PnP eval configs. First test point prompts only; if that does not improve the grasp, test door/open-style point prompts plus the conditioning image. Do not replace the default pick/PnP config until the video/contact summary shows an improvement.

Prompted pick/PnP diagnostic results:

- Point-prompt only run: `eval_output/rby1_freeze_test/frozen_base_point_prompt_debug/MolmoBotRBY1PickPnPPointPromptFrozenBaseEvalConfig/20260415_045558`
  - Success: false.
  - Minimum left TCP-object distance: `0.0502 m` at step 39.
  - Left gripper first closes at step 65, after the closest approach.
  - No robot-object contacts, no touch, no held state.
- Door-style point prompt plus conditioning image run: `eval_output/rby1_freeze_test/frozen_base_door_style_debug/MolmoBotRBY1PickPnPDoorStyleFrozenBaseEvalConfig/20260415_050541`
  - Success: false.
  - Minimum left TCP-object distance: `0.1064 m` at step 45.
  - Left gripper first closes at step 192.
  - No robot-object contacts, no touch, no held state.
- Conclusion: borrowing door/open prompting does not explain the working door behavior for this pick task. It makes the pick trajectory worse than the unprompted frozen-base debug run, which reached about `0.01 m` and at least produced one-finger contact. Keep the prompted configs as diagnostics only; continue with geometry/contact fixes for the unprompted pick/PnP policy.

Torso and contact-gating diagnostic results:

- Fixed-torso baseline: `eval_output/rby1_freeze_test/frozen_base_fixed_torso_debug/MolmoBotRBY1PickPnPFixedTorsoFrozenBaseEvalConfig/20260415_053225`
  - Success: false.
  - Minimum left TCP-object distance: `0.0411 m`.
  - Produced repeatable left-finger contacts, but no touch/held state and no object movement.
  - Conclusion: forcing torso height to `0.2` does not improve the grasp; it makes the closest approach worse than the best non-fixed-torso run.
- Fixed-torso contact-close: `eval_output/rby1_freeze_test/frozen_base_fixed_torso_contact_close_debug/MolmoBotRBY1PickPnPFixedTorsoContactCloseFrozenBaseEvalConfig/20260415_055446`
  - Success: false.
  - Left finger contact appears around steps 343-352.
  - Left gripper closes from step 345 onward, so the contact-triggered close does fire.
  - The object still does not move and the grasp sensor never reports touch/held.
  - Conclusion: the issue is not simply that the gripper never closes.
- Fixed-torso contact-hold: `eval_output/rby1_freeze_test/frozen_base_fixed_torso_contact_hold_debug/MolmoBotRBY1PickPnPFixedTorsoContactHoldFrozenBaseEvalConfig/20260415_060425`
  - Success: false.
  - Holding base/arm relative motion during the contact-close window prevents the hand from sweeping farther away, but the close still happens around a poor one-sided contact.
  - Conclusion: motion hold alone is not enough when the contact geometry is already bad.
- Non-fixed-torso contact-hold: `eval_output/rby1_freeze_test/frozen_base_contact_hold_debug/MolmoBotRBY1PickPnPContactHoldFrozenBaseEvalConfig/20260415_061241`
  - Success: false.
  - Minimum left TCP-object distance: `0.0055 m` at step 360.
  - Left gripper is already closed at the closest approach.
  - Contact occurs after the gripper has closed and passed the useful grasp window.
  - Conclusion: the non-fixed policy reaches very close, but gripper timing/geometry is still misaligned.
- Contact-gated open-until-contact run: `eval_output/rby1_freeze_test/frozen_base_contact_gate_debug/MolmoBotRBY1PickPnPContactGateFrozenBaseEvalConfig/20260415_062143`
  - Success: false.
  - Keeping the gripper open until contact changes the future policy state and worsens the approach; minimum distance is `0.0550 m`, with first close at step 369.
  - Conclusion: fully overriding early close commands is too invasive for this checkpoint.

Current conclusion:

- The best non-fixed run can bring the left TCP to within a few millimeters of the salt shaker.
- Contact/gripper wrappers can alter close timing, but none produced object movement or a held state.
- The most likely remaining non-model issue is geometric: the policy path and the RBY1 gripper/object collision geometry are not aligned for a stable two-finger pinch.
- Next recommended work is a measured object pose/orientation adjustment, not more blind gripper postprocessing. Preserve the model's gripper state as much as possible, because forcing the gripper open changes the later trajectory.

## Recommendation: Start With Contact and Fingertip Verification

Before changing poses or adding more action postprocessing, verify what is physically happening near the closest approach.

The first question to answer is:

> When the reported TCP is within about 1 cm of the salt shaker, where are the actual left and right fingertips, and are there any simulator contact pairs involving the salt shaker?

This determines which class of fix is appropriate.

## Phase 1: Instrument the Existing Run

Extend the debug tooling to report both hands and real contact state.

Tasks:

1. Update `scripts/summarize_rby1_eval.py` to report both left and right gripper information.
2. Report per-hand gripper command windows, not only the currently summarized side.
3. Report per-hand grasp/touch/held state if available in the H5.
4. Inspect available H5 keys for fingertip, TCP, gripper site, collision, and contact data.
5. If fingertip/site positions are available, compute minimum object distance for:
   - left TCP
   - right TCP
   - left fingertip sites
   - right fingertip sites
6. If contact data is available, list contact pairs involving the salt shaker near the closest-approach window.
7. Add a small inspection command or script mode that focuses on a time window around the closest step.

Expected output:

- Which hand is actually approaching the object.
- Whether the reported TCP corresponds to the physical grasp point.
- Whether the fingers touch the salt shaker.
- Whether the simulator reports contacts with the object.
- Whether the gripper is open or closed at the relevant time for each hand.

Decision point:

- If fingertips are not near the object, fix pose, hand selection, or TCP interpretation.
- If fingertips touch the object but the object does not move, fix collision, friction, object setup, or grasp-state interpretation.
- If the wrong hand is being summarized, fix the evaluator/debug summary before changing policy behavior.
- If the gripper closes too early or too late, adjust action postprocessing.

## Phase 2: Examine the Best Video With the New Signals

Use the best current output first:

```bash
.venv/bin/python scripts/summarize_rby1_eval.py \
  eval_output/rby1_freeze_test/frozen_base/MolmoBotRBY1PickPnPFrozenBaseEvalConfig/20260415_021959
```

Focus on `traj_5`.

Check the default camera and exo camera around:

- first close step
- closest TCP-object step
- any contact/touch window
- any moment where the hand moves laterally away from the object

Expected result:

- A concrete explanation for why `traj_5` reaches about 1 cm but does not pick.

## Phase 3: Apply the Smallest Non-Model Fix

Choose the fix based on Phase 1 and Phase 2.

### Case A: TCP Is Close but Fingertips Are Offset

Likely cause:

- The summarized TCP is not the true grasp center, or the model approaches with a repeatable offset.

Possible fixes:

- Move the salt shaker slightly toward the actual fingertip path.
- Change object orientation if the gripper approaches a better side.
- Update the debug summary to use the correct hand/site for future sweeps.

Do not keep sweeping blindly; use the measured fingertip/object offset.

### Case B: Fingertips Touch but Object Does Not Move

Likely cause:

- Collision geometry, friction, contact parameters, or object setup.

Possible fixes:

- Inspect salt-shaker collision mesh and scale.
- Increase object friction or contact stability.
- Raise or reposition the object slightly to avoid table/contact interference.
- Compare against the known working Franka setup for the same object.

### Case C: Gripper Timing Is Wrong

Likely cause:

- The model reaches the object but closes too early, too late, or reopens during contact.

Already applied:

- `clamp_gripper=True`
- close hysteresis for RBY1 pick/pnp

Possible additional fixes:

- Increase close hysteresis duration.
- Delay close until the gripper is near the object.
- Hold close once any contact is detected.
- Add a small near-object final approach rule before closing.

Keep these as policy-wrapper changes, not model changes.

### Case D: Body Pose Is Limiting Reach

Observation:

- Frozen-base runs reach better than default in the current tests.
- Torso/waist commands exist but stay near zero in the best run.
- The model is not meaningfully bending the torso.

Possible fixes:

- Start RBY1 from a more favorable base pose.
- Start RBY1 with a fixed torso/waist preset.
- Add a scripted pre-position step before model rollout.

This should be tested after contact/fingertip verification, because current best reach is already close.

### Case E: The Object Is Poorly Matched to the RBY1 Gripper

Likely cause:

- The salt shaker is pickable in another setup, but its geometry or collision setup may not be suitable for this RBY1 gripper path.

Possible fixes:

- Adjust salt-shaker height or scale.
- Use a more contact-friendly orientation.
- Compare with another object known to work with RBY1 specifically.

## Phase 4: Run a Focused Validation

After the chosen fix, run a focused validation before doing a broad sweep.

Recommended first validation:

```bash
NUM_EPISODES=1 TASK_HORIZON=400 bash scripts/run_rby1_freeze_test.sh frozen_base
```

Then summarize and inspect the video.

Success criteria:

- The gripper makes contact with the salt shaker.
- The object moves after contact.
- The object reaches held state or visibly lifts.
- If success is still false, the failure is later in the lift/place phase rather than the initial grasp.

## Phase 5: Broaden Only After Contact Works

Once contact and object movement are confirmed:

1. Run the default and frozen-base configs.
2. Keep `TASK_HORIZON=400`.
3. Keep `NUM_EPISODES=1` unless testing robustness.
4. Add a small pose sweep only around the verified pickable pose.
5. Commit the code changes and useful output videos.

## Notes

- Do not update the model checkpoint for this plan.
- Do not rely only on TCP-object distance; verify fingertip/contact state.
- Exo camera is for debugging only and is not part of the model input.
- `/dev/dri` warnings are not assumed to be the blocker unless execution actually fails.
- The current priority is to explain why the closest run has no touch and no object movement.
