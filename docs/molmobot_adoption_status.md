# MolmoBot Adoption Status

## End Goal

Determine whether MolmoBot is a credible adoption candidate for the company robot, specifically whether:
- the released MolmoBot methodology or model can transfer zero-shot from simulation to the target environment,
- the released data or training recipe is worth reusing if zero-shot transfer is weak, and
- the expected integration effort is justified relative to the observed performance gap.

The practical decision gate is not just whether MolmoBot runs in simulation. The real question is whether it can produce meaningful zero-shot behavior on the target robot setup, or at least fail in a way that suggests a tractable adaptation path.

## Current Status

### What is already working

- The MolmoBot evaluation environment is installed and reproducible in [`MolmoBot/.venv`](/MolmoBot/MolmoBot/.venv).
- The released checkpoint [`allenai/MolmoBot-DROID`](/MolmoBot/MolmoBot/ckpts/molmobot/MolmoBot-DROID) is downloaded locally.
- Headless MuJoCo rendering is fixed on this host.
- A notebook-equivalent policy sanity check passed.
- A 1-episode built-in MolmoSpaces smoke benchmark passed with `1/1` success.
- A 10-episode pick-only DROID mini subset completed with `7/10` success at horizon `200`.
- Representative rollout videos were copied into [`docs/artifacts/molmobot_feasibility`](/MolmoBot/docs/artifacts/molmobot_feasibility).

### What is not yet proven

- No real-robot zero-shot trial has been run.
- No sim2real claim can be made yet for the company robot or environment.
- Pick-and-place sim benchmarking is still partially blocked by a benchmark/config mismatch.
- JAX is still CPU-only in the current environment, which affects throughput but not the basic validity of the completed sim results.

### Current interpretation

At this point, MolmoBot has cleared the infrastructure and simulation-feasibility bar:
- the checkpoint loads,
- simulation runs end to end,
- benchmark outputs are generated, and
- the model shows non-trivial success on a small pick-focused benchmark slice.

That is enough to say the stack is operational and worth deeper evaluation.

It is not enough to say the method will transfer zero-shot to the target robot. The real sim2real question remains open until hardware-side testing happens.

## Key Artifacts

Primary workflow docs:
- workflow: [`franka_droid_feasibility.md`](/MolmoBot/docs/franka_droid_feasibility.md)
- dated status: [`franka_droid_feasibility_status_20260406.md`](/MolmoBot/docs/franka_droid_feasibility_status_20260406.md)

Representative visuals:
- artifact index: [`README.md`](/MolmoBot/docs/artifacts/molmobot_feasibility/README.md)
- smoke success clip: [`smoke_success_exo.mp4`](/MolmoBot/docs/artifacts/molmobot_feasibility/smoke_success_exo.mp4)
- pick success clip: [`pick_subset_success_exo.mp4`](/MolmoBot/docs/artifacts/molmobot_feasibility/pick_subset_success_exo.mp4)
- pick failure clip: [`pick_subset_failure_exo.mp4`](/MolmoBot/docs/artifacts/molmobot_feasibility/pick_subset_failure_exo.mp4)

## Remaining Steps

### 1. Strengthen the simulation signal

- Run a larger pick-focused benchmark slice so the current `7/10` result is backed by a larger sample.
- Run additional simulation tasks that better match the intended deployment tasks.
- If useful for decision quality, compare `MolmoBot-DROID` against at least one baseline or variant.

Goal of this step:
- reduce uncertainty in the sim-only signal before spending more integration effort.

### 2. Unblock pick-and-place simulation

- Investigate and fix the task-sampling mismatch:
  `Expected max_place_receptacle_pos_displacement=0.15, got 0.05`
- Re-run a pick-and-place subset after the mismatch is resolved.

Goal of this step:
- determine whether MolmoBot remains promising on more realistic manipulation tasks beyond pick-only episodes.

### 3. Decide whether benchmark speed matters enough to change the environment

- Optionally enable CUDA-backed JAX.
- If the environment changes, re-run the notebook-equivalent sanity check and the 1-episode smoke benchmark.

Goal of this step:
- improve throughput without invalidating the current environment assumptions.

### 4. Prepare the real-robot evaluation gate

This is currently blocked by lack of hardware access.

When hardware access becomes available:
- define the target robot embodiment,
- define cameras and observation/action contract,
- collect the robot host and camera identifiers, and
- prepare a short task list for [`run_feasibility_trials.py`](/MolmoBot/robot_eval/scripts/droid/run_feasibility_trials.py).

Goal of this step:
- make the first real trial small, controlled, and attributable.

### 5. Run the real zero-shot trial

- Run the released checkpoint on the target setup without retraining.
- Measure whether any tasks succeed zero-shot.
- Record runtime behavior and failure modes.

Goal of this step:
- answer the actual adoption question instead of the weaker simulation-only question.

### 6. If zero-shot fails, classify the gap before deciding on deeper adoption

Classify whether failure is dominated by:
- perception mismatch,
- camera placement mismatch,
- action semantics or control mismatch,
- timing or latency,
- embodiment mismatch, or
- task distribution mismatch.

Then decide whether the next thing worth evaluating is:
- the released model only,
- the methodology or training recipe,
- the dataset assumptions, or
- a custom adaptation path for the target robot.

Goal of this step:
- avoid committing to retraining or integration work before the real bottleneck is understood.

## Bottom Line

Current evidence supports this conclusion:
- MolmoBot is operational and credible enough in simulation to justify continued evaluation.

Current evidence does not yet support this stronger conclusion:
- MolmoBot will transfer zero-shot to the company robot or target environment.

The biggest remaining milestone is still the real-robot zero-shot gate. Until that is run, adoption feasibility is promising but unproven.
