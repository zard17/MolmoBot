# Franka/DROID Feasibility Workflow

This workflow is the repo-native path for verifying whether MolmoBot is worth deeper adoption work on a Franka/DROID setup before trying to port the stack elsewhere.

Current status snapshot: [`franka_droid_feasibility_status_20260406.md`](./franka_droid_feasibility_status_20260406.md).

It is intentionally split into:

1. an optional notebook sanity check,
2. a simulation smoke benchmark,
3. a real policy server check, and
4. a small real-robot task suite.

## 0. Optional notebook sanity check

If you want a quick visual confirmation before setting up the benchmark or using robot time, open [`demo_policy.ipynb`](../MolmoBot/demo_policy.ipynb).

Use it to confirm:
- the checkpoint loads,
- the scene/task wiring looks correct, and
- the policy produces plausible actions in simulation.

Do not treat the notebook as the actual feasibility result. The real decision signal should still come from the benchmark smoke test and the Franka/DROID task suite.

## 1. Benchmark smoke test

Install the eval dependencies:

```bash
cd MolmoBot/MolmoBot
uv sync --extra eval
. .venv/bin/activate
```

Run a small MolmoSpaces benchmark slice with the released Franka/DROID checkpoint:

```bash
python launch_scripts/run_feasibility.py benchmark-smoke   --benchmark-path /path/to/molmospaces/benchmark_slice   --output-dir /tmp/molmobot_franka_smoke
```

Defaults:
- checkpoint: `allenai/MolmoBot-DROID`
- eval config: `olmo.eval.configure_molmo_spaces:FrankaState8ClampAbsPosConfig`

Optional flags:
- `--local-path <ckpt_dir>` to skip downloading from Hugging Face
- `--use-filament` if the filament renderer is installed
- `--manifest-path <json_path>` to separate metadata from eval outputs

The command writes a manifest JSON so the exact checkpoint source, benchmark path, and eval parameters are recorded with the run.

## 2. Serve the released real policy

On the inference machine:

```bash
cd MolmoBot/MolmoBot
. .venv/bin/activate
PYTHONPATH=. python launch_scripts/serve_molmo.py --hf-repo allenai/MolmoBot-DROID --action-type joint_pos
```

This keeps the real trial aligned with the released Franka/DROID policy assumptions:
- `exo_camera_1`
- `wrist_camera`
- Franka joint-position execution through the DROID client

## 3. Run a small real-task suite

On the DROID / control side:

```bash
cd MolmoBot/robot_eval
conda activate molmobot
python scripts/droid/run_feasibility_trials.py   --robot-host <nuc_ip>   --wrist-camera-id <wrist_id>   --exo-camera-id <exo_id>   --tasks-file config/feasibility_tasks_franka.txt   --output-dir outputs/feasibility_run_01
```

The suite runner:
- calls the existing `scripts/droid/run_policy.py` once per task,
- stores each trial in a deterministic Hydra output directory,
- writes `suite_manifest.json` before the run, and
- aggregates results into `summary.json` afterward.

Use `--dry-run` first if you want to inspect the generated commands.

## Interpreting the result

Use the smoke benchmark to answer:
- Does the environment run cleanly?
- Does the official checkpoint load and produce non-degenerate rollouts?

Use the real suite to answer:
- Can the released policy achieve any zero-shot success in your environment?
- If not, are the failures dominated by camera placement, action semantics, timing, or task mismatch?

If the sim smoke test is healthy but the real suite fails consistently, treat that as a methodology gap-analysis point before investing in retraining or dataset adaptation.
