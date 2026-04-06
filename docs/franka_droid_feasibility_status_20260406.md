# Franka/DROID Feasibility Status - 2026-04-06

## Summary

Current status for the Franka/DROID MolmoBot feasibility track:
- MolmoBot eval environment is installed in [`MolmoBot/.venv`](/MolmoBot/MolmoBot/.venv).
- The released checkpoint [`allenai/MolmoBot-DROID`](/MolmoBot/MolmoBot/ckpts/molmobot/MolmoBot-DROID) is downloaded locally.
- Headless MuJoCo rendering is fixed on this host.
- A notebook-equivalent simulation sanity check passed.
- A 1-episode built-in MolmoSpaces smoke run passed with `1/1` success.
- A 10-episode pick-only DROID mini subset passed with `7/10` success at horizon `200`.
- A 10-episode pick-and-place DROID mini subset did not execute because of a benchmark/config mismatch during task sampling.
- Real Franka/DROID trials are deferred until hardware access is available.

## What Was Done

### Environment bootstrap

Installed `uv` and created the repo-native eval environment:

```bash
python -m pip install uv
cd /MolmoBot/MolmoBot
uv sync --extra eval
```

### Rendering fix

MuJoCo originally failed under both `egl` and `osmesa` because the host was missing generic GLVND / OpenGL runtime libraries.

Installed system packages:

```bash
apt-get update
apt-get install -y libegl1 libgl1 libopengl0 libglvnd0 libosmesa6 libosmesa6-dev
```

Preferred backend for this machine:

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
```

### Notebook-equivalent sanity check

Instead of running Jupyter directly, an equivalent script was executed using the same ingredients as [`demo_policy.ipynb`](/MolmoBot/MolmoBot/demo_policy.ipynb):
- build Franka scene
- render exo plus wrist camera views
- load `RealRobotVLAPolicy`
- query actions for several steps

Observed result:
- model loaded successfully
- actions were produced for multiple steps
- terminal result: `sanity_ok`

### Benchmark smoke run

Ran the feasibility smoke launcher against the installed 1-episode benchmark:

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
/MolmoBot/MolmoBot/.venv/bin/python /MolmoBot/MolmoBot/launch_scripts/run_feasibility.py benchmark-smoke \
  --local-path /MolmoBot/MolmoBot/ckpts/molmobot/MolmoBot-DROID \
  --benchmark-path /MolmoBot/MolmoBot/.venv/lib/python3.11/site-packages/assets/benchmarks/molmospaces-bench-v2/procthor-objaverse/FrankaPickandPlaceHardBench/FrankaPickandPlaceHardBench_20260206_json_1ep_benchmark \
  --output-dir /tmp/molmobot_franka_smoke_1ep \
  --task-horizon 600 \
  --notes '1-episode built-in smoke benchmark'
```

Observed result:
- success count: `1`
- total count: `1`
- success rate: `100%`

Task from trajectory:
- `Pick up the yellow handheld gps with antenna and place it in or on the shallow round wooden bowl with grain`

### Sim-only benchmark expansion

#### Pick-only DROID mini subset

Ran a local 10-episode subset of the installed `FrankaPickDroidMiniBench` benchmark:

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
/MolmoBot/MolmoBot/.venv/bin/python /MolmoBot/MolmoBot/launch_scripts/run_feasibility.py benchmark-smoke \
  --local-path /MolmoBot/MolmoBot/ckpts/molmobot/MolmoBot-DROID \
  --benchmark-path /tmp/molmobot_bench_subsets/franka_pick_droidmini_10ep \
  --output-dir /tmp/molmobot_franka_pick10 \
  --task-horizon 200 \
  --notes '10-episode DROID mini pick subset; shortened horizon for sim-only throughput'
```

Observed result:
- completed houses: `7`
- skipped houses: `0`
- success count: `7`
- total count: `10`
- success rate: `70.00%`

Interpretation:
- this is the strongest current sim-only signal beyond the 1-episode smoke test
- the checkpoint runs stably on a larger benchmark slice
- the shortened horizon makes this a throughput-oriented feasibility check, not a paper-faithful score

#### Pick-and-place DROID mini subset

Ran a local 10-episode subset of the installed `FrankaPickandPlaceDroidMiniBench` benchmark:

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
/MolmoBot/MolmoBot/.venv/bin/python /MolmoBot/MolmoBot/launch_scripts/run_feasibility.py benchmark-smoke \
  --local-path /MolmoBot/MolmoBot/ckpts/molmobot/MolmoBot-DROID \
  --benchmark-path /tmp/molmobot_bench_subsets/franka_pnp_droidmini_10ep \
  --output-dir /tmp/molmobot_franka_pnp10 \
  --task-horizon 300 \
  --notes '10-episode DROID mini pick-and-place subset; shortened horizon for sim-only throughput'
```

Observed result:
- completed houses: `0`
- skipped houses: `10`
- success count: `0`
- total count: `0`
- success rate: `0.00%`

Failure mode from [`running_log.log`](/tmp/molmobot_franka_pnp10/FrankaState8ClampAbsPosConfig/20260406_164116/running_log.log#L150):
- `Expected max_place_receptacle_pos_displacement=0.15, got 0.05`

Interpretation:
- this is a benchmark/config compatibility issue during task sampling
- it is not a meaningful MolmoBot policy score
- do not treat this run as evidence for or against sim performance

## Key Outputs

Smoke benchmark outputs:
- manifest: [`feasibility_manifest.json`](/tmp/molmobot_franka_smoke_1ep/feasibility_manifest.json)
- log: [`running_log.log`](/tmp/molmobot_franka_smoke_1ep/FrankaState8ClampAbsPosConfig/20260406_162001/running_log.log)
- trajectory: [`trajectories_batch_1_of_1.h5`](/tmp/molmobot_franka_smoke_1ep/FrankaState8ClampAbsPosConfig/20260406_162001/house_1001/trajectories_batch_1_of_1.h5)
- rollout videos: [`house_1001`](/tmp/molmobot_franka_smoke_1ep/FrankaState8ClampAbsPosConfig/20260406_162001/house_1001)

Pick-only subset outputs:
- manifest: [`feasibility_manifest.json`](/tmp/molmobot_franka_pick10/feasibility_manifest.json)
- log: [`running_log.log`](/tmp/molmobot_franka_pick10/FrankaState8ClampAbsPosConfig/20260406_164652/running_log.log)
- run directory: [`20260406_164652`](/tmp/molmobot_franka_pick10/FrankaState8ClampAbsPosConfig/20260406_164652)

Pick-and-place subset outputs:
- manifest: [`feasibility_manifest.json`](/tmp/molmobot_franka_pnp10/feasibility_manifest.json)
- log: [`running_log.log`](/tmp/molmobot_franka_pnp10/FrankaState8ClampAbsPosConfig/20260406_164116/running_log.log)
- run directory: [`20260406_164116`](/tmp/molmobot_franka_pnp10/FrankaState8ClampAbsPosConfig/20260406_164116)

Workflow docs:
- workflow: [`franka_droid_feasibility.md`](/MolmoBot/docs/franka_droid_feasibility.md)
- benchmark launcher: [`run_feasibility.py`](/MolmoBot/MolmoBot/launch_scripts/run_feasibility.py)
- real-task suite runner: [`run_feasibility_trials.py`](/MolmoBot/robot_eval/scripts/droid/run_feasibility_trials.py)

## Important Instructions For The Team

### Use these env vars for simulation on this machine

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
```

### Treat the notebook as a sanity check only

Use [`demo_policy.ipynb`](/MolmoBot/MolmoBot/demo_policy.ipynb) or an equivalent script to confirm:
- checkpoint loads
- rendering works
- policy returns actions

Do not use the notebook as the adoption decision signal.

### Use the smoke benchmark as the first real validation step

The smoke benchmark is the first meaningful end-to-end test because it exercises:
- benchmark asset loading
- sim rollout
- policy inference
- saved trajectory outputs

### Use the pick-only subset as the current sim-only comparison point

If hardware is unavailable, the current best follow-on benchmark is the local pick-only subset under [`franka_pick_droidmini_10ep`](/tmp/molmobot_bench_subsets/franka_pick_droidmini_10ep).

This gives a faster signal than full built-in suites while still producing a non-trivial aggregate success rate.

### Do not over-interpret the current pick-and-place subset failure

The current pick-and-place subset run failed before rollout because of a benchmark/config mismatch.

Until that mismatch is resolved, use pick-only results for sim-only feasibility tracking.

## Caveats

- The 1-episode smoke run emitted an asset-version warning: the machine is using a newer Objaverse asset package than the benchmark's original reference version.
- This is acceptable for feasibility smoke testing.
- It is not a paper-faithful reproduction condition.
- JAX is currently CPU-only in this environment.
- The pick-only 10-episode result used a shortened horizon of `200`, so it should be interpreted as a throughput-oriented sim feasibility signal.

## CUDA-enabled JAX Check

Current state:

```bash
PYTHONPATH=/MolmoBot/MolmoBot /MolmoBot/MolmoBot/.venv/bin/python - <<'PY'
import jax
print(jax.devices())
PY
```

Observed result:
- only `CpuDevice(id=0)`
- warning says CUDA-enabled `jaxlib` is not installed

What the installed metadata says:
- JAX version is `0.6.2`
- this version expects the `jax-cuda12-plugin` path for NVIDIA GPUs

Dry-run compatibility check:

```bash
uv pip install --python /MolmoBot/MolmoBot/.venv/bin/python --dry-run 'jax-cuda12-plugin[with-cuda]==0.6.2'
```

Dry-run result:
- feasible in this environment
- would install:
  - `jax-cuda12-pjrt==0.6.2`
  - `jax-cuda12-plugin==0.6.2`
  - CUDA support packages
- would also replace `nvidia-cudnn-cu12` with a newer version

Recommendation:
- CUDA-enabled JAX is possible here.
- Do not change it casually right before environment-sensitive validation runs.
- Upgrade only if benchmark speed becomes important enough to justify env churn.
- If changed, re-run the notebook-equivalent sanity check and the 1-episode smoke benchmark before using the env for anything else.

## Recommended Next Step

### TODO: Real Franka/DROID trial

Defer the real-robot step until hardware access is available. When the team has the robot host, camera serials, and a small task list, run [`run_feasibility_trials.py`](/MolmoBot/robot_eval/scripts/droid/run_feasibility_trials.py) and append the results to this status document.

### Current active track: sim-only evaluation

Continue with larger built-in MolmoSpaces pick-focused benchmark slices first. If a pick-and-place sim study is needed, resolve the `max_place_receptacle_pos_displacement` mismatch before treating those runs as meaningful.
