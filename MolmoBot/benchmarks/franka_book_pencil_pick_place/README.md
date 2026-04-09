# Franka Pick-and-Place Benchmark

Custom scene benchmark with a desk, bookcase, and Franka robot.

## Tasks

| Episode | Task | Pickup | Target |
|---------|------|--------|--------|
| 0 | Tissue box → bookcase | Tissue_Box_1 on desk | Bookcase shelf |
| 1 | Pencil → cup | Pencil_1 on desk | Cup_5 on desk |

## Setup

```bash
# Clone and install (from MolmoBot/ directory)
pip install -e .

# Download checkpoint
python -c "from huggingface_hub import snapshot_download; snapshot_download('allenai/MolmoBot-DROID')"
```

## Quick Start: Run with Real-Time Viewer

```bash
# Linux (X11/EGL) — shows interactive MuJoCo viewer
MUJOCO_GL=egl python scripts/run_benchmark_with_viewer.py \
  --checkpoint_path ~/.cache/huggingface/hub/models--allenai--MolmoBot-DROID/snapshots/*/

# macOS — requires mjpython for viewer
mjpython scripts/run_benchmark_with_viewer.py \
  --checkpoint_path ~/.cache/huggingface/hub/models--allenai--MolmoBot-DROID/snapshots/*/

# Without viewer (saves video only)
python scripts/run_benchmark_with_viewer.py \
  --checkpoint_path ~/.cache/huggingface/hub/models--allenai--MolmoBot-DROID/snapshots/*/ \
  --no-viewer
```

Options:
- `--task_horizon 200` — max steps per episode (default: 200)
- `--episode 0` — which task to run: 0=tissue_box, 1=pencil (default: 0)
- `--output_dir <path>` — where to save video

## Full Evaluation (run_eval.py)

Runs both episodes and reports success rate.

```bash
# Generate benchmark JSON + custom scene
python scripts/create_book_pencil_benchmark.py

# Run evaluation
python launch_scripts/run_eval.py \
  --checkpoint_path ~/.cache/huggingface/hub/models--allenai--MolmoBot-DROID/snapshots/*/ \
  --benchmark_path benchmarks/franka_book_pencil_pick_place \
  --eval_config_cls olmo.eval.configure_molmo_spaces:FrankaCustomSceneEvalConfig \
  --task_horizon 600
```

Results are saved to `eval_output/FrankaCustomSceneEvalConfig/<timestamp>/house_0/`:
- `episode_*_exo_camera_1_*.mp4` — exocentric camera video
- `episode_*_wrist_camera_*.mp4` — wrist camera video
- `trajectories_*.h5` — trajectory data

## GPU Setup (for faster inference)

### Fix GPU OOM on 24GB cards

If you hit OOM errors, edit `olmo/models/molmobot/inference_wrapper.py` and remove the `to_empty` line:

```python
# Before (lines 128-136):
with torch.device("meta"):
    self.model = self.model_config.build_model()
if self.use_bfloat16:
    self.model.to(torch.bfloat16)
self.model.to_empty(device=self.device)   # <-- REMOVE this line
load_model_state(self.checkpoint_path, self.model)
self.model.to(self.device)

# After:
with torch.device("meta"):
    self.model = self.model_config.build_model()
if self.use_bfloat16:
    self.model.to(torch.bfloat16)
load_model_state(self.checkpoint_path, self.model)
self.model.to(self.device)
```

This avoids allocating the model twice in GPU memory (empty tensors + loaded weights).

### GPU evaluation

```bash
# With viewer
python scripts/run_benchmark_with_viewer.py \
  --checkpoint_path <path> \
  --task_horizon 600

# Full eval (both episodes)
python launch_scripts/run_eval.py \
  --checkpoint_path <path> \
  --benchmark_path benchmarks/franka_book_pencil_pick_place \
  --eval_config_cls olmo.eval.configure_molmo_spaces:FrankaCustomSceneEvalConfig \
  --task_horizon 600
```

bfloat16 is enabled by default in `SynthManipMolmoInferenceWrapper`. No extra flags needed.

## View Scene (no policy)

```bash
# Render preview images
python scripts/view_benchmark_scene.py --preview

# Interactive viewer (Linux)
python scripts/view_benchmark_scene.py

# Interactive viewer (macOS)
python scripts/view_benchmark_scene.py  # uses mujoco.viewer.launch (blocking)
```

## Files

| File | Description |
|------|-------------|
| `scripts/create_book_pencil_benchmark.py` | Generates benchmark JSON + custom scene XML |
| `scripts/view_benchmark_scene.py` | Scene preview and interactive viewer |
| `scripts/run_benchmark_with_viewer.py` | Eval with real-time MuJoCo viewer |
| `benchmarks/.../benchmark.json` | Episode specifications |
| `benchmarks/.../custom_scene.xml` | Scene XML (ground + desk + bookcase) |
| `benchmarks/.../desk.xml` | Desk primitive XML |
| `benchmarks/.../bookcase.xml` | Bookcase primitive XML |
| `olmo/eval/configure_molmo_spaces.py` | `FrankaCustomSceneEvalConfig` eval config |
