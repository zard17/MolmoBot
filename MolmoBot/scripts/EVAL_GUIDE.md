# MolmoBot Batch Evaluation Guide

Batch evaluation for MolmoBot-DROID: run multiple object combinations and generate a generalization report.

## Quick Start

```bash
# 1. Install dependencies
cd MolmoBot/MolmoBot
uv sync --extra eval
source .venv/bin/activate

# 2. Download checkpoint
python -c "from huggingface_hub import snapshot_download; print(snapshot_download('allenai/MolmoBot-DROID'))"

# 3. Run batch evaluation (uses built-in default config)
python scripts/run_batch_eval.py --checkpoint_path <path>

# Quick pipeline test (10 steps per episode)
python scripts/run_batch_eval.py --checkpoint_path <path> --task_horizon_override 10
```

---

## Batch Evaluation

### Default Config (No `--config` Needed)

When `--config` is omitted, the script uses a built-in default configuration that tests:

| Group | Scene | Objects | Description |
|-------|-------|---------|-------------|
| **A** | ProcTHOR | Thor | Baseline (closest to training distribution) |
| **B** | ProcTHOR | Objaverse | Object generalization |
| **C** | Custom | Thor | Scene generalization |
| **D** | Custom | Objaverse | Scene + object generalization |
| **E** | ProcTHOR | Thor | Position robustness (4 variations) |
| **F** | Custom | Thor | Position robustness (4 variations) |

### Custom Config

```bash
# Generate config file for customization
python scripts/run_batch_eval.py --generate-config
# Edit benchmarks/franka_book_pencil_pick_place/batch_config.json as needed

# Run with custom config
python scripts/run_batch_eval.py \
  --checkpoint_path <path> \
  --config benchmarks/franka_book_pencil_pick_place/batch_config.json
```

Config format (`batch_config.json`):
```json
{
  "task_horizon": 600,
  "repeats": 3,
  "tasks": [
    {"scene": "procthor", "pickup": "Mug_1", "receptacle": "Bowl_3"},
    {"scene": "custom", "pickup": "Apple_1", "receptacle": "objaverse:45bb173c..."},
    {"scene": "custom", "pickup": "Egg_1", "receptacle": "bookcase"}
  ]
}
```

### Resume & Partial Runs

```bash
# Resume from last interrupted run
python scripts/run_batch_eval.py --checkpoint_path <path> --config batch_config.json --resume

# Run only episodes 5-10
python scripts/run_batch_eval.py --checkpoint_path <path> --config batch_config.json --start 5 --end 10
```

### Output

Results are saved to `benchmarks/franka_book_pencil_pick_place/batch_results_<timestamp>/`:

| File | Description |
|------|-------------|
| `report_<timestamp>.md` | Generalization report with success rates per group |
| `results_<timestamp>.csv` | Per-episode data for analysis |
| `results.json` | Complete results |
| `results_partial.json` | Checkpoint for resuming interrupted runs |
| `ep*_*.mp4` | Per-episode videos |

---

## SPACE GPU Server Setup

SPACE has R2 downloads blocked, no display, and requires HuggingFace auth for model/asset downloads.

### First-time Setup

```bash
cd MolmoBot/MolmoBot
uv sync --extra eval
source .venv/bin/activate

# HuggingFace authentication (required — R2 is blocked)
huggingface-cli login

# Download checkpoint
python -c "from huggingface_hub import snapshot_download; print(snapshot_download('allenai/MolmoBot-DROID'))"
```

### Environment Variables

Add to your `.bashrc` or set before each run:

```bash
export MLSPACES_USE_HF=1          # Use HuggingFace instead of R2 for assets
export MUJOCO_GL=egl              # Headless OpenGL rendering (no display)
export PYOPENGL_PLATFORM=egl      # Same for PyOpenGL
```

### Run Batch Evaluation

```bash
source .venv/bin/activate
python scripts/run_batch_eval.py --checkpoint_path <path>

# Quick pipeline test (10 steps per episode)
python scripts/run_batch_eval.py --checkpoint_path <path> --task_horizon_override 10
```

### RunPod Alternative

For RunPod GPU execution, `run_batch.sh` handles all setup automatically:

```bash
bash scripts/run_batch.sh
bash scripts/run_batch.sh --task_horizon_override 10  # quick test
```

---

## Local Machine with MuJoCo Viewer

For team members with a local GPU (20GB+ VRAM) who want real-time visualization:

### Option 1: run_eval_with_glfw.py (single benchmark + viewer)

```bash
# Linux
python run_eval_with_glfw.py \
    --checkpoint_path <path> \
    --benchmark_path ./benchmarks/minimal_benchmark \
    --eval_config_cls olmo.eval.configure_molmo_spaces:FrankaState8ClampAbsPosConfig \
    --task_horizon 600 \
    --output_dir ./eval_results \
    --enable_viewer

# macOS (requires mjpython)
mjpython run_eval_with_glfw.py --checkpoint_path <path> ... --enable_viewer
```

### Option 2: serve_molmo.py + robot_simulation_client.py (remote GPU + local viewer)

Run inference on a GPU server, view simulation locally. Best when the GPU is on a remote machine (e.g., SPACE) and you want the viewer on your laptop.

```bash
# Terminal 1 — GPU server (e.g., SPACE)
python launch_scripts/serve_molmo.py --hf-repo allenai/MolmoBot-DROID --action-type joint_pos

# Terminal 2 — local machine (with display)
python robot_simulation_client.py --enable-viewer --task "pick up the book"
```

> **Note**: `robot_simulation_client.py` will be added in a future PR. The upstream `serve_molmo.py` is already available.

---

## Environment Requirements

| Resource | Requirement |
|----------|-------------|
| **GPU** | 20GB+ VRAM recommended (model needs ~14.67GB) |
| **CPU fallback** | 32GB+ RAM (very slow, ~30-60 min per episode) |
| **Disk** | 10GB+ for asset cache |

### CPU Execution (when GPU is insufficient)

```bash
export JAX_PLATFORMS=cpu
export CUDA_VISIBLE_DEVICES=""
python scripts/run_batch_eval.py --checkpoint_path <path>
```

---

## Asset Cache

Assets are cached at `~/.cache/molmo-spaces-resources/`:

```
~/.cache/molmo-spaces-resources/
├── robots/     # Robot URDFs (Franka, RBY1, etc.)
├── objects/    # Object meshes (Thor, Objaverse)
└── scenes/     # Scene XMLs (iTHOR, ProcTHOR)
```

Assets are downloaded automatically on first run. Download size: ~2-3GB, time: 10-30 min depending on network.

---

## Troubleshooting

### CUDA Out of Memory

MolmoBot-DROID requires ~14.67GB VRAM. If your GPU has insufficient memory:

```bash
export JAX_PLATFORMS=cpu
export CUDA_VISIBLE_DEVICES=""
```

### Cache Manifest Error

```
RuntimeError: Directory path exists on disk but is not recorded in the cache manifest
```

Delete corrupted cache entries and re-run:

```bash
rm -rf ~/.cache/molmo-spaces-resources/objects/thor/20251117
rm -rf ~/.cache/molmo-spaces-resources/objects/objathor_metadata/20260129
rm -rf ~/.cache/molmo-spaces-resources/scenes/thor/20251117
```

If the error persists, rebuild the manifest:

```python
import json

manifest = {
  "robots": {
    "rby1": ["20251224"], "rby1m": ["20251224"],
    "franka_droid": ["20260127"], "franka_cap": ["20260213"],
    "floating_rum": ["20251110"], "floating_robotiq": ["20260208_retry4"],
    "franka_fr3": ["20260303"]
  },
  "scenes": {
    "ithor": ["20251217"], "refs": ["20250923"],
    "procthor-10k-train": ["20251122"], "procthor-10k-val": ["20251217"],
    "procthor-10k-test": ["20251121"],
    "holodeck-objaverse-train": ["20251217"], "holodeck-objaverse-val": ["20251217"],
    "procthor-objaverse-train": ["20251205"], "procthor-objaverse-val": ["20251205"]
  },
  "objects": {
    "thor": ["20251117"], "objathor_metadata": ["20260129"]
  }
}

manifest_path = "~/.cache/molmo-spaces-resources/mjthor_data_type_to_source_to_versions.json"
import os
with open(os.path.expanduser(manifest_path), "w") as f:
    json.dump(manifest, f, indent=2)
print("Manifest updated")
```

### Texture Path Error (PillowF_AO.png)

```
ValueError: Error opening file '../../objects/thor/Textures/PillowF_AO.png'
```

Scene XMLs reference `objects/thor/Textures/` but assets are at `objects/thor/20251117/Textures/`. Fix with symlink:

```bash
cd ~/.cache/molmo-spaces-resources/objects/thor
ln -s 20251117/Textures Textures
```

If specific texture files are missing, create placeholders:

```bash
cd ~/.cache/molmo-spaces-resources/objects/thor/20251117/Textures
python3 -c "from PIL import Image; img = Image.new('RGB', (64, 64), color=(255, 255, 255)); img.save('PillowF_AO.png')"
```

### warp Module Import Warning

```
Failed to import warp: No module named 'warp'
Failed to import mujoco_warp: No module named 'warp'
```

Safe to ignore. `warp` is an optional GPU optimization module.

---

## File Reference

| File | Description |
|------|-------------|
| `scripts/run_batch_eval.py` | Batch evaluation with generalization report (headless) |
| `scripts/run_batch.sh` | RunPod GPU execution wrapper |
| `run_eval_with_glfw.py` | Single benchmark with optional MuJoCo viewer |
| `launch_scripts/run_eval.py` | Upstream single-benchmark evaluation (with molmo_spaces) |
| `launch_scripts/serve_molmo.py` | Upstream WebSocket policy server (for remote GPU) |
| `robot_simulation_client.py` | MuJoCo simulation client with viewer (connects to serve_molmo.py) |
