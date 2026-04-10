#!/bin/bash
# Run batch evaluation on RunPod GPU
#
# Usage:
#   bash scripts/run_batch.sh
#   bash scripts/run_batch.sh --task_horizon_override 10  # quick test

set -e

cd /workspace/MolmoBot/MolmoBot

# Environment
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
source .venv/bin/activate

# Pull latest
git pull

# Generate config
python scripts/run_batch_eval.py --generate-config

# Find checkpoint
CKPT=$(ls -d ~/.cache/huggingface/hub/models--allenai--MolmoBot-DROID/snapshots/*/ 2>/dev/null | head -1)
if [ -z "$CKPT" ]; then
    echo "Checkpoint not found. Downloading..."
    python -c "from huggingface_hub import snapshot_download; print(snapshot_download('allenai/MolmoBot-DROID'))"
    CKPT=$(ls -d ~/.cache/huggingface/hub/models--allenai--MolmoBot-DROID/snapshots/*/ | head -1)
fi
echo "Checkpoint: $CKPT"

# Run
python -u scripts/run_batch_eval.py \
  --checkpoint_path "$CKPT" \
  --config benchmarks/franka_book_pencil_pick_place/batch_config.json \
  "$@"
