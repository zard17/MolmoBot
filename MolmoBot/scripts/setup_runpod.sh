#!/bin/bash
# RunPod A6000/A40/A100 setup script for MolmoBot benchmark evaluation
#
# Usage:
#   curl -sSL https://raw.githubusercontent.com/zard17/MolmoBot/feat/interactive-object-swap/MolmoBot/scripts/setup_runpod.sh | bash
#
# Or manually:
#   bash scripts/setup_runpod.sh

set -e

echo "=== MolmoBot Benchmark Setup for RunPod ==="

# 1. Install Python 3.11 if needed
PYTHON_CMD="python3.11"
if ! command -v $PYTHON_CMD &> /dev/null; then
    echo "Installing Python 3.11..."
    apt-get update && apt-get install -y software-properties-common
    add-apt-repository -y ppa:deadsnakes/ppa
    apt-get update && apt-get install -y python3.11 python3.11-venv python3.11-dev
fi
echo "Python: $($PYTHON_CMD --version)"

# 2. Install system deps for MuJoCo rendering
echo "Installing system dependencies..."
apt-get install -y libegl1 libgl1 libgles2 libglfw3 libosmesa6 libgl1-mesa-dri mesa-utils 2>/dev/null || true

# 3. Clone repo
cd /workspace
if [ ! -d "MolmoBot" ]; then
    echo "Cloning repo..."
    git clone https://github.com/zard17/MolmoBot.git
fi
cd MolmoBot/MolmoBot

# 4. Switch to correct branch
git fetch origin
git checkout feat/interactive-object-swap
git pull

# 5. Create venv and install
if [ ! -d ".venv" ]; then
    echo "Creating venv..."
    $PYTHON_CMD -m venv .venv
fi
source .venv/bin/activate

echo "Installing dependencies (this takes ~3 minutes)..."
pip install --upgrade pip
pip install -e .

# 6. Download checkpoint
echo "Downloading MolmoBot-DROID checkpoint..."
python -c "from huggingface_hub import snapshot_download; print(snapshot_download('allenai/MolmoBot-DROID'))"

# 7. Set environment
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

# 8. Quick test
echo ""
echo "=== Setup complete! ==="
echo ""
echo "Activate env:  source /workspace/MolmoBot/MolmoBot/.venv/bin/activate"
echo "Set rendering: export MUJOCO_GL=egl && export PYOPENGL_PLATFORM=egl"
echo ""
echo "Run batch eval:"
echo "  cd /workspace/MolmoBot/MolmoBot"
echo "  python scripts/run_batch_eval.py --generate-config"
echo '  python scripts/run_batch_eval.py \'
echo '    --checkpoint_path ~/.cache/huggingface/hub/models--allenai--MolmoBot-DROID/snapshots/*/ \'
echo '    --config benchmarks/franka_book_pencil_pick_place/batch_config.json'
echo ""
echo "Run single episode with viewer:"
echo '  python scripts/run_benchmark_with_viewer.py \'
echo '    --checkpoint_path ~/.cache/huggingface/hub/models--allenai--MolmoBot-DROID/snapshots/*/ \'
echo '    --pickup Mug_1 --receptacle Bowl_3 --no-viewer'
