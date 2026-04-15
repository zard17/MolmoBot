#!/bin/bash
# RBY1 Pick&Place freeze test on RunPod GPU
#
# Runs two conditions on the same benchmark:
#   1) Default — mobile base active (as trained)
#   2) Frozen  — mobile base locked (base actions zeroed)
#
# Usage:
#   bash scripts/run_rby1_freeze_test.sh
#   bash scripts/run_rby1_freeze_test.sh --use_filament            # filament renderer
#   bash scripts/run_rby1_freeze_test.sh --task_horizon 200        # shorter episodes
#   bash scripts/run_rby1_freeze_test.sh --checkpoint_path /path   # custom checkpoint
#   bash scripts/run_rby1_freeze_test.sh --num_episodes 3          # fewer episodes
#   bash scripts/run_rby1_freeze_test.sh --only default            # run only one condition
#   bash scripts/run_rby1_freeze_test.sh --only frozen

set -e

cd /workspace/MolmoBot/MolmoBot

# ── Environment ──────────────────────────────────────────────────────────
export MUJOCO_GL="${RBY1_MUJOCO_GL:-osmesa}"
export PYOPENGL_PLATFORM="${RBY1_PYOPENGL_PLATFORM:-osmesa}"
export MLSPACES_RENDER_DEVICE_ID="${MLSPACES_RENDER_DEVICE_ID:-none}"

# Use /workspace for caches so they persist across pod restarts.
# Keep the resource-manager symlink tree and backing cache as distinct paths:
# molmospaces_resources rejects them if they resolve to the same directory.
export MLSPACES_ASSETS_DIR=/workspace/.cache/molmo-spaces-resources
export MLSPACES_CACHE_DIR=/workspace/.cache/molmo-spaces-cache
mkdir -p "$MLSPACES_ASSETS_DIR"
mkdir -p "$MLSPACES_CACHE_DIR"

# Symlink ~/.cache for older resource links that point at the backing cache.
mkdir -p ~/.cache
ln -sfnT "$MLSPACES_CACHE_DIR" ~/.cache/molmo-spaces-resources

source .venv/bin/activate

# Pull latest
git pull

# ── Parse arguments ──────────────────────────────────────────────────────
TASK_HORIZON=400
NUM_EPISODES=1
CHECKPOINT_PATH=""
EXTRA_ARGS=""
ONLY=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --task_horizon)     TASK_HORIZON="$2"; shift 2 ;;
        --num_episodes)     NUM_EPISODES="$2"; shift 2 ;;
        --checkpoint_path)  CHECKPOINT_PATH="$2"; shift 2 ;;
        --only)             ONLY="$2"; shift 2 ;;
        *)                  EXTRA_ARGS="$EXTRA_ARGS $1"; shift ;;
    esac
done

# ── Asset cache initialization ───────────────────────────────────────────
echo "Checking asset cache..."
python -c "
from molmo_spaces.molmo_spaces_constants import get_resource_manager
rm = get_resource_manager()

# Install RBY1M robot
for robot in ['rby1m', 'rby1']:
    try:
        packages = rm.unindexed_archives('robots', robot)
        if packages:
            rm.install_packages('robots', {robot: packages})
            print(f'  Installed robots/{robot}')
        else:
            print(f'  robots/{robot}: already installed')
    except Exception as e:
        print(f'  robots/{robot}: {e}')

# Install Thor objects
try:
    packages = rm.unindexed_archives('objects', 'thor')
    if packages:
        rm.install_packages('objects', {'thor': packages})
        print('  Installed objects/thor')
    else:
        print('  objects/thor: already installed')
except Exception as e:
    print(f'  objects/thor: {e}')

# Install iTHOR scenes
try:
    packages = rm.unindexed_archives('scenes', 'ithor')
    if packages:
        rm.install_packages('scenes', {'ithor': packages})
        print('  Installed scenes/ithor')
    else:
        print('  scenes/ithor: already installed')
except Exception as e:
    print(f'  scenes/ithor: {e}')
" || echo "Asset init had warnings (may be OK if already cached)"

# ── Find or download checkpoint ──────────────────────────────────────────
if [ -z "$CHECKPOINT_PATH" ]; then
    # Try common locations
    for candidate in \
        ckpts/molmobot/MolmoBot-RBY1Multitask \
        /workspace/MolmoBot-RBY1-PickPnP \
        /workspace/MolmoBot-RBY1 \
        ~/.cache/huggingface/hub/models--allenai--MolmoBot-RBY1Multitask/snapshots/*/ \
        ~/.cache/huggingface/hub/models--allenai--MolmoBot-RBY1-PickPnP/snapshots/*/; do
        if [ -d "$candidate" ]; then
            CHECKPOINT_PATH="$candidate"
            break
        fi
    done
fi

if [ -z "$CHECKPOINT_PATH" ]; then
    echo "Checkpoint not found. Downloading allenai/MolmoBot-RBY1Multitask..."
    CHECKPOINT_PATH=$(python -c "from huggingface_hub import snapshot_download; print(snapshot_download('allenai/MolmoBot-RBY1Multitask', local_dir='ckpts/molmobot/MolmoBot-RBY1Multitask'))")
fi
echo "Checkpoint: $CHECKPOINT_PATH"

# ── Generate benchmark ───────────────────────────────────────────────────
BENCHMARK_PATH="benchmarks/rby1_pickpnp_benchmark"
echo "Generating benchmark ($NUM_EPISODES episodes)..."
python generate_rby1_pickpnp_benchmark.py \
    --output_path "${BENCHMARK_PATH}/benchmark.json" \
    --num_episodes "$NUM_EPISODES"

# ── Run evaluations ─────────────────────────────────────────────────────
OUTPUT_DIR="eval_output/rby1_freeze_test"

echo ""
echo "=========================================="
echo "RBY1 Pick&Place Freeze Test"
echo "=========================================="
echo "  Checkpoint:  $CHECKPOINT_PATH"
echo "  Benchmark:   $BENCHMARK_PATH ($NUM_EPISODES episodes)"
echo "  Task horizon: $TASK_HORIZON"
echo "  Output:      $OUTPUT_DIR"
echo "=========================================="
echo ""

run_condition() {
    local name="$1"
    local config_cls="$2"
    local out_dir="$3"

    echo "=== ${name} ==="
    python -u launch_scripts/run_eval.py \
        --checkpoint_path "$CHECKPOINT_PATH" \
        --benchmark_path "$BENCHMARK_PATH" \
        --eval_config_cls "$config_cls" \
        --task_horizon "$TASK_HORIZON" \
        --output_dir "$out_dir" \
        $EXTRA_ARGS
    echo ""
}

if [ "$ONLY" != "frozen" ]; then
    run_condition \
        "Condition 1: Default (mobile base ACTIVE)" \
        "olmo.eval.configure_molmo_spaces:MolmoBotRBY1PickPnPEvalConfig" \
        "${OUTPUT_DIR}/default"
fi

if [ "$ONLY" != "default" ]; then
    run_condition \
        "Condition 2: Frozen (mobile base LOCKED)" \
        "olmo.eval.configure_molmo_spaces:MolmoBotRBY1PickPnPFrozenBaseEvalConfig" \
        "${OUTPUT_DIR}/frozen_base"
fi

echo "=========================================="
echo "DONE"
echo "  Default:     ${OUTPUT_DIR}/default/"
echo "  Frozen base: ${OUTPUT_DIR}/frozen_base/"
echo "=========================================="
