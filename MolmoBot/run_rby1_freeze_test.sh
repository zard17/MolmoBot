#!/bin/bash
# RBY1 Pick&Place: Default vs Frozen Base comparison
#
# Usage:
#   bash run_rby1_freeze_test.sh <checkpoint_path> [benchmark_path]
#
# Example:
#   bash run_rby1_freeze_test.sh /path/to/MolmoBot-RBY1-PickPnP
#   bash run_rby1_freeze_test.sh /path/to/checkpoint benchmarks/rby1_pickpnp_benchmark
set -e

CHECKPOINT_PATH="${1:?Usage: $0 <checkpoint_path> [benchmark_path]}"
BENCHMARK_PATH="${2:-benchmarks/rby1_pickpnp_benchmark}"
OUTPUT_DIR="eval_output/rby1_freeze_test"
TASK_HORIZON=400

# Generate benchmark if it doesn't exist
if [ ! -f "${BENCHMARK_PATH}/benchmark.json" ]; then
    echo "=== Generating RBY1 pick&place benchmark ==="
    python generate_rby1_pickpnp_benchmark.py --output_path "${BENCHMARK_PATH}/benchmark.json"
    echo ""
fi

echo "=========================================="
echo "RBY1 Pick&Place Freeze Test"
echo "=========================================="
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "Benchmark:  ${BENCHMARK_PATH}"
echo "Output:     ${OUTPUT_DIR}"
echo ""

echo "=== Condition 1/2: Default (mobile base ACTIVE) ==="
python run_eval_with_glfw.py \
    --checkpoint_path "${CHECKPOINT_PATH}" \
    --benchmark_path "${BENCHMARK_PATH}" \
    --eval_config_cls "olmo.eval.configure_molmo_spaces:MolmoBotRBY1PickPnPEvalConfig" \
    --task_horizon ${TASK_HORIZON} \
    --output_dir "${OUTPUT_DIR}/default"
echo ""

echo "=== Condition 2/2: Frozen base (mobile base LOCKED) ==="
python run_eval_with_glfw.py \
    --checkpoint_path "${CHECKPOINT_PATH}" \
    --benchmark_path "${BENCHMARK_PATH}" \
    --eval_config_cls "olmo.eval.configure_molmo_spaces:MolmoBotRBY1PickPnPFrozenBaseEvalConfig" \
    --task_horizon ${TASK_HORIZON} \
    --output_dir "${OUTPUT_DIR}/frozen_base"
echo ""

echo "=========================================="
echo "DONE — Compare results:"
echo "  Default:     ${OUTPUT_DIR}/default/"
echo "  Frozen base: ${OUTPUT_DIR}/frozen_base/"
echo "=========================================="
