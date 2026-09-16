#!/usr/bin/env bash
# Runs the full end-to-end evaluation matrix. Every run writes to its own
# tagged folder under runs/pipeline_eval/, so nothing overwrites anything.
#
#   python/bash scripts/run_pipeline_matrix.sh
set -u

cd "$(dirname "$0")/.." || exit 1
EVAL="python scripts/06_evaluate_pipeline.py"
V1="runs/stage2_convnext_tiny/checkpoints/best.pt"
V2="runs/stage2_convnext_tiny_v2/checkpoints/best.pt"
LOG="scripts/pipeline_matrix.log"

run () {
  echo ""
  echo "############################################################"
  echo "# $*"
  echo "############################################################"
  eval "$@" 2>&1 | grep -vE "batch/s|it/s|^  [0-9]+/[0-9]+$"
}

{
echo "PIPELINE MATRIX  started $(date)"

# --- ablation: does the classifier actually help, and did v2 help on REAL crops?
run "$EVAL --mode single    --conf 0.001 --tag 01_single_stage_baseline"
run "$EVAL --mode two-stage --conf 0.001 --cls-checkpoint $V2 --tag 02_two_stage_v2"
run "$EVAL --mode two-stage --conf 0.001 --cls-checkpoint $V1 --tag 03_two_stage_v1"

# --- operating points: accuracy vs speed
run "$EVAL --mode two-stage --conf 0.05 --cls-checkpoint $V2 --tag 04_op_conf005"
run "$EVAL --mode two-stage --conf 0.10 --cls-checkpoint $V2 --tag 05_op_conf010"
run "$EVAL --mode two-stage --conf 0.25 --cls-checkpoint $V2 --tag 06_op_conf025"
run "$EVAL --mode two-stage --conf 0.50 --cls-checkpoint $V2 --tag 07_op_conf050"
run "$EVAL --mode single    --conf 0.25                      --tag 08_single_conf025"

# --- free improvements that need no retraining
run "$EVAL --mode two-stage --conf 0.10 --cls-checkpoint $V2 --tta --tag 09_tta_conf010"
run "$EVAL --mode two-stage --conf 0.10 --cls-checkpoint $V2 --imgsz 960  --tag 10_imgsz960"
run "$EVAL --mode two-stage --conf 0.10 --cls-checkpoint $V2 --imgsz 1280 --tag 11_imgsz1280"
run "$EVAL --mode single    --conf 0.001 --imgsz 960          --tag 12_single_imgsz960"

echo ""
echo "PIPELINE MATRIX  finished $(date)"
} | tee "$LOG"
