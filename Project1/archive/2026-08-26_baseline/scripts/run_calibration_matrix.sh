#!/usr/bin/env bash
# Does temperature scaling improve the end-to-end mAP ranking, and does it
# hold on BOTH splits? T was fitted on val only.
set -u
cd "$(dirname "$0")/.." || exit 1
EVAL="python scripts/06_evaluate_pipeline.py"
V2="runs/stage2_convnext_tiny_v2/checkpoints/best.pt"
T=0.6508

run () {
  echo ""; echo "### $*"; eval "$@" 2>&1 | grep -vE "batch/s|it/s|^  [0-9]+/[0-9]+$"
}

{
echo "CALIBRATION MATRIX  started $(date)"
run "$EVAL --mode two-stage --split test --conf 0.001 --cls-checkpoint $V2 --temperature $T --tag 13_test_calibrated"
run "$EVAL --mode single    --split val  --conf 0.001                       --tag 14_val_single"
run "$EVAL --mode two-stage --split val  --conf 0.001 --cls-checkpoint $V2 --temperature 1.0 --tag 15_val_uncalibrated"
run "$EVAL --mode two-stage --split val  --conf 0.001 --cls-checkpoint $V2 --temperature $T  --tag 16_val_calibrated"
echo ""; echo "CALIBRATION MATRIX  finished $(date)"
} | tee scripts/calibration_matrix.log
