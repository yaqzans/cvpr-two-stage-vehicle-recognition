#!/usr/bin/env bash
# Calibration did not fix the mAP ranking. Test the other hypothesis: the
# problem is the score FORMULA, not its calibration.
#
# YOLO's confidence is trained to rank detections for AP. The classifier's
# probability is trained for balanced classification accuracy - a different
# objective. Multiplying them may simply be the wrong fusion. Try using
# YOLO's confidence for ranking while taking the class from Stage 2.
set -u
cd "$(dirname "$0")/.." || exit 1
EVAL="python scripts/06_evaluate_pipeline.py"
V2="runs/stage2_convnext_tiny_v2/checkpoints/best.pt"
run () { echo ""; echo "### $*"; eval "$@" 2>&1 | grep -vE "batch/s|it/s|^  [0-9]+/[0-9]+$"; }
{
echo "SCORE FUSION  started $(date)"
run "$EVAL --mode two-stage --split test --conf 0.001 --cls-checkpoint $V2 --score det --tag 17_test_scoredet"
run "$EVAL --mode two-stage --split val  --conf 0.001 --cls-checkpoint $V2 --score det --tag 18_val_scoredet"
run "$EVAL --mode two-stage --split test --conf 0.001 --cls-checkpoint $V2 --score cls --tag 19_test_scorecls"
run "$EVAL --mode two-stage --split val  --conf 0.001 --cls-checkpoint $V2 --score cls --tag 20_val_scorecls"
echo ""; echo "SCORE FUSION  finished $(date)"
} | tee scripts/score_fusion.log
