#!/usr/bin/env bash
# Keeps scripts/02_train_yolo12m.py alive across crashes.
#
# 02_train_yolo12m.py already resumes from runs/stage1_yolo12m/train/weights/last.pt
# on its own when re-launched, so the only thing missing was *relaunching* it
# after a crash (Windows DataLoader worker memory leaks, transient CUDA
# errors, etc.) without a person there to notice and restart it by hand.
# This loop is that missing piece: on any non-zero exit, wait 15s and run it
# again. A clean exit (training actually finished) breaks the loop.

cd "$(dirname "$0")/.."

while true; do
    python scripts/02_train_yolo12m.py --imgsz 640 --batch 8 --epochs 100 --patience 20
    status=$?
    if [ $status -eq 0 ]; then
        echo "Training finished cleanly."
        break
    fi
    echo "Training exited with code $status - restarting from last checkpoint in 15s..."
    sleep 15
done
