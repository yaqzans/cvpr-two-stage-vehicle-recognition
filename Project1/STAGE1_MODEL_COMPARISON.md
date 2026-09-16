# Stage 1 Detector Comparison: YOLO26n vs YOLO11m vs YOLO12m

All three evaluated with identical methodology on the untouched RSUD20K test split (649 images, 3,805 objects), same benchmark scripts used throughout the project (`03_evaluate_yolo26n.py` pattern, `05_eval_stage1_crop_recall.py`, `09_benchmark_throughput.py`).

## Training summary

| Model | Params | Epochs trained | Training platform | Notes |
|---|---|---|---|---|
| YOLO26n | 2.38M | 64 (best@44) | Local RTX 4060 Ti | Original baseline |
| YOLO11m | 20.1M | 18 | Kaggle T4 (cloud) | Time-capped at 8.5h; auto-adjusted LR schedule to fit |
| YOLO12m | 20.1M | 51 (best@22) | Local RTX 4060 Ti | Patience never triggered — repeated PC restarts reset ultralytics' early-stop counter each resume (confirmed in source); true best is epoch 22 |

## 1. Detection metrics (test split, imgsz=640)

| Model | mAP50 | mAP50-95 | Precision | Recall |
|---|---|---|---|---|
| YOLO26n | 0.7602 | 0.6110 | 0.7524 | 0.6694 |
| YOLO11m | 0.8232 | 0.6656 | 0.7957 | 0.7385 |
| YOLO12m | **0.8538** | **0.6990** | 0.7936 | **0.7946** |

Both medium backbones clearly beat nano. YOLO12m is best on every metric except precision (statistically tied with 11m).

## 2. Usable crop recall (the pipeline-relevant metric)

Class-agnostic recall = did Stage 1 hand Stage 2 a usable box (IoU≥0.5), regardless of Stage 1's guessed label — the real question for a detect-then-classify pipeline.

| Model | conf=0.05 agnostic | conf=0.05 class-aware | conf=0.10 agnostic | mean IoU |
|---|---|---|---|---|
| YOLO26n | 94.69% | 91.56% | 92.43% | 0.893 |
| YOLO11m | 95.72% | 93.61% | 93.96% | 0.902 |
| YOLO12m | **97.08%** | **95.32%** | **95.98%** | 0.906 |

YOLO12m misses only 2.9% of real objects at conf=0.05, vs 5.3% for the nano baseline — a 46% reduction in missed objects for Stage 2 to never see.

## 3. Two-stage pipeline throughput (1080p, RTX 4060 Ti)

| Model | Best batch | FPS (GPU) | FPS (+decode) | Streaming (batch=1) FPS | Streaming latency |
|---|---|---|---|---|---|
| YOLO26n | 16 | 48.8 | 33.1 | 27.8 | 35.9 ms |
| YOLO11m | 4 | 40.1 | 29.3 | 25.7 | 38.9 ms |
| YOLO12m | 4 | 39.1 | 28.9 | 24.8 | 40.3 ms |

Speed cost of the bigger backbone is modest: ~11% slower streaming FPS for YOLO12m vs nano, ~20% slower at best-case batching. Given the accuracy gain, this is a good trade.

## Verdict

**YOLO12m is the best Stage-1 detector found in this project.** It improves usable crop recall from 94.69% → 97.08% (nearly halving the miss rate) and raises detection mAP50 by +9.4 points over the nano baseline, at a real-time cost of only ~3 FPS streaming (27.8 → 24.8). YOLO11m is a solid second, undertrained relative to 12m (18 vs 51 epochs) due to Kaggle's session/quota limits, but already beats nano on every metric.

**Recommendation:** adopt YOLO12m (`runs/stage1_yolo12m/train/weights/best.pt`) as the new Stage-1 detector, re-run Stage-2 evaluation and full pipeline benchmarks against it, and update the poster/paper's Stage-1 numbers accordingly. If more Kaggle GPU quota becomes available, resuming YOLO11m past epoch 18 could let it approach or match 12m, since both share the same parameter count and architecture family capacity.
