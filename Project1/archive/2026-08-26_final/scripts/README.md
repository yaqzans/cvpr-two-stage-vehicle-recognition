# Scripts

Two-stage Bangladesh road-object recognition.

```
road scene -> Stage 1: YOLO26n (where?) -> crops -> Stage 2: ConvNeXt-Tiny (what?)
```

Run everything from the project root.

---

## Stage 1 — detector

| Script | What it does |
|---|---|
| `01_verify_rsud20k.py` | Integrity check on image/label pairs, class ids, box ranges. |
| `02_train_yolo26n.py` | Resume-only trainer for the YOLO26n baseline (already trained). |
| `02_train_yolo26s.py` | YOLO26s at imgsz 960 with rare-class oversampling. **Not yet run** (~2 days). |
| `02_train_yolo26sshort.py` | Same at imgsz 640, 50 epochs (~10 h). One epoch completed, then stopped. |
| `03_evaluate_yolo26n.py` | Ultralytics `val()` on the untouched test split. |
| `04_infer_yolo26n.py` | Inference on a file, folder, or video. |

## Stage 1 — crop quality

| Script | What it does |
|---|---|
| `scripts04_evaluate_stage1_crops.py` | **Superseded.** Original crop-recall check. Required the detector to predict the right class and ran at conf 0.25, so it reports 84.63% — a floor, not the pipeline's recall. Kept because the results write-up cites it. |
| `05_eval_stage1_crop_recall.py` | Pipeline-correct replacement. Class-agnostic matching (the pipeline discards YOLO's label), confidence sweep, false-positive accounting, no filename collisions. Reports **92.43%** at conf 0.10. |

```bash
python scripts/05_eval_stage1_crop_recall.py
python scripts/05_eval_stage1_crop_recall.py --save-crops-at 0.10
```

## Stage 2 — classifier

| Script | What it does |
|---|---|
| `train_stage2_convnext_tiny_fixed.py` | **v1.** Preserved for the ablation. Accuracy 0.9466, macro F1 0.8691. |
| `train_stage2_convnext_tiny_v2.py` | **v2, current best.** Accuracy 0.9703, macro F1 0.8940. Three changes vs v1 — pad-to-square preprocessing, a cosine schedule that completes, and box jitter. Documented in the file header. |

```bash
python scripts/train_stage2_convnext_tiny_v2.py      # ~3.3 h, 30 epochs
```

## End-to-end

| Script | What it does |
|---|---|
| `06_evaluate_pipeline.py` | Runs both stages and scores the result as a detector — mAP, end-to-end recognition accuracy, per-stage timing. Also runs `--mode single` for the baseline, using the same evaluator so the comparison is fair. |
| `07_summarize_results.py` | Aggregates every run into `runs/pipeline_eval/SUMMARY.{csv,md}`. |
| `08_demo_pipeline.py` | Side-by-side ground truth vs prediction frames. |
| `09_benchmark_throughput.py` | Latency (batch 1) and batched throughput. |
| `run_pipeline_matrix.sh` | Reproduces the full 16-run evaluation matrix. |

```bash
python scripts/06_evaluate_pipeline.py --mode single    --conf 0.001 --tag baseline
python scripts/06_evaluate_pipeline.py --mode two-stage --conf 0.05 --fast-crop --tag deploy
bash   scripts/run_pipeline_matrix.sh
python scripts/07_summarize_results.py
```

Use `--fast-crop` for deployment-representative timing: OpenCV preprocessing
with GPU normalisation, identical accuracy, ~2.6x faster than the PIL path.

## Dataset generation

| Script | What it does |
|---|---|
| `../generate_classification_dataset.py` | Crops every RSUD20K box into `Datasets/BangladeshVehicleClassification/`. Preserves the original split. |
| `10_generate_crops_with_margin.py` | Same, but keeps a context margin around each box so box jitter can land on real pixels instead of grey padding. Writes to a **separate** folder. |

---

## Where results live

| Path | Contents |
|---|---|
| `runs/stage1_yolo26n/` | Detector training, test metrics, crop-recall analysis |
| `runs/stage2_convnext_tiny/` | Stage-2 **v1** — preserved, do not overwrite |
| `runs/stage2_convnext_tiny_v2/` | Stage-2 **v2** — current best |
| `runs/pipeline_eval/` | All end-to-end runs, `SUMMARY.csv`, demo frames, throughput |
| `docs/RESULTS_2026-08-26.md` | Full write-up |
| `archive/` | Frozen snapshots of scripts + results |

## Current headline numbers

| Metric | Value |
|---|---|
| Stage 1 usable crop recall (conf 0.10) | 92.43% |
| Stage 2 test accuracy / macro F1 | 0.9703 / 0.8940 |
| End-to-end recognition accuracy | 0.9590 |
| End-to-end mAP@50 / mAP@50-95 | 0.7595 / 0.6131 |
| Single-stage baseline mAP@50 | 0.7432 |
| Throughput — streaming / batched | 22.3 / 33.1 FPS |
