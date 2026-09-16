# Two-stage Bangladesh vehicle recognition, code handover

A road scene goes to a YOLO detector, every box it returns is cropped, and a
ConvNeXt-Tiny classifier relabels each crop from scratch. Stage 1's own class
prediction is thrown away. That single decision is what the paper argues about.

This folder is a copy. Nothing in the original project was moved, renamed or
edited to produce it.

## Read this before anything else

**The dataset is not here.** RSUD20K is roughly 11 GB and was never inside the
project archive this folder was built from. Without it nothing trains and
nothing re-evaluates. What you can do is read the code and check every number
in the paper against the result files, which are all present.

**Almost none of the trained weights are here either.** They came to 2.21 GB
across 27 files, and the disk the handover was written to had 1.51 GB free.
Two were kept, listed below.

So treat this as a folder for auditing the work, not for re-running it.

## Setup

    python -m venv .venv
    .venv\Scripts\activate
    pip install -r requirements-core.txt

`requirements-core.txt` is the honest list. It was built by parsing every
import in all 64 Python files, and pinned from versions recorded in the
project's own logs. There was no `requirements.txt` in the project to begin
with, so nothing was superseded.

## Where the numbers live, and how to trace one

Every figure in the paper traces to one of these. Nothing was typed from
memory.

| File | What it settles |
|---|---|
| `Project1/runs/pipeline_eval/SUMMARY.csv` | one row per end-to-end run: mAP50, mAP50-95, localised, e2e acc, ms/frame, FPS |
| `Project1/runs/pipeline_eval/METRIC_STABILITY.csv` | per class, per split, single against two-stage, for AP50 and end-to-end |
| `Project1/runs/stage2_convnext_tiny_v2/test_per_class.csv` | Stage-2 precision, recall, F1 and support for all 13 classes |
| `Project1/runs/stage2_convnext_tiny_v2/history.csv` | per-epoch train and validation loss, and the macro F1 the checkpoint is picked on |
| `Project1/runs/stage2_convnext_tiny_v2/calibration.json` | temperature, and ECE and NLL before and after |
| `Project1/runs/stage1_*/crop_recall_eval/crop_recall_sweep_test.csv` | crop recall against confidence, both matching rules |
| `Project1/runs/*/train/args.yaml` | the exact arguments each training run was given |

Worked example. The paper says end-to-end recognition accuracy rises from
91.5% to 95.9% on test. Open `SUMMARY.csv`, take the `e2e acc` column, and read
row `01_single_stage_baseline` against row `15_fastcrop_conf0001`. Both are
conf 0.001 on the test split, so they are the same boxes with a different
labeller. That is the comparison.

Second example. The paper says only 31% of a person crop's long dimension
survived the old preprocessing. That one is arithmetic, not a file: the crop is
234 by 661, `Resize(256)` scales the short side by 256/234, and `CenterCrop(224)`
then keeps 224 of the 723 pixels left along the long axis.

## Running things, in order

Nothing below will complete without RSUD20K. The commands are recorded so the
sequence is unambiguous, with the runtimes the original logs recorded.

    python Project1/scripts/01_verify_rsud20k.py
      checks image and label pairing, class ids, box ranges. Seconds.

    python generate_classification_dataset.py --input <RSUD20K> --output Datasets/BangladeshVehicleClassification
      cuts every annotated box into its own file. Minutes.

    python Project1/scripts/02_train_yolo26n.py
      resume-only wrapper. The original run took 15,162 seconds, which is 4.2
      hours, read from the time column of runs/stage1_yolo26n/train/results.csv.

    python Project1/scripts/train_stage2_convnext_tiny_v2.py
      the classifier the paper uses. 11,959 seconds, which is 3.3 hours, summed
      from epoch_time_sec in that run's history.csv.

    python Project1/scripts/05_eval_stage1_crop_recall.py --weights <best.pt>
      crop recall sweep. This is the pipeline-correct Stage-1 metric.

    python Project1/scripts/06_evaluate_pipeline.py --mode single    --conf 0.001 --tag baseline
    python Project1/scripts/06_evaluate_pipeline.py --mode two-stage --conf 0.05 --fast-crop --tag deploy
    python Project1/scripts/07_summarize_results.py
      the end-to-end comparison and the SUMMARY table.

    bash Project1/scripts/run_pipeline_matrix.sh
      reproduces the whole evaluation matrix in one go.

There are 21 scripts carrying an argparse block or a main guard. The numbered
prefixes are the intended order.

## Layout

    Project1/scripts/      39 files. Every entry point, plus the Kaggle driver.
    Project1/runs/        311 files. Training logs, args, metrics, result CSVs.
    Project1/figures/      46 files. The 21 project figures, as PNG and PDF.
    Project1/archive/      59 files. Two frozen snapshots from 26 August 2026.
    Project1/configs/       4 files. Data YAMLs and the oversampled train list.
    Project1/docs/          5 files. The earlier write-ups.
    Project1/box draw/     12 files. A small box-drawing utility.
    Project1/             10 root files, including PROJECT_HANDOFF.md.

491 files, 187 MB in total, this README and the two files beside it included.

## Landmines

**Hardcoded absolute paths.** 53 text files contain `C:\Users\faalv\Desktop\...`.
Only seven of them block a re-run, and those are the ones to repoint first:
all three YAMLs in `configs/`, the oversampled train list
`configs/rsud20k_train_oversampled.txt`, which carries the path 39,121 times,
once per duplicated image, and the three `args.yaml` files under `runs/`.
The other 46 are records rather than inputs: `metrics.json` provenance fields,
training logs and the handoff itself. Leave those alone, they are evidence of
where the work actually ran.

**PROJECT_HANDOFF.md is stale in places.** It is a good narrative and a poor
reference. Three of its claims disagree with the result files, and the files
win every time. It says eight classes flip the sign of their AP delta between
splits, `METRIC_STABILITY.csv` gives six. It reports 2.7 unmatched boxes per
frame as the deployment figure, which is the conf 0.10 rate, and the conf 0.05
rate is 4.19. It gives a person crop's surviving fraction as 32%, and the
arithmetic gives 31%.

**The detector was never oversampled.** The handoff and the class-imbalance
discussion make rare-class oversampling sound like it applied to Stage 1. It
did not. `runs/stage1_yolo26n/train/args.yaml` points at `configs/rsud20k_yolo.yaml`,
which is the plain split. Oversampling reached only the YOLO11m and YOLO12m runs.

**`scripts04_evaluate_stage1_crops.py` is superseded** and kept only because an
early write-up quotes its 84.63% figure. Use `05_eval_stage1_crop_recall.py`.

**Two mAP numbers exist for the same weights.** Table 4 of the paper reports
Ultralytics `val()` output, and the end-to-end evaluator computes its own mAP so
that both systems are scored by one scorer. They differ by about 1.7 points on
YOLO26n. Neither is wrong.

**Timing is not reproducible to better than about 1.5x.** Two runs of the
identical single-stage validation configuration report 43.6 and 29.2 FPS in
`SUMMARY.csv`.

## What was excluded, and why

| Excluded | Size | Reason |
|---|---|---|
| 8,395 crop JPEGs under `runs/stage1_yolo26n/crop_evaluation` and `crops_conf0.1` | 225 MB | Saved output of `05_eval_stage1_crop_recall.py --save-crops-at`, not an input. They were 94% of all files in the archive and back no number in the paper. |
| 25 of the 27 `.pt` weight files | 2.21 GB of 2.22 GB | Would not fit: 1.51 GB free on the target disk. Four ConvNeXt checkpoints are 334 MB each, which also exceeds GitHub's 100 MB per-file limit. See WEIGHTS.md for all 27 and their archive paths. |
| `figures.rar` | 27.3 MB | Verified duplicate. All 46 files inside are byte-identical by md5 to `figures/`, with nothing extra. |
| `scripts.rar` | 28.2 MB | Superseded snapshot. 36 files byte-identical, none differing, 9 current scripts missing from it, and its only two extra files are pretrained checkpoints covered by the weights policy above. |
| `Ultralytics/settings.json` and `scripts/Ultralytics/settings.json` | 2 files | Machine-local tool config. Both `api_key` and `openai_api_key` were empty, however each carries a 64-character telemetry install uuid tied to the original author's machine. Ultralytics regenerates them on first run. |
| `.pptx_review/` | 61 files | A scratch directory from unzipping a PowerPoint, not project code. |
| 4 `.pyc` files and their `__pycache__` directories | small | Regenerable. |
| `Project1/scripts/Code.txt` | 13.0 MB | Verified duplicate. Byte-identical by md5 to `Code.log` in the same directory, which is kept. |

Two weights were kept: `runs/stage1_yolo26n/train/weights/best.pt` and
`last.pt`, 5,377,342 bytes each. That is the detector the paper reports.

## Provenance and defect ledgers

`Project1/PROJECT_HANDOFF.md` is the narrative record of every run, every dead
end and every bug, written by the original author. `Project1/STAGE1_MODEL_COMPARISON.md`
compares the three detectors. `Project1/docs/RESULTS_2026-08-26.md` is the
earlier write-up the handoff builds on. Read them with the caveat above.
