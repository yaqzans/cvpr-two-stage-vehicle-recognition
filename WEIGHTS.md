# Trained weights, where they are

All 27 checkpoint files in the project archive, measured from the archive
listing itself. Only the two marked **kept** are in this handover. The rest
come to 2.20 GB, and the disk this folder was written to had 1.51 GB free.
GitHub also refuses any single file above 100 MB, so the four 334 MB
classifier checkpoints could never have gone into the repository.

To pull any one of them out of the archive:

    unrar x Project1.rar "<path from the table>" <destination>

The archive is `Project1.rar` in the original project folder.

| Size | Modified | Path inside the archive | |
|---|---|---|---|
| 334.1 MB | 2026-08-26 | `Project1/runs/stage2_convnext_tiny_v2/checkpoints/best.pt` | |
| 334.1 MB | 2026-08-26 | `Project1/runs/stage2_convnext_tiny_v2/checkpoints/last.pt` | |
| 334.1 MB | 2026-08-14 | `Project1/runs/stage2_convnext_tiny/checkpoints/best.pt` | |
| 334.1 MB | 2026-08-14 | `Project1/runs/stage2_convnext_tiny/checkpoints/last.pt` | |
| 121.4 MB | 2026-09-10 | `Project1/runs/stage1_yolo12m/train/weights/last.pt` | |
| 121.4 MB | 2026-09-09 | `Project1/runs/stage1_yolo12m/train/weights/best.pt` | |
| 60.3 MB | 2026-08-25 | `Project1/runs/stage1_yolo26s/train/weights/last.pt` | |
| 40.9 MB | 2026-09-09 | `Project1/yolo12m.pt` | |
| 40.5 MB | 2026-09-10 | `Project1/runs/kaggle_state/stage1-yolo11m_output_cycle1/runs/stage1-yolo11m/train/weights/best.pt` | |
| 40.5 MB | 2026-09-10 | `Project1/runs/kaggle_state/stage1-yolo11m_output_cycle1/runs/stage1-yolo11m/train/weights/last.pt` | |
| 40.5 MB | 2026-09-10 | `Project1/runs/kaggle_state/checkpoint_staging/train/weights/best.pt` | |
| 40.5 MB | 2026-09-10 | `Project1/runs/kaggle_state/checkpoint_staging/train/weights/last.pt` | |
| 40.5 MB | 2026-09-10 | `Project1/runs/kaggle_state/stage1-yolo11m_final_pull/runs/stage1-yolo11m/train/weights/best.pt` | |
| 40.5 MB | 2026-09-10 | `Project1/runs/kaggle_state/stage1-yolo11m_final_pull/runs/stage1-yolo11m/train/weights/last.pt` | |
| 40.5 MB | 2026-09-10 | `Project1/runs/stage1_yolo11m/train/weights/best.pt` | |
| 40.5 MB | 2026-09-10 | `Project1/runs/stage1_yolo11m/train/weights/last.pt` | |
| 40.5 MB | 2026-09-09 | `Project1/runs/kaggle_state/stage1-yolo11m-smoketest7_output_cycle1/runs/stage1-yolo11m-smoketest7/train/weights/best.pt` | |
| 40.5 MB | 2026-09-09 | `Project1/runs/kaggle_state/stage1-yolo11m-smoketest7_output_cycle1/runs/stage1-yolo11m-smoketest7/train/weights/last.pt` | |
| 40.5 MB | 2026-09-09 | `Project1/runs/stage1-yolo11m-smoketest7/train/weights/best.pt` | |
| 40.5 MB | 2026-09-09 | `Project1/runs/stage1-yolo11m-smoketest7/train/weights/last.pt` | |
| 20.4 MB | 2026-08-25 | `Project1/scripts/yolo26s.pt` | |
| 5.5 MB | 2026-08-13 | `Project1/models/yolo26n.pt` | |
| 5.5 MB | 2026-08-13 | `Project1/scripts/yolo26n.pt` | |
| 5.5 MB | 2026-09-09 | `Project1/yolo26n.pt` | |
| 5.4 MB | 2026-08-14 | `Project1/runs/stage1_yolo26n/train/weights/best.pt` | **kept** |
| 5.4 MB | 2026-08-14 | `Project1/runs/stage1_yolo26n/train/weights/last.pt` | **kept** |
| 0.0 MB | 2026-08-25 | `Project1/runs/stage1_yolo26s/train/weights/best.pt` | |

Total across all 27 files: 2.21 GB.

The pair kept is the Stage-1 detector the paper reports. The classifier the
paper reports is `runs/stage2_convnext_tiny_v2/checkpoints/best.pt`, which is
334.1 MB and is the first one to fetch if you intend to run the pipeline.
