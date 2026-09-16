"""Builds the Stage-1 training notebook from plain source strings.

Per the field manual: never hand-edit notebook JSON (escaping bugs), and
never rely on kernel internet access (it kills kernels silently even on
phone-verified accounts — INC 01). Ultralytics and its small missing deps
are installed from the offline wheel bundle at
fairoozalammahi/ultralytics-offline-wheels; RSUD20K comes from the public
hasibzunair dataset; a checkpoint dataset is attached only once a resume is
needed.

Called by push_and_run.py, not run directly.
"""

from __future__ import annotations

import json
from pathlib import Path


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


CELL_ENVIRONMENT_CHECK = '''\
import glob, torch
print("torch", torch.__version__, "cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    name = torch.cuda.get_device_name(0)
    print("GPU:", name)
    assert "P100" not in name, (
        f"Got {name} (CUDA capability 6.0) - incompatible with this PyTorch build. "
        "Push must use --accelerator NvidiaTeslaT4."
    )
else:
    raise RuntimeError("No GPU visible - check the push command's --accelerator flag.")

print("\\n/kaggle/input contents (up to 3 levels deep):")
for depth in ("*", "*/*", "*/*/*"):
    for p in sorted(glob.glob(f"/kaggle/input/{depth}")):
        print(" ", p)
'''

CELL_OFFLINE_INSTALL = '''\
import os, subprocess, sys

wheels_dir = None
for dirpath, _dirnames, filenames in os.walk("/kaggle/input"):
    if any(f.startswith("ultralytics-") and f.endswith(".whl") for f in filenames):
        wheels_dir = dirpath
        break
assert wheels_dir, "ultralytics-offline-wheels dataset not attached or empty"
print("Installing ultralytics from offline wheels:", wheels_dir)

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "--no-index",
     "--find-links", wheels_dir, "ultralytics"],
    check=True,
)
import ultralytics
print("ultralytics", ultralytics.__version__, "installed offline, OK")
'''

CELL_TRAINING = '''\
import os
from collections import Counter
from pathlib import Path

MODEL_NAME = {model_name!r}
IMGSZ = {imgsz}
BATCH = {batch}
EPOCHS = {epochs}
PATIENCE = {patience}
TIME_LIMIT_HOURS = {time_limit_hours}
RUN_NAME = {run_name!r}

WORKING = Path("/kaggle/working")
INPUT = Path("/kaggle/input")
RUNS_DIR = WORKING / "runs" / RUN_NAME
train_dir = RUNS_DIR / "train"

CLASS_NAMES = [
    "person", "rickshaw", "rickshaw_van", "auto_rickshaw", "truck",
    "pickup_truck", "private_car", "motorcycle", "bicycle", "bus",
    "micro_bus", "covered_van", "human_hauler",
]
OVERSAMPLE_EXTRA_COPIES = {{
    "truck": 3, "pickup_truck": 3, "covered_van": 3,
    "rickshaw_van": 2, "bus": 2,
}}
IMAGE_EXTENSIONS = {{".jpg", ".jpeg", ".png", ".bmp", ".webp"}}


def find_dataset_root():
    # os.walk, not glob/rglob: both of Path.rglob's and glob.glob's
    # recursive "**" matching were unreliable against this read-only
    # FUSE-mounted dataset (20k+ files) even though the identical target
    # directory was found correctly via os.walk in a standalone diagnostic
    # run moments earlier. os.walk is what's proven to work here.
    import os as _os
    for dirpath, _dirnames, _filenames in _os.walk("/kaggle/input"):
        if _os.path.basename(dirpath) == "train" and _os.path.basename(_os.path.dirname(dirpath)) == "images":
            root = Path(_os.path.dirname(_os.path.dirname(dirpath)))
            if (root / "labels" / "train").is_dir():
                return root
    raise FileNotFoundError("RSUD20K not found under /kaggle/input (expected images/train + labels/train)")


def find_checkpoint_run():
    import os as _os
    for dirpath, _dirnames, filenames in _os.walk("/kaggle/input"):
        if _os.path.basename(dirpath) == "weights" and "last.pt" in filenames:
            run_dir = Path(_os.path.dirname(dirpath))
            if (run_dir / "args.yaml").is_file():
                return run_dir
    return None


def find_pretrained_weights(model_name):
    # Internet is off, so YOLO(MODEL_NAME) can't download the pretrained
    # weights itself - they're attached as a dataset and located by filename.
    import os as _os
    for dirpath, _dirnames, filenames in _os.walk("/kaggle/input"):
        if model_name in filenames:
            return str(Path(dirpath) / model_name)
    raise FileNotFoundError(f"{{model_name}} not found under /kaggle/input - is the pretrained-weights dataset attached?")


def build_oversampled_train_list(dataset_root):
    images_dir = dataset_root / "images" / "train"
    labels_dir = dataset_root / "labels" / "train"
    target_ids = {{CLASS_NAMES.index(name): extra for name, extra in OVERSAMPLE_EXTRA_COPIES.items()}}
    images = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    lines, dup_counts = [], Counter()
    for image_path in images:
        lines.append(str(image_path.resolve()))
        label_path = labels_dir / f"{{image_path.stem}}.txt"
        if not label_path.exists():
            continue
        present_ids = set()
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            try:
                present_ids.add(int(parts[0]))
            except ValueError:
                continue
        matched = [target_ids[cid] for cid in present_ids if cid in target_ids]
        if matched:
            extra = max(matched)
            for _ in range(extra):
                lines.append(str(image_path.resolve()))
            for cid in present_ids:
                if cid in target_ids:
                    dup_counts[CLASS_NAMES[cid]] += extra
    WORKING.mkdir(parents=True, exist_ok=True)
    list_path = WORKING / "rsud20k_train_oversampled.txt"
    list_path.write_text("\\n".join(lines) + "\\n", encoding="utf-8")
    print(f"Base train images: {{len(images)}}, total after oversampling: {{len(lines)}}")
    for name in OVERSAMPLE_EXTRA_COPIES:
        print(f"  {{name:15s}} +{{dup_counts.get(name, 0)}}")
    return list_path


def write_data_yaml(dataset_root, train_list_path):
    val_dir = (dataset_root / "images" / "val").resolve()
    test_dir = (dataset_root / "images" / "test").resolve()
    names_block = "\\n".join(f"  {{i}}: {{name}}" for i, name in enumerate(CLASS_NAMES))
    yaml_text = (
        f"train: {{train_list_path.resolve().as_posix()}}\\n"
        f"val: {{val_dir.as_posix()}}\\n"
        f"test: {{test_dir.as_posix()}}\\n"
        f"names:\\n{{names_block}}\\n"
    )
    yaml_path = WORKING / "rsud20k_oversampled.yaml"
    yaml_path.write_text(yaml_text, encoding="utf-8")
    print("Data yaml:", yaml_path)
    return yaml_path


dataset_root = find_dataset_root()
print("Dataset root:", dataset_root)
assert (dataset_root / "images" / "train").is_dir(), "train images missing"
assert (dataset_root / "labels" / "train").is_dir(), "train labels missing"

checkpoint_run = find_checkpoint_run()
RUNS_DIR.mkdir(parents=True, exist_ok=True)

from ultralytics import YOLO

if checkpoint_run is not None:
    print("Resuming from checkpoint:", checkpoint_run)
    import shutil
    if train_dir.exists():
        shutil.rmtree(train_dir)
    shutil.copytree(checkpoint_run, train_dir)
    model = YOLO(str(train_dir / "weights" / "last.pt"))
    model.train(resume=True, time=TIME_LIMIT_HOURS)
else:
    print("Starting fresh training run:", MODEL_NAME)
    train_list_path = build_oversampled_train_list(dataset_root)
    data_yaml = write_data_yaml(dataset_root, train_list_path)
    pretrained_path = find_pretrained_weights(MODEL_NAME)
    print("Pretrained weights:", pretrained_path)
    model = YOLO(pretrained_path)
    model.train(
        data=str(data_yaml), imgsz=IMGSZ, epochs=EPOCHS, patience=PATIENCE,
        batch=BATCH, time=TIME_LIMIT_HOURS, device=0, workers=4,
        optimizer="auto", close_mosaic=10, seed=42, deterministic=True,
        project=str(RUNS_DIR), name="train", exist_ok=True,
        fliplr=0.5, flipud=0.0, degrees=0.0, translate=0.10, scale=0.50,
        hsv_h=0.015, hsv_s=0.70, hsv_v=0.40,
    )
'''

CELL_COMPLETION_CHECK = '''\
results_csv = train_dir / "results.csv"
done = False
if results_csv.exists():
    last_line = results_csv.read_text(encoding="utf-8").strip().splitlines()[-1]
    try:
        last_epoch = int(last_line.split(",")[0])
        done = last_epoch >= EPOCHS
    except ValueError:
        pass
    if (train_dir.parent / "train.log").exists():
        log_text = (train_dir.parent / "train.log").read_text(encoding="utf-8", errors="ignore")
        done = done or "Stopping training early" in log_text

if done:
    (WORKING / "DONE.flag").write_text("complete\\n", encoding="utf-8")
    print("\\n=== Training complete: DONE.flag written ===")
else:
    print("\\n=== Time limit reached before convergence: checkpoint saved, resume needed ===")

assert (train_dir / "weights" / "last.pt").exists(), "no checkpoint written - nothing to resume from or evaluate"
'''


def build(fig_dir: Path, config: dict) -> Path:
    """Write train.ipynb into fig_dir (the staging directory for this push)."""
    training_source = CELL_TRAINING.format(**config)

    notebook = {
        "cells": [
            code_cell(CELL_ENVIRONMENT_CHECK),
            code_cell(CELL_OFFLINE_INSTALL),
            code_cell(training_source),
            code_cell(CELL_COMPLETION_CHECK),
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    nb_path = fig_dir / "train.ipynb"
    nb_path.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    return nb_path
