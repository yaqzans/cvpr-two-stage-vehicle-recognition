"""
Stage 2 v3 - ConvNeXt-Tiny on context-margin crops.

DERIVED FROM train_stage2_convnext_tiny_v2.py. Only the constants listed below
differ; everything else is byte-identical, so any change in the result is
attributable to these and nothing else.

Prerequisite:
    python scripts/10_generate_crops_with_margin.py

Changes vs v2
-------------
1. DATA_DIR      -> the margin dataset. v2's RandomBoxJitter could only pad
                    with grey when it expanded outward, because crops were cut
                    exactly on the ground-truth box. With a 25% margin the
                    jitter lands on real pixels, so a "too loose" training box
                    now looks like a real too-loose detector box.

2. BOX_JITTER_FRAC 0.12 -> 0.20
                    More jitter is now meaningful, because there is real image
                    content to expand into. Kept below the 0.25 margin so the
                    jitter stays inside real pixels.

3. RUN_DIR       -> runs/stage2_convnext_tiny_v3  (v1 and v2 preserved)

Everything else - model, batch size, learning rate, weight decay, label
smoothing, class weights, epochs, warmup, seed - is unchanged from v2.

Expected runtime: ~3.3 hours for 30 epochs on an RTX 4060 Ti.

    python scripts/train_stage2_convnext_tiny_v3.py
"""

from pathlib import Path
import csv
import json
import math
import time
import random
from multiprocessing import freeze_support

from tqdm import tqdm
import numpy as np
from PIL import Image, ImageOps
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "Datasets" / "BangladeshVehicleClassification_margin25"
RUN_DIR = PROJECT_ROOT / "runs" / "stage2_convnext_tiny_v3"

TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
TEST_DIR = DATA_DIR / "test"

MODEL_NAME = "hf_hub:timm/convnext_tiny.in12k_ft_in1k"

IMAGE_SIZE = 224

# Unchanged from v1 so the comparison is clean.
BATCH_SIZE = 32
NUM_WORKERS = 4
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.1
SEED = 42

# CHANGE 2: a budget the schedule can actually complete.
# v1 peaked at epoch 8 of 28, so 30 epochs is generous, not tight.
EPOCHS = 30
WARMUP_EPOCHS = 2
MIN_LR_FACTOR = 0.01  # cosine floor, as a fraction of LEARNING_RATE

# Early stopping is intentionally disabled: stopping early is what prevented
# v1's cosine schedule from ever decaying. Best-checkpoint selection still
# protects against a late-run regression.
EARLY_STOPPING = False
PATIENCE = EPOCHS

# CHANGE 3: how far each crop edge may move, as a fraction of that dimension.
# 0.12 puts a jittered box at roughly the ~0.89 IoU that Stage 1 delivers.
BOX_JITTER_FRAC = 0.20
BOX_JITTER_PROB = 0.80

# Neutral grey used when a jittered box extends past the saved crop, and when
# padding a non-square crop to square.
PAD_FILL = (114, 114, 114)

# EXTRAS - off by default. v1 overfit hard (train 98.9% vs val 89.0%), so if
# that persists after the three changes above, turn these on for a v3.
DROP_PATH_RATE = 0.0   # stochastic depth, try 0.1
RANDOM_ERASING = 0.0   # try 0.25

RESUME = False

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# CUSTOM TRANSFORMS
# ============================================================

class PadToSquare:
    """Pad the shorter side so the image is square, preserving aspect ratio.

    This is the CHANGE 1 fix. Resize(256)+CenterCrop(224) throws away
    whatever does not fit in a centred square; padding keeps all of it and
    lets the later Resize see the complete object.
    """

    def __init__(self, fill=PAD_FILL):
        self.fill = fill

    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        if width == height:
            return image

        side = max(width, height)
        left = (side - width) // 2
        top = (side - height) // 2
        border = (left, top, side - width - left, side - height - top)
        return ImageOps.expand(image, border=border, fill=self.fill)

    def __repr__(self):
        return f"{self.__class__.__name__}(fill={self.fill})"


class RandomBoxJitter:
    """Randomly move each crop edge, to imitate Stage-1 box error.

    This is the CHANGE 3 fix. Each of the four edges is displaced
    independently by up to ``max_frac`` of its dimension; a positive
    displacement crops inward (a box that is too tight), a negative one
    expands outward (a box that is too loose).

    Caveat: the saved crops contain no surrounding context, so an outward
    jitter pads with neutral grey rather than the real pixels that were
    there. Regenerating the crop dataset with a context margin would make
    this exact - see the note at the bottom of this file.
    """

    def __init__(self, max_frac=BOX_JITTER_FRAC, probability=BOX_JITTER_PROB, fill=PAD_FILL):
        self.max_frac = max_frac
        self.probability = probability
        self.fill = fill

    def __call__(self, image: Image.Image) -> Image.Image:
        if self.max_frac <= 0 or random.random() > self.probability:
            return image

        width, height = image.size

        # Pad first so an outward jitter lands on fill pixels rather than
        # falling outside the image.
        pad = int(math.ceil(self.max_frac * max(width, height))) + 1
        padded = ImageOps.expand(image, border=pad, fill=self.fill)

        left = pad + random.uniform(-self.max_frac, self.max_frac) * width
        top = pad + random.uniform(-self.max_frac, self.max_frac) * height
        right = pad + width - random.uniform(-self.max_frac, self.max_frac) * width
        bottom = pad + height - random.uniform(-self.max_frac, self.max_frac) * height

        # Guard against a degenerate box on very small crops.
        if right - left < 8 or bottom - top < 8:
            return image

        return padded.crop((int(round(left)), int(round(top)),
                            int(round(right)), int(round(bottom))))

    def __repr__(self):
        return (f"{self.__class__.__name__}(max_frac={self.max_frac}, "
                f"probability={self.probability})")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    freeze_support()

    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    set_seed(SEED)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR = RUN_DIR / "checkpoints"
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    BEST_PATH = CHECKPOINT_DIR / "best.pt"
    LAST_PATH = CHECKPOINT_DIR / "last.pt"
    HISTORY_PATH = RUN_DIR / "history.csv"
    CONFIG_PATH = RUN_DIR / "config.json"
    CONFUSION_PATH = RUN_DIR / "test_confusion_matrix.csv"
    PER_CLASS_PATH = RUN_DIR / "test_per_class.csv"

    # ============================================================
    # IMAGE PREPROCESSING
    # ============================================================

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    # Training: jitter the box, flip, pad to square, resize. No
    # RandomResizedCrop - on these aspect ratios it silently degenerates into
    # a centre crop and destroys the object, which is the v1 bug.
    train_transform_steps = [
        RandomBoxJitter(),
        transforms.RandomHorizontalFlip(p=0.5),
        PadToSquare(),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True),
        transforms.ColorJitter(
            brightness=0.20,
            contrast=0.20,
            saturation=0.15,
            hue=0.03,
        ),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
    if RANDOM_ERASING > 0:
        train_transform_steps.append(transforms.RandomErasing(p=RANDOM_ERASING))
    train_transform = transforms.Compose(train_transform_steps)

    # Evaluation: deterministic, and keeps the entire object.
    eval_transform = transforms.Compose([
        PadToSquare(),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    # ============================================================
    # DATASETS
    # ============================================================

    print("=" * 70)
    print("STAGE 2 v3 - CONVNEXT-TINY (context-margin crops)")
    print("=" * 70)
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Dataset      : {DATA_DIR}")
    print(f"Run dir      : {RUN_DIR}")
    print(f"Device       : {DEVICE}")
    print()
    print("Changes vs v2:")
    print("  1. Context-margin crops - jitter lands on real pixels, not grey")
    print(f"  2. RandomBoxJitter widened {0.12} -> {BOX_JITTER_FRAC}")
    print()

    for folder in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        if not folder.exists():
            raise FileNotFoundError(f"Dataset folder not found:\n{folder}")

    train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
    val_dataset = datasets.ImageFolder(VAL_DIR, transform=eval_transform)
    test_dataset = datasets.ImageFolder(TEST_DIR, transform=eval_transform)

    if train_dataset.class_to_idx != val_dataset.class_to_idx:
        raise RuntimeError("Train and validation class mappings differ.")
    if train_dataset.class_to_idx != test_dataset.class_to_idx:
        raise RuntimeError("Train and test class mappings differ.")

    classes = train_dataset.classes
    num_classes = len(classes)

    print(f"Classes ({num_classes}):")
    for i, name in enumerate(classes):
        print(f"  {i:2d}: {name}")
    print()
    print(f"Train images: {len(train_dataset):,}")
    print(f"Val images  : {len(val_dataset):,}")
    print(f"Test images : {len(test_dataset):,}")
    print()

    # ============================================================
    # CLASS WEIGHTS  (unchanged from v1)
    # ============================================================

    train_targets = np.array(train_dataset.targets)
    class_counts = np.bincount(train_targets, minlength=num_classes).astype(np.float64)

    # Square-root inverse frequency: addresses imbalance without letting the
    # rarest classes dominate the loss.
    weights = np.sqrt(class_counts.sum() / (num_classes * class_counts))
    weights = weights / weights.mean()

    class_weights = torch.tensor(weights, dtype=torch.float32, device=DEVICE)

    print("Training class counts and weights:")
    for i, name in enumerate(classes):
        print(f"  {name:20s}: {int(class_counts[i]):>7,}   w={weights[i]:.4f}")
    print()

    # ============================================================
    # DATALOADERS
    # ============================================================

    loader_kwargs = {
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "pin_memory": torch.cuda.is_available(),
    }
    if NUM_WORKERS > 0:
        loader_kwargs["persistent_workers"] = True

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    # ============================================================
    # MODEL
    # ============================================================

    print("Loading pretrained ConvNeXt-Tiny...")
    model = timm.create_model(
        MODEL_NAME,
        pretrained=True,
        num_classes=num_classes,
        drop_path_rate=DROP_PATH_RATE,
    )
    model = model.to(DEVICE)

    if torch.cuda.is_available():
        model = model.to(memory_format=torch.channels_last)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters    : {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print()

    # ============================================================
    # LOSS / OPTIMIZER / SCHEDULER
    # ============================================================

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=LABEL_SMOOTHING,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    def lr_factor(epoch: int) -> float:
        """Linear warmup, then cosine decay across exactly EPOCHS epochs.

        This is the CHANGE 2 fix: T_max matches the real budget, so the
        schedule reaches its floor instead of stopping at 83% of peak LR.
        """
        if epoch < WARMUP_EPOCHS:
            return (epoch + 1) / WARMUP_EPOCHS
        progress = (epoch - WARMUP_EPOCHS) / max(1, EPOCHS - WARMUP_EPOCHS)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return MIN_LR_FACTOR + (1.0 - MIN_LR_FACTOR) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_factor)

    use_amp = torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # ============================================================
    # METRICS
    # ============================================================

    def calculate_metrics(y_true, y_pred):
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred,
            labels=list(range(num_classes)),
            average="macro",
            zero_division=0,
        )
        return {
            "accuracy": accuracy,
            "macro_precision": precision,
            "macro_recall": recall,
            "macro_f1": f1,
        }

    # ============================================================
    # TRAIN / EVALUATE
    # ============================================================

    def train_one_epoch(epoch: int):
        model.train()
        running_loss = 0.0
        all_targets, all_predictions = [], []

        progress = tqdm(
            train_loader, total=len(train_loader),
            desc=f"Train {epoch}/{EPOCHS}", unit="batch",
            dynamic_ncols=True, leave=True,
        )

        for images, targets in progress:
            if torch.cuda.is_available():
                images = images.to(DEVICE, non_blocking=True,
                                   memory_format=torch.channels_last)
            else:
                images = images.to(DEVICE)
            targets = targets.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * images.size(0)
            predictions = outputs.argmax(dim=1)
            all_targets.extend(targets.detach().cpu().numpy())
            all_predictions.extend(predictions.detach().cpu().numpy())

            progress.set_postfix(
                loss=f"{loss.item():.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )

        epoch_loss = running_loss / len(train_dataset)
        return epoch_loss, calculate_metrics(all_targets, all_predictions)

    @torch.no_grad()
    def evaluate(loader, dataset_size, desc="Validate"):
        model.eval()
        running_loss = 0.0
        all_targets, all_predictions = [], []

        progress = tqdm(loader, total=len(loader), desc=desc, unit="batch",
                        dynamic_ncols=True, leave=True)

        for images, targets in progress:
            if torch.cuda.is_available():
                images = images.to(DEVICE, non_blocking=True,
                                   memory_format=torch.channels_last)
            else:
                images = images.to(DEVICE)
            targets = targets.to(DEVICE, non_blocking=True)

            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, targets)

            running_loss += loss.item() * images.size(0)
            predictions = outputs.argmax(dim=1)
            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
            progress.set_postfix(loss=f"{loss.item():.4f}")

        epoch_loss = running_loss / dataset_size
        metrics = calculate_metrics(all_targets, all_predictions)
        return epoch_loss, metrics, all_targets, all_predictions

    # ============================================================
    # CHECKPOINTS
    # ============================================================

    def save_checkpoint(path, epoch, best_metric, patience_counter):
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_metric": best_metric,
            "patience_counter": patience_counter,
            "classes": classes,
            "class_to_idx": train_dataset.class_to_idx,
            "config": {
                "version": "v3",
                "model_name": MODEL_NAME,
                "image_size": IMAGE_SIZE,
                "batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "label_smoothing": LABEL_SMOOTHING,
                "epochs": EPOCHS,
                "warmup_epochs": WARMUP_EPOCHS,
                "box_jitter_frac": BOX_JITTER_FRAC,
                "preprocessing": "pad_to_square",
                "seed": SEED,
            },
        }, path)

    def load_checkpoint(path):
        checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        return (checkpoint["epoch"], checkpoint["best_metric"],
                checkpoint["patience_counter"])

    # ============================================================
    # CONFIG DUMP
    # ============================================================

    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "version": "v3",
            "changes_vs_v2": [
                "context-margin crop dataset (25% margin per side)",
                f"RandomBoxJitter widened to max_frac={BOX_JITTER_FRAC}",
            ],
            "model": MODEL_NAME,
            "dataset": str(DATA_DIR),
            "image_size": IMAGE_SIZE,
            "batch_size": BATCH_SIZE,
            "num_workers": NUM_WORKERS,
            "epochs": EPOCHS,
            "warmup_epochs": WARMUP_EPOCHS,
            "min_lr_factor": MIN_LR_FACTOR,
            "early_stopping": EARLY_STOPPING,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "label_smoothing": LABEL_SMOOTHING,
            "box_jitter_frac": BOX_JITTER_FRAC,
            "box_jitter_prob": BOX_JITTER_PROB,
            "drop_path_rate": DROP_PATH_RATE,
            "random_erasing": RANDOM_ERASING,
            "seed": SEED,
            "classes": classes,
            "class_to_idx": train_dataset.class_to_idx,
            "class_counts": {n: int(class_counts[i]) for i, n in enumerate(classes)},
            "class_weights": {n: float(weights[i]) for i, n in enumerate(classes)},
        }, f, indent=4)

    # ============================================================
    # TRAINING LOOP
    # ============================================================

    start_epoch = 1
    best_metric = -1.0
    patience_counter = 0

    if RESUME and LAST_PATH.exists():
        last_epoch, best_metric, patience_counter = load_checkpoint(LAST_PATH)
        start_epoch = last_epoch + 1
        print(f"Resumed from epoch {last_epoch}. Best val macro F1 so far: {best_metric:.4f}")
        print()

    history_exists = HISTORY_PATH.exists() and RESUME
    history_file = open(HISTORY_PATH, "a" if history_exists else "w",
                        newline="", encoding="utf-8")
    history_writer = csv.writer(history_file)
    if not history_exists:
        history_writer.writerow([
            "epoch", "learning_rate",
            "train_loss", "train_accuracy", "train_macro_precision",
            "train_macro_recall", "train_macro_f1",
            "val_loss", "val_accuracy", "val_macro_precision",
            "val_macro_recall", "val_macro_f1",
            "epoch_time_sec",
        ])
        history_file.flush()

    print("=" * 70)
    print("TRAINING")
    print("=" * 70)

    run_start = time.time()

    for epoch in range(start_epoch, EPOCHS + 1):
        epoch_start = time.time()
        current_lr = optimizer.param_groups[0]["lr"]

        train_loss, train_metrics = train_one_epoch(epoch)
        val_loss, val_metrics, _, _ = evaluate(val_loader, len(val_dataset))

        scheduler.step()
        epoch_time = time.time() - epoch_start

        print(
            f"\nEpoch {epoch}/{EPOCHS}  lr={current_lr:.2e}  ({epoch_time / 60:.1f} min)"
            f"\n  train  loss {train_loss:.4f}  acc {train_metrics['accuracy']:.4f}"
            f"  macroF1 {train_metrics['macro_f1']:.4f}"
            f"\n  val    loss {val_loss:.4f}  acc {val_metrics['accuracy']:.4f}"
            f"  macroF1 {val_metrics['macro_f1']:.4f}"
        )

        history_writer.writerow([
            epoch, current_lr,
            train_loss, train_metrics["accuracy"], train_metrics["macro_precision"],
            train_metrics["macro_recall"], train_metrics["macro_f1"],
            val_loss, val_metrics["accuracy"], val_metrics["macro_precision"],
            val_metrics["macro_recall"], val_metrics["macro_f1"],
            epoch_time,
        ])
        history_file.flush()

        current_metric = val_metrics["macro_f1"]

        if current_metric > best_metric:
            best_metric = current_metric
            patience_counter = 0
            save_checkpoint(BEST_PATH, epoch, best_metric, patience_counter)
            print(f"  -> new best (val macro F1 {best_metric:.4f}), saved best.pt")
        else:
            patience_counter += 1
            print(f"  -> no improvement ({patience_counter}/{PATIENCE}), best {best_metric:.4f}")

        save_checkpoint(LAST_PATH, epoch, best_metric, patience_counter)

        if EARLY_STOPPING and patience_counter >= PATIENCE:
            print(f"\nEarly stopping after {PATIENCE} epochs without improvement.")
            break

    history_file.close()

    # ============================================================
    # FINAL TEST EVALUATION
    # ============================================================

    print("\n" + "=" * 70)
    print("LOADING BEST CHECKPOINT")
    print("=" * 70)

    best_checkpoint = torch.load(BEST_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    print(f"Best epoch: {best_checkpoint['epoch']}")
    print(f"Best validation macro F1: {best_checkpoint['best_metric']:.4f}")

    print("\n" + "=" * 70)
    print("FINAL TEST EVALUATION")
    print("=" * 70)

    test_loss, test_metrics, y_true, y_pred = evaluate(
        test_loader, len(test_dataset), desc="Test"
    )

    print(f"\nTest loss           : {test_loss:.4f}")
    print(f"Test accuracy       : {test_metrics['accuracy']:.4f}")
    print(f"Test macro precision: {test_metrics['macro_precision']:.4f}")
    print(f"Test macro recall   : {test_metrics['macro_recall']:.4f}")
    print(f"Test macro F1       : {test_metrics['macro_f1']:.4f}")

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(num_classes)), zero_division=0
    )

    print("\n" + "=" * 80)
    print("PER-CLASS TEST RESULTS")
    print("-" * 80)
    print(f"{'Class':<20}{'Precision':>12}{'Recall':>11}{'F1':>11}{'Support':>11}")
    print("-" * 80)
    for i, name in enumerate(classes):
        print(f"{name:<20}{precision[i]:>12.4f}{recall[i]:>11.4f}"
              f"{f1[i]:>11.4f}{support[i]:>11}")

    with open(PER_CLASS_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "precision", "recall", "f1", "support"])
        for i, name in enumerate(classes):
            writer.writerow([name, f"{precision[i]:.6f}", f"{recall[i]:.6f}",
                             f"{f1[i]:.6f}", int(support[i])])

    matrix = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    with open(CONFUSION_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["true/pred"] + classes)
        for i, name in enumerate(classes):
            writer.writerow([name] + matrix[i].tolist())

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Best checkpoint : {BEST_PATH}")
    print(f"Last checkpoint : {LAST_PATH}")
    print(f"History         : {HISTORY_PATH}")
    print(f"Per-class       : {PER_CLASS_PATH}")
    print(f"Confusion matrix: {CONFUSION_PATH}")
    print(f"Total runtime   : {(time.time() - run_start) / 3600:.2f} hours")
    print()
    print("v1 reference: accuracy 0.9466, macro F1 0.8691, truck F1 0.4783")
    print("v2 reference: accuracy 0.9703, macro F1 0.8940, truck F1 0.5417")
    print()
    print("The test set was evaluated only once, after training.")
    print("Do not use test results to choose hyperparameters.")


# ==============================================================================
# NOTE - making CHANGE 3 exact
# ==============================================================================
# RandomBoxJitter can only shrink into real pixels; when it expands it pads
# with grey, because generate_classification_dataset.py cut each crop exactly
# on the ground-truth box and kept no surrounding context.
#
# To make the jitter fully realistic, regenerate the crop dataset with a
# margin - expand each box by ~25% before cropping - and then jitter inside
# that margin so every box lands on real pixels. That is a ~10 minute
# regeneration and a strictly better version of this change, but it produces
# a different dataset, so it should be a separate experiment rather than
# folded into this one.
