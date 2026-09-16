"""
Stage 2 - ConvNeXt-Tiny training for BangladeshVehicleClassification

Dataset:
    Project1/Datasets/BangladeshVehicleClassification/
        train/<class>/*.jpg
        val/<class>/*.jpg
        test/<class>/*.jpg

The original RSUD20K train/val/test split is preserved.

Model:
    timm/convnext_tiny.in12k_ft_in1k
    Pretrained on ImageNet-12k, fine-tuned on ImageNet-1k.

Features:
    - 13-class ImageFolder dataset
    - ImageNet normalization
    - Training augmentation
    - Class-weighted CrossEntropyLoss for class imbalance
    - Mixed precision (AMP)
    - AdamW + cosine learning-rate schedule
    - Early stopping
    - Saves best.pt and last.pt
    - Resume support
    - CSV training history
    - Validation accuracy + macro precision/recall/F1
    - Saves best model based on validation macro F1
    - Final test evaluation using the best checkpoint
"""

from pathlib import Path
import csv
import json
import time
import random
from multiprocessing import freeze_support

from tqdm import tqdm
import numpy as np
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

# Project root is the folder containing this script's parent.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "Datasets" / "BangladeshVehicleClassification"
RUN_DIR = PROJECT_ROOT / "runs" / "stage2_convnext_tiny"

TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
TEST_DIR = DATA_DIR / "test"

# Hugging Face / timm pretrained ConvNeXt-Tiny.
MODEL_NAME = "hf_hub:timm/convnext_tiny.in12k_ft_in1k"

IMAGE_SIZE = 224

BATCH_SIZE = 32
NUM_WORKERS = 4  # Windows DataLoader workers; main guard below is required.

EPOCHS = 100
PATIENCE = 20

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

# Label smoothing helps reduce overconfidence.
LABEL_SMOOTHING = 0.1

SEED = 42

# Set True if you want to continue from RUN_DIR/checkpoints/last.pt.
RESUME = False

# Use GPU if available.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# REPRODUCIBILITY
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


    # ============================================================
    # DIRECTORIES
    # ============================================================

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR = RUN_DIR / "checkpoints"
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    BEST_PATH = CHECKPOINT_DIR / "best.pt"
    LAST_PATH = CHECKPOINT_DIR / "last.pt"
    HISTORY_PATH = RUN_DIR / "history.csv"
    CONFIG_PATH = RUN_DIR / "config.json"
    CONFUSION_PATH = RUN_DIR / "test_confusion_matrix.csv"


    # ============================================================
    # IMAGE PREPROCESSING
    # ============================================================

    # ConvNeXt pretrained weights use standard ImageNet normalization.
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    # Training:
    # RandomResizedCrop provides scale/position variation while
    # keeping the model input fixed at 224x224.
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(
            IMAGE_SIZE,
            scale=(0.70, 1.0),
            ratio=(0.75, 1.3333),
            antialias=True,
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=0.20,
            contrast=0.20,
            saturation=0.15,
            hue=0.03,
        ),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    # Validation/test:
    # No random augmentation. Resize and center crop give deterministic
    # evaluation.
    eval_transform = transforms.Compose([
        transforms.Resize(
            int(IMAGE_SIZE * 256 / 224),
            antialias=True,
        ),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


    # ============================================================
    # DATASETS
    # ============================================================

    print("=" * 70)
    print("STAGE 2 - CONVNEXT-TINY")
    print("=" * 70)

    print(f"Project root : {PROJECT_ROOT}")
    print(f"Dataset      : {DATA_DIR}")
    print(f"Device       : {DEVICE}")
    print()

    for folder in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        if not folder.exists():
            raise FileNotFoundError(
                f"Dataset folder not found:\n{folder}"
            )

    train_dataset = datasets.ImageFolder(
        TRAIN_DIR,
        transform=train_transform,
    )

    val_dataset = datasets.ImageFolder(
        VAL_DIR,
        transform=eval_transform,
    )

    test_dataset = datasets.ImageFolder(
        TEST_DIR,
        transform=eval_transform,
    )

    # Make sure all splits use exactly the same class mapping.
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
    # CLASS COUNTS / CLASS WEIGHTS
    # ============================================================

    # ImageFolder stores target labels in train_dataset.targets.
    train_targets = np.array(train_dataset.targets)

    class_counts = np.bincount(
        train_targets,
        minlength=num_classes,
    ).astype(np.float64)

    print("Training class counts:")
    for i, name in enumerate(classes):
        print(f"  {name:20s}: {int(class_counts[i]):,}")

    # Use square-root inverse-frequency weighting rather than full
    # inverse-frequency weighting. This addresses imbalance without
    # letting the rarest classes dominate the loss.
    weights = np.sqrt(
        class_counts.sum() / (num_classes * class_counts)
    )

    # Normalize weights so the average weight is approximately 1.
    weights = weights / weights.mean()

    class_weights = torch.tensor(
        weights,
        dtype=torch.float32,
        device=DEVICE,
    )

    print()
    print("Class weights:")
    for i, name in enumerate(classes):
        print(f"  {name:20s}: {weights[i]:.4f}")

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

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        **loader_kwargs,
    )

    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **loader_kwargs,
    )

    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        **loader_kwargs,
    )


    # ============================================================
    # MODEL
    # ============================================================

    print("Loading pretrained ConvNeXt-Tiny...")
    print("If this is the first run, pretrained weights may download from Hugging Face.")
    print("You will see download progress below if a download is required.")
    print()

    model = timm.create_model(
        MODEL_NAME,
        pretrained=True,
        num_classes=num_classes,
    )

    model = model.to(DEVICE)

    # Channels-last can improve convolution performance on CUDA.
    if torch.cuda.is_available():
        model = model.to(memory_format=torch.channels_last)

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

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

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
        eta_min=1e-6,
    )

    # AMP for RTX 4060 Ti.
    use_amp = torch.cuda.is_available()

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp,
    )


    # ============================================================
    # METRICS
    # ============================================================

    def calculate_metrics(y_true, y_pred):
        accuracy = accuracy_score(y_true, y_pred)

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
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
    # TRAIN ONE EPOCH
    # ============================================================

    def train_one_epoch():
        model.train()

        running_loss = 0.0
        all_targets = []
        all_predictions = []

        progress = tqdm(
            train_loader,
            total=len(train_loader),
            desc=f"Train",
            unit="batch",
            dynamic_ncols=True,
            leave=True,
        )

        for images, targets in progress:

            if torch.cuda.is_available():
                images = images.to(
                    DEVICE,
                    non_blocking=True,
                    memory_format=torch.channels_last,
                )
            else:
                images = images.to(DEVICE)

            targets = targets.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=use_amp,
            ):
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
                gpu="ON" if torch.cuda.is_available() else "CPU",
            )

        epoch_loss = running_loss / len(train_dataset)

        metrics = calculate_metrics(
            all_targets,
            all_predictions,
        )

        return epoch_loss, metrics


    # ============================================================
    # VALIDATE
    # ============================================================

    @torch.no_grad()
    def evaluate(loader, dataset_size):
        model.eval()

        running_loss = 0.0
        all_targets = []
        all_predictions = []

        progress = tqdm(
            loader,
            total=len(loader),
            desc="Validate",
            unit="batch",
            dynamic_ncols=True,
            leave=True,
        )

        for images, targets in progress:

            if torch.cuda.is_available():
                images = images.to(
                    DEVICE,
                    non_blocking=True,
                    memory_format=torch.channels_last,
                )
            else:
                images = images.to(DEVICE)

            targets = targets.to(DEVICE, non_blocking=True)

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=use_amp,
            ):
                outputs = model(images)
                loss = criterion(outputs, targets)

            running_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)

            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

            progress.set_postfix(
                loss=f"{loss.item():.4f}",
            )

        epoch_loss = running_loss / dataset_size

        metrics = calculate_metrics(
            all_targets,
            all_predictions,
        )

        return epoch_loss, metrics, all_targets, all_predictions


    # ============================================================
    # CHECKPOINT HELPERS
    # ============================================================

    def save_checkpoint(path, epoch, best_metric, patience_counter):
        checkpoint = {
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
                "model_name": MODEL_NAME,
                "image_size": IMAGE_SIZE,
                "batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "label_smoothing": LABEL_SMOOTHING,
                "epochs": EPOCHS,
                "seed": SEED,
            },
        }

        torch.save(checkpoint, path)


    def load_checkpoint(path):
        checkpoint = torch.load(
            path,
            map_location=DEVICE,
            weights_only=False,
        )

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

        return (
            checkpoint["epoch"],
            checkpoint["best_metric"],
            checkpoint["patience_counter"],
        )


    # ============================================================
    # SAVE CONFIG
    # ============================================================

    config = {
        "model": MODEL_NAME,
        "dataset": str(DATA_DIR),
        "image_size": IMAGE_SIZE,
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "epochs": EPOCHS,
        "patience": PATIENCE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "label_smoothing": LABEL_SMOOTHING,
        "seed": SEED,
        "classes": classes,
        "class_to_idx": train_dataset.class_to_idx,
        "class_counts": {
            classes[i]: int(class_counts[i])
            for i in range(num_classes)
        },
        "class_weights": {
            classes[i]: float(weights[i])
            for i in range(num_classes)
        },
    }

    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)


    # ============================================================
    # RESUME
    # ============================================================

    start_epoch = 0
    best_metric = -1.0
    patience_counter = 0

    if RESUME and LAST_PATH.exists():
        print(f"Resuming from: {LAST_PATH}")

        (
            previous_epoch,
            best_metric,
            patience_counter,
        ) = load_checkpoint(LAST_PATH)

        start_epoch = previous_epoch + 1

        print(f"Resuming at epoch {start_epoch + 1}")
        print(f"Best validation macro F1: {best_metric:.4f}")
        print()


    # ============================================================
    # HISTORY FILE
    # ============================================================

    if not HISTORY_PATH.exists() or start_epoch == 0:

        with open(
            HISTORY_PATH,
            "w",
            newline="",
            encoding="utf-8",
        ) as f:

            writer = csv.writer(f)

            writer.writerow([
                "epoch",
                "learning_rate",
                "train_loss",
                "train_accuracy",
                "train_macro_precision",
                "train_macro_recall",
                "train_macro_f1",
                "val_loss",
                "val_accuracy",
                "val_macro_precision",
                "val_macro_recall",
                "val_macro_f1",
                "epoch_time_sec",
            ])


    # ============================================================
    # TRAINING LOOP
    # ============================================================

    print("=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)

    training_start = time.time()

    for epoch in range(start_epoch, EPOCHS):

        epoch_start = time.time()

        print()
        print("-" * 70)
        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"{len(train_loader):,} training batches"
        )

        current_lr = optimizer.param_groups[0]["lr"]

        train_loss, train_metrics = train_one_epoch()

        val_loss, val_metrics, _, _ = evaluate(
            val_loader,
            len(val_dataset),
        )

        scheduler.step()

        epoch_time = time.time() - epoch_start

        print(
            f"LR: {current_lr:.7f}"
        )

        print(
            f"Train | "
            f"Loss: {train_loss:.4f} | "
            f"Acc: {train_metrics['accuracy']:.4f} | "
            f"Macro F1: {train_metrics['macro_f1']:.4f}"
        )

        print(
            f"Val   | "
            f"Loss: {val_loss:.4f} | "
            f"Acc: {val_metrics['accuracy']:.4f} | "
            f"Macro F1: {val_metrics['macro_f1']:.4f}"
        )

        print(
            f"Time: {epoch_time / 60:.2f} min"
        )

        if torch.cuda.is_available():
            gpu_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)
            print(
                f"Peak GPU memory: {gpu_mem:.2f} GB"
            )
            torch.cuda.reset_peak_memory_stats()

        # Save history.
        with open(
            HISTORY_PATH,
            "a",
            newline="",
            encoding="utf-8",
        ) as f:

            writer = csv.writer(f)

            writer.writerow([
                epoch + 1,
                current_lr,
                train_loss,
                train_metrics["accuracy"],
                train_metrics["macro_precision"],
                train_metrics["macro_recall"],
                train_metrics["macro_f1"],
                val_loss,
                val_metrics["accuracy"],
                val_metrics["macro_precision"],
                val_metrics["macro_recall"],
                val_metrics["macro_f1"],
                epoch_time,
            ])

        # Always save last checkpoint.
        save_checkpoint(
            LAST_PATH,
            epoch,
            best_metric,
            patience_counter,
        )

        # Best model is selected using validation macro F1.
        current_metric = val_metrics["macro_f1"]

        if current_metric > best_metric:

            best_metric = current_metric
            patience_counter = 0

            save_checkpoint(
                BEST_PATH,
                epoch,
                best_metric,
                patience_counter,
            )

            print(
                f"*** New best model! "
                f"Val Macro F1 = {best_metric:.4f}"
            )

        else:

            patience_counter += 1

            print(
                f"No improvement. "
                f"Patience: {patience_counter}/{PATIENCE}"
            )

        # Update last checkpoint again with the latest patience state.
        save_checkpoint(
            LAST_PATH,
            epoch,
            best_metric,
            patience_counter,
        )

        if patience_counter >= PATIENCE:

            print()
            print(
                f"Early stopping triggered after "
                f"{PATIENCE} epochs without improvement."
            )

            break


    # ============================================================
    # LOAD BEST MODEL
    # ============================================================

    print()
    print("=" * 70)
    print("LOADING BEST CHECKPOINT")
    print("=" * 70)

    best_checkpoint = torch.load(
        BEST_PATH,
        map_location=DEVICE,
        weights_only=False,
    )

    model.load_state_dict(
        best_checkpoint["model_state_dict"]
    )

    print(
        f"Best epoch: {best_checkpoint['epoch'] + 1}"
    )

    print(
        f"Best validation macro F1: "
        f"{best_checkpoint['best_metric']:.4f}"
    )


    # ============================================================
    # FINAL TEST EVALUATION
    # ============================================================

    print()
    print("=" * 70)
    print("FINAL TEST EVALUATION")
    print("=" * 70)

    test_loss, test_metrics, y_true, y_pred = evaluate(
        test_loader,
        len(test_dataset),
    )

    print(
        f"Test loss          : {test_loss:.4f}"
    )

    print(
        f"Test accuracy      : "
        f"{test_metrics['accuracy']:.4f}"
    )

    print(
        f"Test macro precision: "
        f"{test_metrics['macro_precision']:.4f}"
    )

    print(
        f"Test macro recall   : "
        f"{test_metrics['macro_recall']:.4f}"
    )

    print(
        f"Test macro F1       : "
        f"{test_metrics['macro_f1']:.4f}"
    )


    # ============================================================
    # PER-CLASS TEST METRICS
    # ============================================================

    precision, recall, f1, support = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=list(range(num_classes)),
            zero_division=0,
        )
    )

    print()
    print("PER-CLASS TEST RESULTS")
    print("-" * 80)

    print(
        f"{'Class':20s} "
        f"{'Precision':>10s} "
        f"{'Recall':>10s} "
        f"{'F1':>10s} "
        f"{'Support':>10s}"
    )

    print("-" * 80)

    for i, name in enumerate(classes):

        print(
            f"{name:20s} "
            f"{precision[i]:10.4f} "
            f"{recall[i]:10.4f} "
            f"{f1[i]:10.4f} "
            f"{support[i]:10d}"
        )


    # ============================================================
    # CONFUSION MATRIX
    # ============================================================

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(num_classes)),
    )

    with open(
        CONFUSION_PATH,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.writer(f)

        writer.writerow(["true/pred"] + classes)

        for i, row in enumerate(cm):

            writer.writerow(
                [classes[i]] + row.tolist()
            )


    # ============================================================
    # FINISHED
    # ============================================================

    total_time = time.time() - training_start

    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    print(
        f"Best checkpoint : {BEST_PATH}"
    )

    print(
        f"Last checkpoint : {LAST_PATH}"
    )

    print(
        f"History         : {HISTORY_PATH}"
    )

    print(
        f"Confusion matrix: {CONFUSION_PATH}"
    )

    print(
        f"Total runtime   : {total_time / 3600:.2f} hours"
    )

    print()
    print("IMPORTANT:")
    print("The test set was evaluated only once, after training.")
    print("Do not use test results to choose hyperparameters.")
