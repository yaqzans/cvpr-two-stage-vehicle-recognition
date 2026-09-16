"""09 - Throughput benchmark for the two-stage pipeline.

Script 06 measures *streaming latency*: one frame in, one result out,
batch = 1. That is the right number for a live camera. It is the wrong number
for processing recorded video, where frames can be batched and the per-call
Python overhead amortised.

This measures both, so the real-time claim can be stated precisely:

    latency mode     batch = 1, the live-camera number
    throughput mode  batched detector + batched classifier, the offline number

No accuracy is computed here - 06 already did that, and these produce
identical predictions. This only times them.

    python scripts/09_benchmark_throughput.py
    python scripts/09_benchmark_throughput.py --batches 1 4 8 16
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT_ROOT / ".ultralytics"))

import cv2
import numpy as np
import timm
import torch
from ultralytics import YOLO

import importlib.util

_spec = importlib.util.spec_from_file_location(
    "pipeline_eval", Path(__file__).with_name("06_evaluate_pipeline.py"))
_pipeline = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_pipeline)
fast_pad_resize = _pipeline.fast_pad_resize
normalize_batch = _pipeline.normalize_batch


def main() -> None:
    parser = argparse.ArgumentParser(description="Two-stage pipeline throughput benchmark.")
    parser.add_argument("--det-weights", type=Path,
                        default=PROJECT_ROOT / "runs/stage1_yolo26n/train/weights/best.pt")
    parser.add_argument("--cls-checkpoint", type=Path,
                        default=PROJECT_ROOT / "runs/stage2_convnext_tiny_v2/checkpoints/best.pt")
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--frames", type=int, default=200)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 4, 8, 16])
    parser.add_argument("--out", type=Path,
                        default=PROJECT_ROOT / "runs" / "pipeline_eval" / "throughput.json")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    detector = YOLO(str(args.det_weights))

    checkpoint = torch.load(args.cls_checkpoint, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})
    classifier = timm.create_model(
        config.get("model_name", "hf_hub:timm/convnext_tiny.in12k_ft_in1k"),
        pretrained=False, num_classes=len(checkpoint["classes"]))
    classifier.load_state_dict(checkpoint["model_state_dict"])
    classifier.eval().to(device)
    if device == "cuda":
        classifier = classifier.to(memory_format=torch.channels_last)

    images_dir = PROJECT_ROOT / "Datasets" / "rsud20k" / "images" / "test"
    files = sorted(images_dir.glob("*.jpg"))[:args.frames]

    print("=" * 68)
    print("THROUGHPUT BENCHMARK")
    print("=" * 68)
    print(f"Device : {torch.cuda.get_device_name(0) if device == 'cuda' else 'cpu'}")
    print(f"Frames : {len(files)}  (1920x1080)   conf={args.conf}  imgsz={args.imgsz}")
    print("\nDecoding frames into RAM (decode is measured separately)...")

    frames = [cv2.imread(str(f)) for f in files]
    frames = [f for f in frames if f is not None]

    def sync():
        if device == "cuda":
            torch.cuda.synchronize()

    # ---- measure JPEG decode separately so it can be attributed properly
    t0 = time.perf_counter()
    for f in files[:50]:
        cv2.imread(str(f))
    decode_ms = (time.perf_counter() - t0) / 50 * 1000

    # ---- warmup
    for _ in range(3):
        detector.predict(source=frames[0], conf=args.conf, imgsz=args.imgsz,
                         device=0 if device == "cuda" else "cpu", verbose=False)
        dummy = torch.zeros(8, 3, 224, 224, device=device)
        if device == "cuda":
            dummy = dummy.to(memory_format=torch.channels_last)
        with torch.no_grad(), torch.autocast("cuda", torch.float16, enabled=device == "cuda"):
            classifier(dummy)
    sync()

    results = {}

    for batch_size in args.batches:
        sync()
        start = time.perf_counter()
        detect_time = classify_time = crop_time = 0.0
        n_crops = 0

        for offset in range(0, len(frames), batch_size):
            chunk = frames[offset:offset + batch_size]

            t0 = time.perf_counter()
            outputs = detector.predict(source=chunk, conf=args.conf, imgsz=args.imgsz,
                                       device=0 if device == "cuda" else "cpu", verbose=False)
            sync(); detect_time += time.perf_counter() - t0

            t0 = time.perf_counter()
            crops = []
            for image, output in zip(chunk, outputs):
                if output.boxes is None or len(output.boxes) == 0:
                    continue
                height, width = image.shape[:2]
                for box in output.boxes.xyxy.cpu().numpy():
                    x1, y1 = max(0, int(box[0])), max(0, int(box[1]))
                    x2, y2 = min(width, int(box[2])), min(height, int(box[3]))
                    if x2 - x1 < 2 or y2 - y1 < 2:
                        continue
                    crops.append(fast_pad_resize(image[y1:y2, x1:x2]))
            sync(); crop_time += time.perf_counter() - t0
            n_crops += len(crops)

            if crops:
                t0 = time.perf_counter()
                for start_index in range(0, len(crops), 128):
                    batch = normalize_batch(np.stack(crops[start_index:start_index + 128]), device)
                    if device == "cuda":
                        batch = batch.to(memory_format=torch.channels_last)
                    with torch.no_grad(), torch.autocast("cuda", torch.float16,
                                                         enabled=device == "cuda"):
                        classifier(batch).float().softmax(1)
                sync(); classify_time += time.perf_counter() - t0

        total = time.perf_counter() - start
        n = len(frames)
        results[batch_size] = {
            "detect_ms": detect_time / n * 1000,
            "crop_ms": crop_time / n * 1000,
            "classify_ms": classify_time / n * 1000,
            "gpu_ms": total / n * 1000,
            "with_decode_ms": total / n * 1000 + decode_ms,
            "fps_gpu": n / total,
            "fps_with_decode": 1000 / (total / n * 1000 + decode_ms),
            "crops_per_frame": n_crops / n,
        }

    print(f"\nJPEG decode: {decode_ms:.2f} ms/frame (CPU, parallelisable)\n")
    print(f"{'batch':>6}{'detect':>9}{'crop':>8}{'classify':>10}{'GPU total':>11}"
          f"{'FPS (GPU)':>11}{'FPS (+decode)':>15}")
    print("-" * 70)
    for batch_size, r in results.items():
        print(f"{batch_size:>6}{r['detect_ms']:>9.2f}{r['crop_ms']:>8.2f}"
              f"{r['classify_ms']:>10.2f}{r['gpu_ms']:>11.2f}"
              f"{r['fps_gpu']:>11.1f}{r['fps_with_decode']:>15.1f}")

    best = max(results.items(), key=lambda kv: kv[1]["fps_gpu"])
    print(f"\nBest: batch={best[0]}  {best[1]['fps_gpu']:.1f} FPS on GPU, "
          f"{best[1]['fps_with_decode']:.1f} FPS including JPEG decode")
    print(f"Streaming (batch=1) latency: {results[1]['gpu_ms']:.1f} ms "
          f"-> {results[1]['fps_gpu']:.1f} FPS")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({
            "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
            "frames": len(frames), "conf": args.conf, "imgsz": args.imgsz,
            "jpeg_decode_ms": decode_ms,
            "by_batch_size": {str(k): v for k, v in results.items()},
        }, f, indent=4)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
