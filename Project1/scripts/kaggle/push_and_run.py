"""Push, poll, pull, and auto-resume a Stage-1 training notebook on Kaggle.

Built around the field manual's rules (scripts/kaggle/KAGGLE_MANUAL.md-derived
lessons), specifically:
    - internet is OFF; everything the notebook needs is attached as a dataset
      (RSUD20K, offline ultralytics wheels, and later the checkpoint dataset)
    - the T4 accelerator is forced with the CLI flag, since the metadata
      field is silently ignored and the API default (P100, CUDA capability
      6.0) can't run this PyTorch build
    - the CLI exits 0 even when a push is REJECTED (e.g. concurrent GPU
      session cap) — the literal "successfully pushed" string is checked,
      never the exit code
    - kernel status is matched case-insensitively (Kaggle returns
      "KernelWorkerStatus.ERROR", not "error")
    - output is pulled into a fresh directory every cycle, since `kernels
      output` does not reliably overwrite stale files
    - the notebook is validated (ast.parse on every cell, reject !pip/!apt)
      before every push, so a syntax error never costs GPU quota

Usage:
    KAGGLE_API_TOKEN=$(cat ~/.kaggle/access_token) python scripts/kaggle/push_and_run.py \\
        --model yolo11m.pt --run-name stage1-yolo11m

Re-running the same command continues an interrupted loop from its saved
checkpoint dataset.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_notebook import build as build_notebook  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STATE_DIR = PROJECT_ROOT / "runs" / "kaggle_state"

RSUD20K_DATASET = "hasibzunair/rsud20k-bangladesh-road-scene-understanding"
WHEELS_DATASET = "fairoozalammahi/ultralytics-offline-wheels"
WEIGHTS_DATASET = "fairoozalammahi/yolo-pretrained-weights"
POLL_INTERVAL_SECONDS = 120
ACCELERATOR = "NvidiaTeslaT4"


def kaggle(*args: str, capture: bool = True) -> subprocess.CompletedProcess:
    # -X utf8: the Kaggle CLI prints a checkmark (U+2705) on some commands,
    # which crashes on Windows' default console codepage otherwise.
    cmd = [sys.executable, "-X", "utf8", "-m", "kaggle", *args]
    print(f"$ {' '.join(cmd)}")
    return subprocess.run(cmd, capture_output=capture, text=True)


def get_username() -> str:
    legacy = Path.home() / ".kaggle" / "kaggle.json"
    if legacy.exists():
        return json.loads(legacy.read_text(encoding="utf-8"))["username"]
    result = subprocess.run(
        [sys.executable, "-m", "kaggle", "config", "view"],
        capture_output=True, text=True,
    )
    match = re.search(r"username[\"': ]+([\w-]+)", result.stdout)
    if match:
        return match.group(1)
    raise SystemExit(
        "Can't determine Kaggle username from a KGAT_ token alone. "
        f"Write it to {Path.home() / '.kaggle' / 'username'} or set KAGGLE_USERNAME."
    )


def validate_notebook(nb_path: Path) -> None:
    notebook = json.loads(nb_path.read_text(encoding="utf-8"))
    problems = []
    for i, cell in enumerate(notebook.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        try:
            ast.parse(source)
        except SyntaxError as exc:
            problems.append(f"cell {i}: SyntaxError line {exc.lineno}")
        if re.search(r"^\s*!\s*(pip|apt)", source, re.M):
            problems.append(f"cell {i}: uses a shell install; internet is off")
    if problems:
        raise SystemExit("Notebook rejected before upload:\n  " + "\n  ".join(problems))
    print(f"[validate] {nb_path.name}: {len(notebook.get('cells', []))} cells parse cleanly, no shell installs")


def write_kernel_metadata(stage_dir: Path, slug: str, username: str, dataset_sources: list[str]) -> None:
    metadata = {
        "id": f"{username}/{slug}",
        "title": slug,
        "code_file": "train.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": False,
        "accelerator": ACCELERATOR,
        "dataset_sources": dataset_sources,
        "model_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
    }
    (stage_dir / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def push(stage_dir: Path) -> None:
    result = kaggle("kernels", "push", "-p", str(stage_dir), "--accelerator", ACCELERATOR)
    out = (result.stdout or "") + (result.stderr or "")
    print(out)
    if "successfully pushed" not in out.lower():
        reason = next(
            (line.strip() for line in out.splitlines() if "error" in line.lower() or "maximum" in line.lower()),
            "no confirmation line in output",
        )
        raise SystemExit(f"PUSH REJECTED (exit code was 0 anyway): {reason}")


def kernel_status(username: str, slug: str) -> str:
    result = kaggle("kernels", "status", f"{username}/{slug}")
    text = ((result.stdout or "") + (result.stderr or "")).strip().lower()
    print(f"  status: {text}")
    if "complete" in text:
        return "complete"
    if "error" in text:
        return "error"
    if "running" in text or "queued" in text:
        return "running"
    return "unknown"


def pull_output(username: str, slug: str, dest: Path) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)
    kaggle("kernels", "output", f"{username}/{slug}", "-p", str(dest), capture=False)


def find_run_dir(output_dir: Path) -> Path | None:
    for candidate in output_dir.rglob("weights/last.pt"):
        return candidate.parent.parent
    return None


def push_checkpoint_dataset(checkpoint_slug: str, username: str, run_dir: Path, first_time: bool) -> str:
    staging = STATE_DIR / "checkpoint_staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    shutil.copytree(run_dir, staging / "train")

    if first_time:
        metadata = {
            "title": checkpoint_slug,
            "id": f"{username}/{checkpoint_slug}",
            "licenses": [{"name": "CC0-1.0"}],
        }
        (staging / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        kaggle("datasets", "create", "-p", str(staging), "-q", "-r", "zip", capture=False)
    else:
        kaggle("datasets", "version", "-p", str(staging), "-m", "resume", "-q", "-r", "zip", capture=False)

    return f"{username}/{checkpoint_slug}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolo11m.pt", help="yolo11m.pt or yolo12m.pt")
    parser.add_argument("--run-name", required=True, help="e.g. stage1-yolo11m")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--time-limit-hours", type=float, default=8.5)
    parser.add_argument("--max-cycles", type=int, default=6)
    parser.add_argument("--attach", action="store_true",
                        help="A kernel with this run-name is already running (e.g. after a local "
                             "restart) - skip the push on cycle 1 and just resume polling it, "
                             "instead of pushing a fresh version and restarting it from scratch.")
    args = parser.parse_args()

    username = get_username()
    print(f"Kaggle username: {username}")

    slug = args.run_name
    checkpoint_slug = f"{args.run_name}-ckpt"
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    stage_dir = STATE_DIR / f"{slug}_stage"
    stage_dir.mkdir(parents=True, exist_ok=True)
    final_dest = PROJECT_ROOT / "runs" / args.run_name

    config = {
        "model_name": args.model,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "epochs": args.epochs,
        "patience": args.patience,
        "time_limit_hours": args.time_limit_hours,
        "run_name": args.run_name,
    }

    state_file = STATE_DIR / f"{slug}.json"
    checkpoint_dataset = None
    if state_file.exists():
        saved = json.loads(state_file.read_text(encoding="utf-8"))
        checkpoint_dataset = saved.get("checkpoint_dataset")
        print(f"Resuming existing loop; checkpoint dataset so far: {checkpoint_dataset}")

    for cycle in range(1, args.max_cycles + 1):
        print(f"\n========== Cycle {cycle}/{args.max_cycles} ==========")

        skip_push = args.attach and cycle == 1
        if skip_push:
            print(f"--attach: assuming {username}/{slug} is already running on Kaggle, skipping push.")
        else:
            nb_path = build_notebook(stage_dir, config)
            validate_notebook(nb_path)

            dataset_sources = [RSUD20K_DATASET, WHEELS_DATASET, WEIGHTS_DATASET]
            if checkpoint_dataset:
                dataset_sources.append(checkpoint_dataset)
            write_kernel_metadata(stage_dir, slug, username, dataset_sources)

            push(stage_dir)

        print("Waiting for the run to finish (polling every 2 minutes)...")
        status = "unknown"
        while True:
            time.sleep(POLL_INTERVAL_SECONDS)
            status = kernel_status(username, slug)
            if status in ("complete", "error"):
                break

        if status == "error":
            print(f"Kernel errored — inspect https://www.kaggle.com/code/{username}/{slug}")
            output_dir = STATE_DIR / f"{slug}_error_cycle{cycle}"
            pull_output(username, slug, output_dir)
            print(f"Log pulled to: {output_dir}")
            sys.exit(1)

        output_dir = STATE_DIR / f"{slug}_output_cycle{cycle}"
        pull_output(username, slug, output_dir)

        if (output_dir / "DONE.flag").exists():
            run_dir = find_run_dir(output_dir)
            if run_dir is not None:
                final_dest.mkdir(parents=True, exist_ok=True)
                shutil.copytree(run_dir, final_dest / "train", dirs_exist_ok=True)
                print(f"\nTraining complete. Weights: {final_dest / 'train' / 'weights' / 'best.pt'}")
            return

        run_dir = find_run_dir(output_dir)
        if run_dir is None:
            print("No checkpoint weights found and no DONE.flag — inspect the output manually. Stopping.")
            sys.exit(1)

        print("Time limit reached, not yet converged — pushing checkpoint and resuming next cycle.")
        checkpoint_dataset = push_checkpoint_dataset(
            checkpoint_slug, username, run_dir, first_time=(checkpoint_dataset is None)
        )
        state_file.write_text(json.dumps({"checkpoint_dataset": checkpoint_dataset}), encoding="utf-8")

    print(f"\nHit --max-cycles ({args.max_cycles}) without finishing. Re-run this command to continue.")


if __name__ == "__main__":
    main()
