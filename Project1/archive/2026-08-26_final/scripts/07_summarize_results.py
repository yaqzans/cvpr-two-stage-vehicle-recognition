"""07 - Collect every pipeline_eval run into one comparison table and report.

Reads runs/pipeline_eval/*/metrics.json and writes:
    runs/pipeline_eval/SUMMARY.csv
    runs/pipeline_eval/SUMMARY.md

Nothing is recomputed here - this only aggregates what 06 already measured,
so it is safe to re-run at any time.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = PROJECT_ROOT / "runs" / "pipeline_eval"

COLUMNS = [
    ("run", lambda r, n: n),
    ("mode", lambda r, n: r["mode"]),
    ("conf", lambda r, n: f"{r['conf']:g}"),
    ("imgsz", lambda r, n: r.get("imgsz", 640)),
    ("tta", lambda r, n: "yes" if r.get("tta") else ""),
    ("mAP50", lambda r, n: f"{r['map50']:.4f}"),
    ("mAP50-95", lambda r, n: f"{r['map50_95']:.4f}"),
    ("localised", lambda r, n: f"{r['localisation_rate']:.4f}"),
    ("e2e acc", lambda r, n: f"{r['end_to_end_accuracy']:.4f}"),
    ("cls|loc", lambda r, n: f"{r['class_accuracy_given_localised']:.4f}"),
    ("ms/frame", lambda r, n: f"{r['timing_ms_per_frame']['pipeline_total']:.1f}"),
    ("FPS", lambda r, n: f"{r['fps_pipeline']:.1f}"),
    ("crops/f", lambda r, n: f"{r['crops_per_frame']:.1f}"),
]


def load_runs():
    runs = []
    for metrics_path in sorted(EVAL_DIR.glob("*/metrics.json")):
        name = metrics_path.parent.name
        if name.startswith("_"):
            continue
        with open(metrics_path, encoding="utf-8") as f:
            runs.append((name, json.load(f)))
    return runs


def main() -> None:
    runs = load_runs()
    if not runs:
        print(f"No runs found under {EVAL_DIR}")
        return

    header = [c[0] for c in COLUMNS]
    rows = [[fn(record, name) for _, fn in COLUMNS] for name, record in runs]

    widths = [max(len(str(header[i])), max(len(str(r[i])) for r in rows))
              for i in range(len(header))]

    def render(cells):
        return "  ".join(str(c).ljust(widths[i]) for i, c in enumerate(cells))

    print("=" * (sum(widths) + 2 * len(widths)))
    print("PIPELINE EVALUATION SUMMARY")
    print("=" * (sum(widths) + 2 * len(widths)))
    print(render(header))
    print("-" * (sum(widths) + 2 * len(widths)))
    for row in rows:
        print(render(row))

    with open(EVAL_DIR / "SUMMARY.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    with open(EVAL_DIR / "SUMMARY.md", "w", encoding="utf-8") as f:
        f.write("# Pipeline evaluation summary\n\n")
        f.write("| " + " | ".join(header) + " |\n")
        f.write("|" + "|".join(["---"] * len(header)) + "|\n")
        for row in rows:
            f.write("| " + " | ".join(str(c) for c in row) + " |\n")
        f.write("\n")
        device = runs[0][1].get("device", "unknown")
        f.write(f"Device: {device}. Test split: 649 images, 3,805 objects, 1920x1080.\n")

    print(f"\nSaved -> {EVAL_DIR / 'SUMMARY.csv'}")
    print(f"Saved -> {EVAL_DIR / 'SUMMARY.md'}")


if __name__ == "__main__":
    main()
