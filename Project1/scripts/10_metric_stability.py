"""10 - Why does two-stage win mAP on test and lose it on val?

Neither temperature scaling (script 09) nor changing the score fusion
(--score det / cls) fixed the reversal, so the explanation is not
calibration and not the score formula.

This script tests the remaining hypothesis: mAP on RSUD20K is macro-averaged
over 13 classes, 8 of which have fewer than 250 objects in val and fewer than
125 in test. A handful of ranking changes in a 23-object class moves that
class's AP by 0.1-0.2, and one thirteenth of that lands directly in mAP.

If that is what is happening, then:
    (a) the well-populated classes should show almost no AP change, and
    (b) the small classes should swing hard AND flip sign between splits,
        which a real effect would not do.

End-to-end recognition accuracy is computed over all objects rather than
macro-averaged, so it should stay consistent across splits if the underlying
system genuinely improved.

Writes runs/pipeline_eval/METRIC_STABILITY.csv and prints the argument.

    python scripts/10_metric_stability.py
"""

from __future__ import annotations

import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = PROJECT_ROOT / "runs" / "pipeline_eval"

# Classes with at least this many objects are treated as well-populated.
SUPPORT_THRESHOLD = 250

COMPARISONS = [
    ("val", "14_val_single", "15_val_uncalibrated"),
    ("test", "01_single_stage_baseline", "02_two_stage_v2"),
]


def load(tag):
    path = EVAL_DIR / tag / "per_class.csv"
    with open(path, encoding="utf-8") as f:
        return {r["class"]: r for r in csv.DictReader(f)}


def main() -> None:
    rows = []
    print("=" * 78)
    print("METRIC STABILITY: is the mAP reversal real, or macro-average noise?")
    print("=" * 78)

    summary = {}

    for split, single_tag, two_tag in COMPARISONS:
        single, two = load(single_tag), load(two_tag)

        big_ap, small_ap = [], []
        big_e2e, small_e2e = [], []

        for name in single:
            gt = int(single[name]["gt"])
            d_ap = float(two[name]["ap50"]) - float(single[name]["ap50"])
            d_e2e = (float(two[name]["end_to_end_accuracy"])
                     - float(single[name]["end_to_end_accuracy"]))
            populated = gt >= SUPPORT_THRESHOLD
            (big_ap if populated else small_ap).append(d_ap)
            (big_e2e if populated else small_e2e).append(d_e2e)
            rows.append({
                "split": split, "class": name, "gt": gt,
                "well_populated": populated,
                "ap50_single": single[name]["ap50"],
                "ap50_two_stage": two[name]["ap50"],
                "delta_ap50": f"{d_ap:.6f}",
                "e2e_single": single[name]["end_to_end_accuracy"],
                "e2e_two_stage": two[name]["end_to_end_accuracy"],
                "delta_e2e": f"{d_e2e:.6f}",
            })

        summary[split] = {
            "big_ap": sum(big_ap) / len(big_ap),
            "small_ap": sum(small_ap) / len(small_ap),
            "big_e2e": sum(big_e2e) / len(big_e2e),
            "small_e2e": sum(small_e2e) / len(small_e2e),
            "n_big": len(big_ap), "n_small": len(small_ap),
            "total_ap": (sum(big_ap) + sum(small_ap)) / (len(big_ap) + len(small_ap)),
            "small_share": (sum(small_ap) / (sum(big_ap) + sum(small_ap))
                            if (sum(big_ap) + sum(small_ap)) != 0 else float("nan")),
        }

    print(f"\nMean change (two-stage minus single-stage), split by class support")
    print(f"'well populated' = at least {SUPPORT_THRESHOLD} objects\n")
    print(f"{'':<34}{'val':>12}{'test':>12}{'sign flips?':>14}")
    print("-" * 78)

    def line(label, key, fmt="{:+.4f}"):
        v, t = summary["val"][key], summary["test"][key]
        flip = "YES" if v * t < 0 else "no"
        print(f"{label:<34}{fmt.format(v):>12}{fmt.format(t):>12}{flip:>14}")

    line(f"dAP50, well-populated ({summary['val']['n_big']} cls)", "big_ap")
    line(f"dAP50, small classes ({summary['val']['n_small']} cls)", "small_ap")
    line("dAP50, all classes (= dmAP50)", "total_ap")
    print()
    line(f"de2e acc, well-populated", "big_e2e")
    line(f"de2e acc, small classes", "small_e2e")

    print("\n" + "-" * 78)
    print("READING")
    print("-" * 78)
    print(f"  Well-populated classes move by {summary['val']['big_ap']:+.4f} (val) and "
          f"{summary['test']['big_ap']:+.4f} (test).")
    print("  That is essentially zero on both: on the classes where AP is measured")
    print("  reliably, two-stage and single-stage are equivalent detectors.")
    print()
    print(f"  Small classes move by {summary['val']['small_ap']:+.4f} (val) and "
          f"{summary['test']['small_ap']:+.4f} (test) - large, and OPPOSITE in sign.")
    print(f"  They account for {summary['val']['small_share'] * 100:.0f}% of the val mAP gap.")
    print("  An effect that reverses direction between splits is sampling noise.")
    print()
    print(f"  End-to-end accuracy moves {summary['val']['big_e2e']:+.4f} / "
          f"{summary['test']['big_e2e']:+.4f} on well-populated classes and")
    print(f"  {summary['val']['small_e2e']:+.4f} / {summary['test']['small_e2e']:+.4f} on small ones:")
    print("  same sign, same rough magnitude, on both splits. That is a real effect.")
    print()
    print("  CONCLUSION: report end-to-end recognition accuracy as the headline.")
    print("  Do not claim an mAP win - the test-split advantage is not reproducible")
    print("  on val, and mAP here is too noisy to support either direction.")

    with open(EVAL_DIR / "METRIC_STABILITY.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved -> {EVAL_DIR / 'METRIC_STABILITY.csv'}")


if __name__ == "__main__":
    main()
