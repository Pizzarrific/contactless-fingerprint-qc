"""
test_quality.py
================
Runs quality_gate() on all images under data/{good,blurry,dark,glare}
and prints a results table, plus a pass/fail summary confirming each
category is flagged with the expected defect. Also writes results.json
and results_table.csv for inclusion in the report.
"""

import json
import csv
from pathlib import Path

from quality_assessment import quality_gate

DATA_ROOT = Path(__file__).parent / "data"
CATEGORIES = ["good", "blurry", "dark", "glare"]

# Which flag we EXPECT to fire for each category, used only to print a
# self-check column — the grader's own images may differ, this is just a
# sanity check against the synthetic set / the 20 photos from Part D.
EXPECTED_FLAG = {
    "good": None,
    "blurry": "is_blurry",
    "dark": "too_dark",
    "glare": "has_glare",
}


def flag_fired(result: dict, flag_name: str) -> bool:
    if flag_name is None:
        return result["passed"]
    for section in ("blur", "brightness", "glare", "roi", "ridge"):
        if flag_name in result[section] and result[section][flag_name]:
            return True
    return False


def main():
    rows = []
    all_results = {}

    for category in CATEGORIES:
        folder = DATA_ROOT / category
        if not folder.exists():
            print(f"WARNING: {folder} does not exist — skipping. "
                  f"Did you run generate_sample_images.py or add your own photos?")
            continue

        images = sorted([p for p in folder.iterdir()
                          if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
        if not images:
            print(f"WARNING: no images found in {folder}")
            continue

        for img_path in images:
            try:
                result = quality_gate(str(img_path))
            except Exception as e:
                # Never let one bad file crash the whole batch run.
                print(f"ERROR processing {img_path}: {e}")
                rows.append({
                    "file": str(img_path.relative_to(DATA_ROOT.parent)),
                    "category": category, "composite_score": "ERROR",
                    "passed": "ERROR", "expected_flag_correct": "ERROR",
                    "guidance": str(e),
                })
                continue

            all_results[str(img_path.name)] = result
            expected = EXPECTED_FLAG[category]
            correct = flag_fired(result, expected)

            rows.append({
                "file": f"{category}/{img_path.name}",
                "category": category,
                "composite_score": result["composite_score"],
                "passed": result["passed"],
                "expected_flag_correct": correct,
                "guidance": result["guidance"],
                "processing_time_ms": result["processing_time_ms"],
            })

    # Print table
    col_widths = {"file": 24, "category": 9, "composite_score": 9,
                  "passed": 7, "expected_flag_correct": 10, "processing_time_ms": 8}
    header = (f"{'file':<24}{'category':<9}{'score':<9}{'passed':<7}"
              f"{'correct?':<10}{'ms':<8}guidance")
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['file']:<24}{r['category']:<9}{str(r['composite_score']):<9}"
              f"{str(r['passed']):<7}{str(r['expected_flag_correct']):<10}"
              f"{str(r.get('processing_time_ms', '')):<8}{r['guidance']}")

    n_correct = sum(1 for r in rows if r["expected_flag_correct"] is True)
    n_total = sum(1 for r in rows if r["expected_flag_correct"] in (True, False))
    print(f"\nSelf-check: correctly identified defect in {n_correct}/{n_total} images")

    with open("results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    with open("results_table.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    print("\nWrote results.json and results_table.csv")


if __name__ == "__main__":
    main()
