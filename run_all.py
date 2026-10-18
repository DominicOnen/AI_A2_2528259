"""One command: recreate data, clean it, train every task, regenerate artifacts.

    python run_all.py --student-id ug2528259@ines.ac.rw --seed 8259
"""

from __future__ import annotations

import argparse
import sys

from src.pipeline import list_artifacts, run


def main() -> int:
    parser = argparse.ArgumentParser(description="Musanze Harvest Intelligence Service")
    parser.add_argument("--student-id", required=True, help="e.g. ug2528259@ines.ac.rw")
    parser.add_argument("--seed", type=int, required=True, help="LLLL, last four digits")
    args = parser.parse_args()

    try:
        summary = run(args.student_id, args.seed)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    reg = summary["regression"]
    clf = summary["classification"]
    clu = summary["clustering"]
    best_reg = reg["models"][reg["selected_model"]]
    best_clf = clf["models"][clf["selected_model"]]
    print()
    print(f"Regression ({reg['selected_model']}): test MAE {best_reg['test_mae']}, "
          f"RMSE {best_reg['test_rmse']}, R2 {best_reg['test_r2']}")
    print(f"Classification ({clf['selected_model']}): test precision {best_clf['test_precision']}, "
          f"recall {best_clf['test_recall']}, F1 {best_clf['test_f1']}, "
          f"ROC-AUC {best_clf['test_roc_auc']}")
    print(f"Clustering: selected K={clu['selected_k']} "
          f"(silhouette {clu['silhouette_by_k'][str(clu['selected_k'])]})")
    print()
    print("Generated artifacts:")
    for path in list_artifacts():
        print(f"  {path}")
    print()
    print(f"Done. Verification code: {summary['identity'].verification_code}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
