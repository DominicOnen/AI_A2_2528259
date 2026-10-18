"""End-to-end run: data -> validation -> cleaning -> models -> artifacts."""

from __future__ import annotations

import json
import math
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from data.generate_data import base_row_count, sha256_of_file, write_dataset

from .config import FEATURES, NUMERIC_FEATURES, ROOT
from .evaluation import (
    build_test_predictions,
    plot_clusters,
    plot_confusion_matrix,
    plot_residuals,
    save_predictions,
)
from .identity import StudentIdentity, check_seed, parse_student_id
from .modeling import run_classification, run_clustering, run_regression
from .preprocessing import split_data
from .validation import check_no_leakage, clean_dataset, quality_report, validate_dataset


def _clean_json(obj):
    """Make numpy values and NaN JSON-safe (NaN -> null)."""
    if isinstance(obj, dict):
        return {str(k): _clean_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean_json(v) for v in obj]
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float) and math.isnan(obj):
        return None
    return obj


def write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(_clean_json(payload), indent=2, sort_keys=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def run(student_id: str, seed: int, root: Path = ROOT, quiet: bool = False) -> dict:
    """Run the whole system and regenerate every artifact under ``root``."""
    say = (lambda *a: None) if quiet else print
    identity: StudentIdentity = parse_student_id(student_id)
    check_seed(identity, seed)
    code = identity.verification_code

    root = Path(root)
    data_path = root / "data" / identity.dataset_name
    artifacts = root / "artifacts"
    plots = artifacts / "plots"
    models = root / "models"
    for folder in (data_path.parent, plots, models):
        folder.mkdir(parents=True, exist_ok=True)

    say(f"Verification code: {code}")
    say(f"Student ID: {identity.student_id}   Seed (LLLL): {seed}")

    # 1. Recreate the dataset and fingerprint it.
    write_dataset(seed, data_path)
    fingerprint = sha256_of_file(data_path)
    (artifacts / "data_fingerprint.txt").write_text(
        f"verification_code={code}\n"
        f"file=data/{identity.dataset_name}\n"
        f"base_rows={base_row_count(seed)}\n"
        f"sha256={fingerprint}\n",
        encoding="utf-8",
    )
    say(f"Dataset: data/{identity.dataset_name}")
    say(f"SHA-256 fingerprint: {fingerprint}")

    # 2. Validate, then clean.
    raw = pd.read_csv(data_path)  # row ID = position of the row in the CSV
    before = validate_dataset(raw)
    clean = clean_dataset(raw)
    after = quality_report(clean)
    check_no_leakage(FEATURES)
    write_json(
        artifacts / "validation_report.json",
        {
            "verification_code": code,
            "before_cleaning": before,
            "after_cleaning": after,
            "cleaning_steps": [
                "dropped exact duplicate rows (first occurrence kept)",
                "missing numeric values are imputed inside the pipeline using training rows only",
            ],
            "leakage_check": "passed: yield_tons, high_risk and yield per hectare are not inputs",
            "input_features": FEATURES,
        },
    )
    say(
        f"Rows: {before['rows']} raw -> {after['rows']} after removing "
        f"{before['exact_duplicate_rows']} duplicates; missing values: {before['missing_values']}"
    )

    # 3. One fixed split for every task.
    train, test = split_data(clean, seed)
    split_info = {"train_rows": len(train), "test_rows": len(test), "random_state": seed}

    # 4. Train, compare and evaluate.
    reg = run_regression(train, test, seed)
    clf = run_classification(train, test, seed)
    clu = run_clustering(train, test, seed)

    header = {"verification_code": code, "seed": seed, "split": split_info}
    write_json(artifacts / "regression_metrics.json", {**header, **reg["report"]})
    write_json(artifacts / "classification_metrics.json", {**header, **clf["report"]})
    write_json(artifacts / "clustering_report.json", {**header, **clu["report"]})

    pred_df = build_test_predictions(
        test, reg["predictions"], clf["predictions"], clf["proba"], clu["test_labels"]
    )
    save_predictions(pred_df, artifacts / "test_predictions.csv")

    # 5. Plots.
    selected_clf = clf["report"]["selected_model"]
    matrix = clf["report"]["models"][selected_clf]["confusion_matrix"][
        "rows_actual_columns_predicted"
    ]
    plot_residuals(pred_df, plots / "residuals.png", code)
    plot_confusion_matrix(matrix, plots / "confusion_matrix.png", code)
    plot_clusters(clu["model"], train, test, plots / "clusters.png", code)

    # 6. Save models and metadata.
    joblib.dump(reg["model"], models / "regression_model.joblib")
    joblib.dump(clf["model"], models / "classification_model.joblib")
    joblib.dump(clu["model"], models / "cluster_model.joblib")
    write_json(
        models / "model_metadata.json",
        {
            "service": "Musanze Harvest Intelligence Service",
            "verification_code": code,
            "student_id": identity.student_id,
            "seed": seed,
            "dataset": f"data/{identity.dataset_name}",
            "dataset_sha256": fingerprint,
            "synthetic_data": True,
            "rows_after_cleaning": after["rows"],
            "split": split_info,
            "input_features": FEATURES,
            "cluster_features": NUMERIC_FEATURES,
            "regression_model": reg["report"]["selected_model"],
            "classification_model": clf["report"]["selected_model"],
            "clusters": clu["report"]["selected_k"],
            "fitted_on": "training rows only",
            "environment": {
                "python": platform.python_version(),
                "scikit_learn": sklearn.__version__,
                "numpy": np.__version__,
                "pandas": pd.__version__,
            },
        },
    )

    summary = {
        "identity": identity,
        "fingerprint": fingerprint,
        "rows_raw": before["rows"],
        "rows_clean": after["rows"],
        "regression": reg["report"],
        "classification": clf["report"],
        "clustering": clu["report"],
    }
    return summary


def list_artifacts(root: Path = ROOT) -> list[str]:
    """Relative paths of generated files (for the run summary)."""
    root = Path(root)
    files = sorted(
        p.relative_to(root).as_posix()
        for folder in ("artifacts", "models")
        for p in (root / folder).rglob("*")
        if p.is_file()
    )
    return files
