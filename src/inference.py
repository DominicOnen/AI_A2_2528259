"""Load saved artifacts and make one validated prediction."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from .config import DISCLAIMER, FEATURES, ROOT
from .schemas import FarmRecord

MODEL_FILES = {
    "regression": "regression_model.joblib",
    "classification": "classification_model.joblib",
    "clustering": "cluster_model.joblib",
}


class Predictor:
    """Holds the saved models. Fails with a helpful message if they are missing."""

    def __init__(self, root: Path = ROOT):
        self.root = Path(root)
        models_dir = self.root / "models"
        missing = [f for f in MODEL_FILES.values() if not (models_dir / f).exists()]
        meta_path = models_dir / "model_metadata.json"
        if missing or not meta_path.exists():
            raise FileNotFoundError(
                "Saved models not found. Run: python run_all.py --student-id <id> --seed <LLLL>"
            )
        self.regression = joblib.load(models_dir / MODEL_FILES["regression"])
        self.classification = joblib.load(models_dir / MODEL_FILES["classification"])
        self.clustering = joblib.load(models_dir / MODEL_FILES["clustering"])
        self.metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        report = json.loads(
            (self.root / "artifacts" / "clustering_report.json").read_text(encoding="utf-8")
        )
        self.cluster_profiles = {p["cluster"]: p["profile"] for p in report["profiles"]}

    @property
    def verification_code(self) -> str:
        return self.metadata["verification_code"]

    def predict(self, record: dict) -> dict:
        """Validate one record (raises pydantic.ValidationError) and predict."""
        farm = FarmRecord.model_validate(record)
        row = pd.DataFrame([farm.model_dump()])[FEATURES]
        yield_tons = float(self.regression.predict(row)[0])
        probability = float(self.classification.predict_proba(row)[0, 1])
        cluster = int(self.clustering.predict(row[self.metadata["cluster_features"]])[0])
        return {
            "yield_estimate_tons": round(yield_tons, 4),
            "risk_prediction": "high" if probability >= 0.5 else "not_high",
            "high_risk_probability": round(probability, 4),
            "cluster": cluster,
            "cluster_profile": self.cluster_profiles[cluster],
            "verification_code": self.verification_code,
            "disclaimer": DISCLAIMER,
        }
