"""Shared constants: column names, ranges and modelling settings."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

NUMERIC_FEATURES = [
    "rainfall_mm",
    "avg_temp_c",
    "soil_moisture_pct",
    "fertilizer_kg",
    "field_size_ha",
]
CATEGORICAL_FEATURES = ["season"]
BINARY_FEATURES = ["cooperative_support"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES

TARGET_REGRESSION = "yield_tons"
TARGET_CLASSIFICATION = "high_risk"
REQUIRED_COLUMNS = FEATURES + [TARGET_REGRESSION, TARGET_CLASSIFICATION]

# Columns that must never be model inputs (target leakage).
FORBIDDEN_FEATURES = {TARGET_REGRESSION, TARGET_CLASSIFICATION, "yield_per_ha"}

# Valid ranges for the generated data (inclusive).
RANGES = {
    "rainfall_mm": (600.0, 1400.0),
    "avg_temp_c": (12.0, 26.0),
    "soil_moisture_pct": (20.0, 80.0),
    "fertilizer_kg": (0.0, 250.0),
    "field_size_ha": (0.2, 4.0),
}
SEASONS = ("A", "B")
MIN_YIELD_TONS = 0.50

TEST_SIZE = 0.20
CV_FOLDS = 5
CLUSTER_K_RANGE = range(2, 7)

DISCLAIMER = (
    "Classroom decision-support prototype trained on synthetic data. "
    "It does not replace an agronomist and must not be used to decide access "
    "to finance, insurance or public services. A human must review every result."
)
