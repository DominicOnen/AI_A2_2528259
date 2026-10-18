"""Data validation, quality reporting and cleaning."""

from __future__ import annotations

import pandas as pd

from .config import (
    FORBIDDEN_FEATURES,
    MIN_YIELD_TONS,
    RANGES,
    REQUIRED_COLUMNS,
    SEASONS,
)

TOLERANCE = 1e-6  # CSV values are rounded to four decimals


def check_schema(df: pd.DataFrame) -> None:
    """Raise ValueError if a required column is missing."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")


def range_violations(df: pd.DataFrame) -> dict[str, int]:
    """Count out-of-range values per column (missing values are ignored)."""
    problems: dict[str, int] = {}
    for column, (low, high) in RANGES.items():
        values = df[column].dropna()
        count = int(((values < low - TOLERANCE) | (values > high + TOLERANCE)).sum())
        if count:
            problems[column] = count
    bad_season = int((~df["season"].isin(SEASONS)).sum())
    if bad_season:
        problems["season"] = bad_season
    bad_support = int((~df["cooperative_support"].isin([0, 1])).sum())
    if bad_support:
        problems["cooperative_support"] = bad_support
    low_yield = int((df["yield_tons"] < MIN_YIELD_TONS - TOLERANCE).sum())
    if low_yield:
        problems["yield_tons"] = low_yield
    bad_risk = int((~df["high_risk"].isin([0, 1])).sum())
    if bad_risk:
        problems["high_risk"] = bad_risk
    return problems


def quality_report(df: pd.DataFrame) -> dict:
    """Evidence of missing values, duplicates and range problems."""
    return {
        "rows": int(len(df)),
        "missing_values": {c: int(n) for c, n in df.isna().sum().items() if n},
        "exact_duplicate_rows": int(df.duplicated().sum()),
        "range_violations": range_violations(df),
    }


def validate_dataset(df: pd.DataFrame) -> dict:
    """Run all checks. Schema and range errors raise; missing/duplicates are reported."""
    check_schema(df)
    report = quality_report(df)
    if report["range_violations"]:
        raise ValueError(f"Out-of-range values found: {report['range_violations']}")
    return report


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicate rows (first occurrence kept, original row ID retained).

    Missing numeric values are NOT filled here. They are imputed inside the
    modelling pipeline, using statistics from the training split only.
    """
    return df.drop_duplicates(keep="first")


def check_no_leakage(feature_columns: list[str]) -> None:
    """Raise if a target (or a value derived from it) is used as an input."""
    leaked = sorted(set(feature_columns) & FORBIDDEN_FEATURES)
    if leaked:
        raise ValueError(f"Target leakage: {leaked} must not be model inputs.")
