"""Train/test split and the reusable preprocessing transformers."""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import (
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET_CLASSIFICATION,
    TEST_SIZE,
)


def split_data(df: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fixed, stratified train/test split with random_state=LLLL.

    One split is shared by the regression, classification and clustering tasks
    so no test row is ever seen during any fitting step.
    """
    return train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=seed,
        stratify=df[TARGET_CLASSIFICATION],
    )


def build_preprocessor() -> ColumnTransformer:
    """Impute and scale numbers, one-hot encode the season, pass the binary flag.

    Fitted on training rows only; the same fitted object is applied at inference.
    """
    numeric = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    return ColumnTransformer(
        [
            ("num", numeric, NUMERIC_FEATURES),
            (
                "season",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
            ("support", "passthrough", BINARY_FEATURES),
        ]
    )


def build_cluster_preprocessor() -> Pipeline:
    """Impute and scale the five field-condition numbers used for clustering."""
    return Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
