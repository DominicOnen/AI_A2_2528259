"""Plots and the test-prediction table."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

from .config import NUMERIC_FEATURES, TARGET_CLASSIFICATION, TARGET_REGRESSION  # noqa: E402

DPI = 150


def build_test_predictions(
    test: pd.DataFrame,
    yield_pred: np.ndarray,
    risk_pred: np.ndarray,
    risk_proba: np.ndarray,
    clusters: np.ndarray,
) -> pd.DataFrame:
    """Row-level results for the held-out test rows (row ID = CSV data-row index)."""
    actual = test[TARGET_REGRESSION].to_numpy()
    out = pd.DataFrame(
        {
            "row_id": test.index.to_numpy(),
            "actual_yield_tons": actual,
            "predicted_yield_tons": yield_pred,
            "yield_error_tons": yield_pred - actual,
            "abs_error_tons": np.abs(yield_pred - actual),
            "actual_high_risk": test[TARGET_CLASSIFICATION].to_numpy(),
            "predicted_high_risk": risk_pred,
            "high_risk_probability": risk_proba,
            "cluster": clusters,
        }
    )
    return out.sort_values("row_id").reset_index(drop=True)


def save_predictions(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, float_format="%.4f", lineterminator="\n")


def plot_residuals(pred_df: pd.DataFrame, path: Path, code: str) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.scatter(pred_df["predicted_yield_tons"], pred_df["yield_error_tons"], s=22, alpha=0.8)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xlabel("Predicted yield (tons)")
    ax.set_ylabel("Error = predicted - actual (tons)")
    ax.set_title(f"Regression residuals on test rows ({code})")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def plot_confusion_matrix(matrix: list[list[int]], path: Path, code: str) -> None:
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    data = np.array(matrix)
    ax.imshow(data, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                str(data[i, j]),
                ha="center",
                va="center",
                fontsize=16,
                color="white" if data[i, j] > data.max() / 2 else "black",
            )
    ax.set_xticks([0, 1], ["Not high risk", "High risk"])
    ax.set_yticks([0, 1], ["Not high risk", "High risk"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion matrix, test rows ({code})")
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def plot_clusters(
    cluster_model, train: pd.DataFrame, test: pd.DataFrame, path: Path, code: str
) -> None:
    prep = cluster_model.named_steps["prep"]
    kmeans = cluster_model.named_steps["kmeans"]
    combined = pd.concat([train, test])
    scaled = prep.transform(combined[NUMERIC_FEATURES])
    labels = cluster_model.predict(combined[NUMERIC_FEATURES])
    pca = PCA(n_components=2, random_state=0)
    points = pca.fit_transform(scaled)
    centres = pca.transform(kmeans.cluster_centers_)

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for cluster in sorted(set(labels)):
        mask = labels == cluster
        ax.scatter(points[mask, 0], points[mask, 1], s=20, alpha=0.75, label=f"Cluster {cluster}")
    ax.scatter(
        centres[:, 0], centres[:, 1], marker="X", s=140, c="black", edgecolors="white", label="Centres"
    )
    ax.set_xlabel(f"Component 1 ({pca.explained_variance_ratio_[0]:.0%} of variance)")
    ax.set_ylabel(f"Component 2 ({pca.explained_variance_ratio_[1]:.0%} of variance)")
    ax.set_title(f"Farm clusters, K={kmeans.n_clusters} ({code})")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
