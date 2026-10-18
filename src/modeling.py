"""Candidate models, selection by cross-validation, and clustering."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from .config import (
    CLUSTER_K_RANGE,
    CV_FOLDS,
    FEATURES,
    NUMERIC_FEATURES,
    TARGET_CLASSIFICATION,
    TARGET_REGRESSION,
)
from .preprocessing import build_cluster_preprocessor, build_preprocessor
from .validation import check_no_leakage

REGRESSION_COMPLEXITY = ["linear_regression", "random_forest", "gradient_boosting"]
CLASSIFICATION_COMPLEXITY = ["logistic_regression", "random_forest"]
REGRESSION_BASELINE = "mean_baseline"
CLASSIFICATION_BASELINE = "majority_baseline"
DIGITS = 6


def _r(value: float) -> float:
    return round(float(value), DIGITS)


def select_simplest_within_one_se(
    results: dict, metric: str, higher_is_better: bool, complexity_order: list[str]
) -> tuple[str, str, float]:
    """One-standard-error rule on cross-validation scores.

    Find the best candidate, then pick the simplest candidate whose score is within
    one standard error of the best. Only training-row CV scores are used.
    Returns (selected, best, threshold).
    """
    pool = {k: results[k] for k in complexity_order}
    pick = max if higher_is_better else min
    best = pick(pool, key=lambda k: pool[k][metric])
    std_error = pool[best][metric + "_std"] / (CV_FOLDS ** 0.5)
    if higher_is_better:
        threshold = pool[best][metric] - std_error
        ok = [k for k in complexity_order if pool[k][metric] >= threshold]
    else:
        threshold = pool[best][metric] + std_error
        ok = [k for k in complexity_order if pool[k][metric] <= threshold]
    return ok[0], best, _r(threshold)


def regression_candidates(seed: int) -> dict:
    """Mean baseline plus three candidate regressors."""
    return {
        REGRESSION_BASELINE: DummyRegressor(strategy="mean"),
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(
            n_estimators=300, min_samples_leaf=2, random_state=seed, n_jobs=1
        ),
        "gradient_boosting": GradientBoostingRegressor(random_state=seed),
    }


def classification_candidates(seed: int) -> dict:
    """Majority baseline plus two candidate classifiers."""
    return {
        CLASSIFICATION_BASELINE: DummyClassifier(strategy="most_frequent"),
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=seed),
        "random_forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2, random_state=seed, n_jobs=1
        ),
    }


def _pipeline(estimator) -> Pipeline:
    return Pipeline([("prep", build_preprocessor()), ("model", estimator)])


def run_regression(train: pd.DataFrame, test: pd.DataFrame, seed: int) -> dict:
    """Compare candidates by CV on the training rows, then score the test rows."""
    check_no_leakage(FEATURES)
    x_train, y_train = train[FEATURES], train[TARGET_REGRESSION]
    x_test, y_test = test[FEATURES], test[TARGET_REGRESSION]
    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=seed)

    results, fitted = {}, {}
    for name, estimator in regression_candidates(seed).items():
        pipe = _pipeline(estimator)
        scores = cross_validate(
            pipe,
            x_train,
            y_train,
            cv=cv,
            scoring={
                "rmse": "neg_root_mean_squared_error",
                "mae": "neg_mean_absolute_error",
                "r2": "r2",
            },
        )
        pipe.fit(x_train, y_train)
        pred = pipe.predict(x_test)
        results[name] = {
            "cv_rmse": _r(-scores["test_rmse"].mean()),
            "cv_rmse_std": _r(scores["test_rmse"].std(ddof=1)),
            "cv_mae": _r(-scores["test_mae"].mean()),
            "cv_r2": _r(scores["test_r2"].mean()),
            "test_mae": _r(mean_absolute_error(y_test, pred)),
            "test_rmse": _r(np.sqrt(mean_squared_error(y_test, pred))),
            "test_r2": _r(r2_score(y_test, pred)),
        }
        fitted[name] = pipe

    selected, best, threshold = select_simplest_within_one_se(
        results, "cv_rmse", False, REGRESSION_COMPLEXITY
    )
    reason = (
        f"{best} had the lowest {CV_FOLDS}-fold cross-validated RMSE on the training rows "
        f"({results[best]['cv_rmse']} tons). Rule: choose the simplest model within one "
        f"standard error of the best (RMSE <= {threshold}). Selected: {selected} "
        f"(RMSE {results[selected]['cv_rmse']} tons) versus the mean baseline "
        f"({results[REGRESSION_BASELINE]['cv_rmse']} tons). Test rows were used only to report "
        "final scores, not to choose the model."
    )
    pred = fitted[selected].predict(x_test)
    error = pred - y_test.to_numpy()
    worst = int(np.argmax(np.abs(error)))
    report = {
        "task": "regression",
        "target": TARGET_REGRESSION,
        "baseline": REGRESSION_BASELINE,
        "selection_metric": f"{CV_FOLDS}-fold CV RMSE on training rows",
        "models": results,
        "selected_model": selected,
        "selection_reason": reason,
        "largest_test_error": {
            "row_id": int(test.index[worst]),
            "actual_yield_tons": _r(y_test.iloc[worst]),
            "predicted_yield_tons": _r(pred[worst]),
            "error_tons": _r(error[worst]),
            "inputs": {c: test.iloc[worst][c] for c in FEATURES},
        },
    }
    return {"report": report, "model": fitted[selected], "predictions": pred}


def run_classification(train: pd.DataFrame, test: pd.DataFrame, seed: int) -> dict:
    """Compare classifiers by CV F1 on training rows, then score the test rows."""
    check_no_leakage(FEATURES)
    x_train, y_train = train[FEATURES], train[TARGET_CLASSIFICATION]
    x_test, y_test = test[FEATURES], test[TARGET_CLASSIFICATION]
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=seed)

    results, fitted = {}, {}
    for name, estimator in classification_candidates(seed).items():
        pipe = _pipeline(estimator)
        scores = cross_validate(
            pipe,
            x_train,
            y_train,
            cv=cv,
            scoring={"f1": "f1", "roc_auc": "roc_auc"},
        )
        pipe.fit(x_train, y_train)
        pred = pipe.predict(x_test)
        proba = pipe.predict_proba(x_test)[:, 1]
        tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
        auc_valid = y_test.nunique() == 2
        results[name] = {
            "cv_f1": _r(scores["test_f1"].mean()),
            "cv_f1_std": _r(scores["test_f1"].std(ddof=1)),
            "cv_roc_auc": _r(scores["test_roc_auc"].mean()),
            "test_precision": _r(precision_score(y_test, pred, zero_division=0)),
            "test_recall": _r(recall_score(y_test, pred, zero_division=0)),
            "test_f1": _r(f1_score(y_test, pred, zero_division=0)),
            "test_roc_auc": _r(roc_auc_score(y_test, proba)) if auc_valid else None,
            "confusion_matrix": {
                "labels": ["not_high_risk(0)", "high_risk(1)"],
                "rows_actual_columns_predicted": [[int(tn), int(fp)], [int(fn), int(tp)]],
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            },
        }
        fitted[name] = pipe

    selected, best, threshold = select_simplest_within_one_se(
        results, "cv_f1", True, CLASSIFICATION_COMPLEXITY
    )
    reason = (
        f"{best} had the highest {CV_FOLDS}-fold cross-validated F1 on the training rows "
        f"({results[best]['cv_f1']}). Rule: choose the simplest model within one standard "
        f"error of the best (F1 >= {threshold}). Selected: {selected} "
        f"(CV F1 {results[selected]['cv_f1']}) versus the majority baseline "
        f"({results[CLASSIFICATION_BASELINE]['cv_f1']}). Test rows were used only to report "
        "final scores."
    )
    pred = fitted[selected].predict(x_test)
    proba = fitted[selected].predict_proba(x_test)[:, 1]
    wrong = np.where(pred != y_test.to_numpy())[0]
    most_confident = None
    if len(wrong):
        confidence = np.abs(proba[wrong] - 0.5)
        idx = int(wrong[int(np.argmax(confidence))])
        most_confident = {
            "row_id": int(test.index[idx]),
            "actual_high_risk": int(y_test.iloc[idx]),
            "predicted_high_risk": int(pred[idx]),
            "predicted_probability": _r(proba[idx]),
            "yield_tons": _r(test.iloc[idx][TARGET_REGRESSION]),
            "inputs": {c: test.iloc[idx][c] for c in FEATURES},
        }
    report = {
        "task": "classification",
        "target": TARGET_CLASSIFICATION,
        "positive_class": "high_risk = 1 (yield per hectare below the median)",
        "baseline": CLASSIFICATION_BASELINE,
        "baseline_note": (
            "The majority baseline predicts one class for every row, so its ROC-AUC is 0.5 "
            "(no ranking ability) and its precision or recall can be 0."
        ),
        "selection_metric": f"{CV_FOLDS}-fold CV F1 on training rows",
        "models": results,
        "selected_model": selected,
        "selection_reason": reason,
        "most_confident_test_error": most_confident,
    }
    return {"report": report, "model": fitted[selected], "predictions": pred, "proba": proba}


def _describe_cluster(means: pd.Series, overall_mean: pd.Series, overall_std: pd.Series) -> str:
    labels = {
        "rainfall_mm": "rainfall",
        "avg_temp_c": "temperature",
        "soil_moisture_pct": "soil moisture",
        "fertilizer_kg": "fertilizer use",
        "field_size_ha": "field size",
    }
    z = (means - overall_mean) / overall_std
    strong = z[z.abs() >= 0.5].sort_values(key=lambda s: -s.abs())
    if strong.empty:
        return "Farms close to the overall average on every field condition."
    parts = [("high " if v > 0 else "low ") + labels[k] for k, v in strong.items()]
    return "Farms with " + ", ".join(parts[:3]) + "."


def run_clustering(train: pd.DataFrame, test: pd.DataFrame, seed: int) -> dict:
    """K-means on scaled field conditions; K in 2..6 chosen by silhouette score.

    Targets (yield, risk) are NOT used to form clusters. They are only reported
    afterwards to help interpret each cluster.
    """
    prep = build_cluster_preprocessor()
    x_train = prep.fit_transform(train[NUMERIC_FEATURES])

    scores = {}
    for k in CLUSTER_K_RANGE:
        labels = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(x_train)
        scores[k] = _r(silhouette_score(x_train, labels))
    selected_k = max(scores, key=scores.get)

    model = Pipeline(
        [
            ("prep", build_cluster_preprocessor()),
            ("kmeans", KMeans(n_clusters=selected_k, n_init=10, random_state=seed)),
        ]
    )
    model.fit(train[NUMERIC_FEATURES])
    train_labels = model.predict(train[NUMERIC_FEATURES])
    test_labels = model.predict(test[NUMERIC_FEATURES])

    overall_mean = train[NUMERIC_FEATURES].mean()
    overall_std = train[NUMERIC_FEATURES].std()
    overall_risk = float(train[TARGET_CLASSIFICATION].mean())
    centers = model.named_steps["prep"].named_steps["scale"].inverse_transform(
        model.named_steps["kmeans"].cluster_centers_
    )

    profiles = []
    for cluster in range(selected_k):
        mask = train_labels == cluster
        rows = train[mask]
        means = rows[NUMERIC_FEATURES].mean()
        risk = float(rows[TARGET_CLASSIFICATION].mean())
        meaning = (
            "Visit first: a larger share of these farms is at high production risk."
            if risk >= overall_risk + 0.10
            else "Lower priority: fewer farms here are at high production risk."
            if risk <= overall_risk - 0.10
            else "Average priority: risk is close to the cooperative average."
        )
        profiles.append(
            {
                "cluster": cluster,
                "train_farms": int(mask.sum()),
                "test_farms": int((test_labels == cluster).sum()),
                "mean_conditions": {c: _r(means[c]) for c in NUMERIC_FEATURES},
                "centre_in_original_units": {
                    c: _r(v) for c, v in zip(NUMERIC_FEATURES, centers[cluster])
                },
                "profile": _describe_cluster(means, overall_mean, overall_std),
                "after_the_fact_mean_yield_tons": _r(rows[TARGET_REGRESSION].mean()),
                "after_the_fact_high_risk_rate": _r(risk),
                "operational_meaning": meaning,
            }
        )

    report = {
        "task": "clustering",
        "features_used": NUMERIC_FEATURES,
        "targets_used_to_form_clusters": False,
        "silhouette_by_k": {str(k): v for k, v in scores.items()},
        "selected_k": int(selected_k),
        "selection_reason": (
            f"K={selected_k} has the highest silhouette score ({scores[selected_k]}) "
            "among K=2 to 6."
        ),
        "overall_train_high_risk_rate": _r(overall_risk),
        "profiles": profiles,
        "interpretation": (
            "Silhouette scores are low because the synthetic farm conditions were drawn "
            "independently and uniformly, so there are no natural, well-separated groups. "
            "The clusters are convenient groupings for field visits, not distinct farm types."
        ),
    }
    return {
        "report": report,
        "model": model,
        "train_labels": train_labels,
        "test_labels": test_labels,
    }
