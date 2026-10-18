"""Evidence for the three written interpretations (one error, one error, one cluster).

    python -m src.error_analysis

Because the data are synthetic, the generator can be re-run to recover the values
that were hidden (missing soil moisture) and the random noise of each farm. This is
only possible for synthetic data; with real data those values would be unknown.
"""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd

from data.generate_data import compute_yield, generate_base_frame

from .config import FEATURES, ROOT


def main() -> None:
    meta = json.loads((ROOT / "models" / "model_metadata.json").read_text(encoding="utf-8"))
    seed = meta["seed"]
    base = generate_base_frame(seed)
    observed = pd.read_csv(ROOT / "data" / meta["dataset"].split("/")[-1])
    reg_model = joblib.load(ROOT / "models" / "regression_model.joblib")
    reg = json.loads((ROOT / "artifacts" / "regression_metrics.json").read_text(encoding="utf-8"))
    clf = json.loads((ROOT / "artifacts" / "classification_metrics.json").read_text(encoding="utf-8"))
    clu = json.loads((ROOT / "artifacts" / "clustering_report.json").read_text(encoding="utf-8"))

    # --- one regression error ---
    err = reg["largest_test_error"]
    rid = err["row_id"]
    truth = base.loc[rid]
    no_noise = compute_yield(
        truth["rainfall_mm"], truth["soil_moisture_pct"], truth["fertilizer_kg"],
        truth["field_size_ha"], truth["avg_temp_c"], truth["cooperative_support"],
        truth["season"], 0.0,
    )
    noise = float(truth["yield_tons"] - no_noise)
    row = observed.loc[[rid], FEATURES].copy()
    imputed = reg_model.named_steps["prep"].named_transformers_["num"].named_steps["impute"].statistics_[2]
    fixed = row.copy()
    fixed["soil_moisture_pct"] = truth["soil_moisture_pct"]
    pred_fixed = float(reg_model.predict(fixed)[0])
    print(f"REGRESSION: row {rid}, actual {err['actual_yield_tons']}, predicted {err['predicted_yield_tons']}, "
          f"error {err['error_tons']} tons")
    print(f"  soil moisture is missing in the CSV; training median used = {imputed:.2f}; true value = "
          f"{truth['soil_moisture_pct']:.2f}")
    print(f"  prediction with the true soil moisture = {pred_fixed:.4f} "
          f"(error {pred_fixed - err['actual_yield_tons']:.4f})")
    print(f"  random noise in this farm's yield = {noise:+.4f} tons")

    # --- one classification error ---
    c = clf["most_confident_test_error"]
    per_ha = base["yield_tons"] / base["field_size_ha"]
    median = float(np.median(per_ha))
    print(f"\nCLASSIFICATION: row {c['row_id']}, actual {c['actual_high_risk']}, predicted "
          f"{c['predicted_high_risk']} (probability {c['predicted_probability']})")
    print(f"  yield per hectare = {c['yield_tons'] / c['inputs']['field_size_ha']:.4f}; "
          f"median = {median:.4f}; field size = {c['inputs']['field_size_ha']} ha")
    flag = per_ha < median
    print(f"  mean field size: high-risk farms {base.loc[flag, 'field_size_ha'].mean():.2f} ha, "
          f"others {base.loc[~flag, 'field_size_ha'].mean():.2f} ha")

    # --- one cluster ---
    top = max(clu["profiles"], key=lambda p: p["after_the_fact_high_risk_rate"])
    print(f"\nCLUSTER {top['cluster']}: {top['profile']}")
    print(f"  train farms {top['train_farms']}, mean conditions {top['mean_conditions']}")
    print(f"  after-the-fact high-risk rate {top['after_the_fact_high_risk_rate']} "
          f"(overall {clu['overall_train_high_risk_rate']}), mean yield "
          f"{top['after_the_fact_mean_yield_tons']} tons")
    print(f"  meaning: {top['operational_meaning']}")


if __name__ == "__main__":
    main()
