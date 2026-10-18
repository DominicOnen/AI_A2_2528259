"""Tests for the Musanze Harvest Intelligence Service (failure paths included)."""

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from data.generate_data import base_row_count, generate_dataframe, sha256_of_file, write_dataset
from src.config import FEATURES, FORBIDDEN_FEATURES
from src.identity import check_seed, parse_student_id
from src.pipeline import run
from src.validation import check_no_leakage, check_schema

ROOT = Path(__file__).resolve().parents[1]
STUDENT_ID = "ug2528259@ines.ac.rw"
SEED = 8259
CODE = "AI-A2-8259"


# ---------- 1. valid inference ----------
def test_valid_inference_returns_all_fields(predictor, valid_record):
    result = predictor.predict(valid_record)
    assert result["verification_code"] == CODE
    assert 0.5 <= result["yield_estimate_tons"] <= 30
    assert result["risk_prediction"] in {"high", "not_high"}
    assert 0.0 <= result["high_risk_probability"] <= 1.0
    assert isinstance(result["cluster"], int)
    assert result["cluster_profile"]


# ---------- 2-4. invalid input is rejected ----------
def test_missing_field_is_rejected(predictor, valid_record):
    del valid_record["rainfall_mm"]
    with pytest.raises(ValidationError) as err:
        predictor.predict(valid_record)
    assert "rainfall_mm" in str(err.value)


def test_wrong_type_is_rejected(predictor, valid_record):
    valid_record["avg_temp_c"] = "warm"
    with pytest.raises(ValidationError) as err:
        predictor.predict(valid_record)
    assert "avg_temp_c" in str(err.value)


def test_range_error_is_rejected(predictor, valid_record):
    valid_record["rainfall_mm"] = 2500.0
    with pytest.raises(ValidationError) as err:
        predictor.predict(valid_record)
    assert "rainfall_mm" in str(err.value)


def test_unknown_season_and_extra_field_are_rejected(predictor, valid_record):
    valid_record["season"] = "C"
    valid_record["yield_tons"] = 12.0  # a target must never be accepted as input
    with pytest.raises(ValidationError) as err:
        predictor.predict(valid_record)
    text = str(err.value)
    assert "season" in text and "yield_tons" in text


# ---------- 5. reproducibility ----------
def test_dataset_recreates_same_fingerprint(tmp_path):
    first = write_dataset(SEED, tmp_path / "a.csv")
    second = write_dataset(SEED, tmp_path / "b.csv")
    assert sha256_of_file(first) == sha256_of_file(second)
    stored = (ROOT / "artifacts" / "data_fingerprint.txt").read_text(encoding="utf-8")
    assert f"sha256={sha256_of_file(first)}" in stored


def test_full_pipeline_reproduces_metrics(tmp_path):
    run(STUDENT_ID, SEED, root=tmp_path / "one", quiet=True)
    run(STUDENT_ID, SEED, root=tmp_path / "two", quiet=True)
    for name in ("regression_metrics.json", "classification_metrics.json", "clustering_report.json"):
        a = json.loads((tmp_path / "one" / "artifacts" / name).read_text(encoding="utf-8"))
        b = json.loads((tmp_path / "two" / "artifacts" / name).read_text(encoding="utf-8"))
        assert a == b


# ---------- data problems and leakage ----------
def test_dataset_has_the_deliberate_problems():
    df = generate_dataframe(SEED)
    base = base_row_count(SEED)
    m = round(0.05 * base)
    assert base == 258 and len(df) == base + 3
    assert df["soil_moisture_pct"].isna().sum() == m
    assert df["fertilizer_kg"].isna().sum() == m
    assert df.duplicated().sum() == 3


def test_targets_are_never_inputs():
    assert not set(FEATURES) & FORBIDDEN_FEATURES
    with pytest.raises(ValueError):
        check_no_leakage(FEATURES + ["yield_tons"])


def test_schema_check_rejects_missing_column():
    df = generate_dataframe(SEED).drop(columns=["season"])
    with pytest.raises(ValueError, match="season"):
        check_schema(df)


# ---------- identity and seed rule ----------
def test_wrong_seed_is_rejected():
    identity = parse_student_id(STUDENT_ID)
    assert identity.llll == SEED and identity.verification_code == CODE
    with pytest.raises(ValueError, match="8259"):
        check_seed(identity, 1234)


# ---------- CLI failure path ----------
def test_cli_invalid_record_exits_with_error():
    proc = subprocess.run(
        [sys.executable, "predict.py", "--json", "samples/invalid.json"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 2
    assert "invalid_input" in proc.stderr


def test_cli_valid_record_prints_json():
    proc = subprocess.run(
        [sys.executable, "predict.py", "--json", "samples/valid.json"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert json.loads(proc.stdout)["verification_code"] == CODE


# ---------- API ----------
@pytest.fixture()
def client(saved_models):
    from app import app

    with TestClient(app) as test_client:
        yield test_client


def test_api_predict_returns_expected_response(client, valid_record):
    response = client.post("/predict", json=valid_record)
    assert response.status_code == 200
    body = response.json()
    assert body["verification_code"] == CODE
    assert set(body) >= {
        "yield_estimate_tons",
        "risk_prediction",
        "high_risk_probability",
        "cluster",
        "verification_code",
    }


def test_api_invalid_request_returns_422_with_field_names(client, valid_record):
    valid_record["rainfall_mm"] = 5000
    valid_record["season"] = "Z"
    response = client.post("/predict", json=valid_record)
    assert response.status_code == 422
    fields = {p["field"] for p in response.json()["problems"]}
    assert {"rainfall_mm", "season"} <= fields


def test_api_rejects_non_json_body(client):
    response = client.post(
        "/predict", content="not json", headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 422
    assert response.json()["error"] == "invalid_input"


def test_api_health_and_model_info_show_code(client):
    assert client.get("/health").json() == {"status": "ok", "verification_code": CODE}
    info = client.get("/model-info").json()
    assert info["verification_code"] == CODE
    assert info["synthetic_data"] is True
    assert len(info["dataset_sha256"]) == 64
