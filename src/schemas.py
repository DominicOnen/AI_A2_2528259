"""Typed request/response models shared by the CLI and the API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .config import RANGES


class FarmRecord(BaseModel):
    """One farm record. Strict types: numbers must be numbers, not strings."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        json_schema_extra={
            "examples": [
                {
                    "rainfall_mm": 1000.0,
                    "avg_temp_c": 18.5,
                    "soil_moisture_pct": 55.0,
                    "fertilizer_kg": 120.0,
                    "field_size_ha": 1.8,
                    "season": "A",
                    "cooperative_support": 1,
                }
            ]
        },
    )

    rainfall_mm: float = Field(ge=RANGES["rainfall_mm"][0], le=RANGES["rainfall_mm"][1])
    avg_temp_c: float = Field(ge=RANGES["avg_temp_c"][0], le=RANGES["avg_temp_c"][1])
    soil_moisture_pct: float = Field(
        ge=RANGES["soil_moisture_pct"][0], le=RANGES["soil_moisture_pct"][1]
    )
    fertilizer_kg: float = Field(ge=RANGES["fertilizer_kg"][0], le=RANGES["fertilizer_kg"][1])
    field_size_ha: float = Field(ge=RANGES["field_size_ha"][0], le=RANGES["field_size_ha"][1])
    season: Literal["A", "B"]
    cooperative_support: int = Field(ge=0, le=1, description="0 = no, 1 = yes")


class PredictionResponse(BaseModel):
    """Prediction returned by the CLI and POST /predict."""

    yield_estimate_tons: float
    risk_prediction: Literal["high", "not_high"]
    high_risk_probability: float
    cluster: int
    cluster_profile: str
    verification_code: str
    disclaimer: str


def format_validation_errors(exc: ValidationError) -> list[dict]:
    """Short, safe messages: which field, and what is wrong."""
    problems = []
    for err in exc.errors():
        field = ".".join(str(p) for p in err["loc"]) or "record"
        problems.append({"field": field, "message": err["msg"]})
    return problems
