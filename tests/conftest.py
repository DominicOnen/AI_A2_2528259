"""Test setup: make the project importable and make sure saved models exist."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

STUDENT_ID = "ug2528259@ines.ac.rw"
SEED = 8259


@pytest.fixture(scope="session", autouse=True)
def saved_models():
    """If artifacts are missing (fresh clone), build them once with the real pipeline."""
    needed = ROOT / "models" / "model_metadata.json"
    if not needed.exists():
        from src.pipeline import run

        run(STUDENT_ID, SEED, quiet=True)
    return ROOT


@pytest.fixture(scope="session")
def predictor(saved_models):
    from src.inference import Predictor

    return Predictor()


@pytest.fixture()
def valid_record():
    import json

    return json.loads((ROOT / "samples" / "valid.json").read_text(encoding="utf-8"))
