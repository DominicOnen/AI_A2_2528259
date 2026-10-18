"""Predict from one JSON record using the saved models.

    python predict.py --json samples/valid.json

Exit code 0 on success, 2 on invalid input, 1 if the models are missing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pydantic import ValidationError

from src.inference import Predictor
from src.schemas import format_validation_errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Predict yield, risk and cluster for one farm")
    parser.add_argument("--json", required=True, type=Path, help="path to a JSON record")
    args = parser.parse_args()

    try:
        record = json.loads(args.json.read_text(encoding="utf-8"))
    except FileNotFoundError:
        print(json.dumps({"error": f"File not found: {args.json}"}), file=sys.stderr)
        return 2
    except json.JSONDecodeError as exc:
        print(json.dumps({"error": f"Not valid JSON: {exc}"}), file=sys.stderr)
        return 2

    try:
        predictor = Predictor()
    except FileNotFoundError as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        return 1

    try:
        result = predictor.predict(record)
    except ValidationError as exc:
        print(
            json.dumps({"error": "invalid_input", "problems": format_validation_errors(exc)}, indent=2),
            file=sys.stderr,
        )
        return 2

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
