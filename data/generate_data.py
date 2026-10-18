"""Synthetic Musanze potato-cooperative dataset generator.

The same script recreates the submitted CSV on a clean computer.

Rules (from the assignment brief), with LLLL = last four digits of the
registration number used as the random seed:

* ``numpy.random.default_rng(LLLL)`` is the only source of randomness.
* base rows = 240 + (LLLL mod 41)
* yield_tons follows the stated formula, clipped at 0.50 tons
* high_risk = 1 when yield per hectare is below its median
* m = round(0.05 * base rows) unique positions are set to missing in
  soil_moisture_pct, another m in fertilizer_kg, then three exact duplicate
  rows are appended
* numeric values are saved with four decimal places

Usage::

    python data/generate_data.py --seed 8259 --out data/AI_A2_2528259.csv
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS = [
    "rainfall_mm",
    "avg_temp_c",
    "soil_moisture_pct",
    "fertilizer_kg",
    "field_size_ha",
    "season",
    "cooperative_support",
    "yield_tons",
    "high_risk",
]


def base_row_count(seed: int) -> int:
    """Number of base rows before duplicates: 240 + (seed mod 41)."""
    return 240 + (seed % 41)


def compute_yield(rainfall, soil, fertilizer, field_size, temp, support, season, noise):
    """Yield formula from the brief, clipped at a minimum of 0.50 tons."""
    season_effect = np.where(season == "A", 0.50, -0.20)
    value = (
        0.004 * rainfall
        + 0.06 * soil
        + 0.02 * fertilizer
        + 1.30 * field_size
        - 0.25 * np.abs(temp - 18)
        + 0.70 * support
        + season_effect
        + noise
    )
    return np.clip(value, 0.50, None)


def _draw_base(seed: int):
    """Draw the base rows. Returns (dataframe, rng) so later steps continue the sequence."""
    rng = np.random.default_rng(seed)
    n = base_row_count(seed)

    # The order of these draws is part of the personal data signature.
    rainfall = rng.uniform(600, 1400, n)
    temp = rng.uniform(12, 26, n)
    soil = rng.uniform(20, 80, n)
    fertilizer = rng.uniform(0, 250, n)
    field_size = rng.uniform(0.2, 4.0, n)
    season = rng.choice(np.array(["A", "B"]), size=n)
    support = rng.integers(0, 2, size=n)
    noise = rng.normal(0.0, 0.80, n)

    yield_tons = compute_yield(
        rainfall, soil, fertilizer, field_size, temp, support, season, noise
    )
    yield_per_ha = yield_tons / field_size
    high_risk = (yield_per_ha < np.median(yield_per_ha)).astype(int)

    df = pd.DataFrame(
        {
            "rainfall_mm": rainfall,
            "avg_temp_c": temp,
            "soil_moisture_pct": soil,
            "fertilizer_kg": fertilizer,
            "field_size_ha": field_size,
            "season": season,
            "cooperative_support": support,
            "yield_tons": yield_tons,
            "high_risk": high_risk,
        }
    )
    return df, rng


def generate_base_frame(seed: int) -> pd.DataFrame:
    """The base rows before missing values and duplicates (used for error analysis)."""
    return _draw_base(seed)[0]


def generate_dataframe(seed: int) -> pd.DataFrame:
    """Create the dataset (with deliberate missing values and duplicates)."""
    df, rng = _draw_base(seed)
    n = len(df)

    # Deliberate data problems: missing values, then exact duplicates.
    m = round(0.05 * n)
    soil_missing = rng.choice(n, size=m, replace=False)
    fert_missing = rng.choice(n, size=m, replace=False)
    df.loc[soil_missing, "soil_moisture_pct"] = np.nan
    df.loc[fert_missing, "fertilizer_kg"] = np.nan

    duplicate_positions = rng.choice(n, size=3, replace=False)
    df = pd.concat([df, df.iloc[duplicate_positions]], ignore_index=True)
    return df[COLUMNS]


def write_dataset(seed: int, out_path: Path) -> Path:
    """Write the CSV with four decimals and Unix line endings."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = generate_dataframe(seed)
    df.to_csv(out_path, index=False, float_format="%.4f", lineterminator="\n")
    return out_path


def sha256_of_file(path: Path) -> str:
    """SHA-256 of the exact CSV bytes."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, required=True, help="LLLL, last four digits")
    parser.add_argument("--out", type=Path, required=True, help="output CSV path")
    args = parser.parse_args()
    path = write_dataset(args.seed, args.out)
    print(f"Wrote {path}")
    print(f"SHA-256: {sha256_of_file(path)}")


if __name__ == "__main__":
    main()
