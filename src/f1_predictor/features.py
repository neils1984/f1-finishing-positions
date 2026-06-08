"""Stage 3: engineer features per race from the Stage 2 sessionised table.

Pure, deterministic transforms producing raw (unscaled) human-readable values.
Scaling happens in Stage 4. Cross-race priors live in priors.py.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl
import yaml

_CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"


def load_circuits(path: Path | None = None) -> dict:
    """Load the circuit reference (lengths + street-circuit list)."""
    path = path or (_CONFIG_DIR / "circuits.yaml")
    with open(path) as f:
        return yaml.safe_load(f)


def circuit_length_km(circuit_short_name: str, circuits: dict) -> float | None:
    """Track length in km for a circuit_short_name, or None if unknown."""
    return circuits.get("lengths_km", {}).get(circuit_short_name)


def is_street_circuit(circuit_short_name: str, circuits: dict) -> bool:
    """True if the circuit is a street circuit (Baku/Singapore/Las Vegas/Miami)."""
    return circuit_short_name in set(circuits.get("street", []))


_GAP_COLUMNS = ["gap_to_leader", "interval_to_ahead"]


def _parse_gap_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Coerce gap columns to Float64; non-numeric markers (e.g. '+1 LAP') -> null.

    pl.col(...).cast(Float64, strict=False) turns any value that doesn't parse as
    a number into null, which is exactly the desired behaviour for '+N LAP(S)',
    '' and existing nulls. Already-Float64 columns pass through unchanged.
    """
    exprs = []
    for col in _GAP_COLUMNS:
        if col in df.columns and df.schema[col] != pl.Float64:
            exprs.append(pl.col(col).cast(pl.Float64, strict=False).alias(col))
    return df.with_columns(exprs) if exprs else df
