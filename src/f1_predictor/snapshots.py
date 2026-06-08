"""Stage 4: build chronologically split, scaled snapshot training tensors.

Snapshots are extracted at fixed laps from the Stage 3 feature tables. The
StandardScaler is fitted on the train split only; nulls are imputed to 0.0
before scaling. Output: data/snapshots/{train,val,test}.parquet + metadata.json.
"""
from __future__ import annotations

from datetime import datetime

import polars as pl

# The validation season; anything earlier is train.
_VAL_YEAR = 2024


def assign_split(date_start: str, val_cutoff: str) -> str:
    """Classify a race into 'train' | 'val' | 'test' by its start date.

    train: any race before the validation season (2024).
    val:   a 2024 race strictly before val_cutoff.
    test:  a 2024 race on or after val_cutoff.
    """
    dt = datetime.fromisoformat(date_start)
    cutoff = datetime.fromisoformat(val_cutoff).date()
    if dt.year < _VAL_YEAR:
        return "train"
    return "val" if dt.date() < cutoff else "test"
