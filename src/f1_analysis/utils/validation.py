"""Small validation helpers shared by the analysis and visualization modules."""

from typing import Iterable

import pandas as pd

VALID_LAP_COL = "is_valid_representative_lap"


def require_columns(df: pd.DataFrame, columns: Iterable[str], name: str = "DataFrame") -> None:
    """Raise a clear ValueError if any required column is missing.

    Args:
        df: DataFrame to check.
        columns: Column names that must be present.
        name: Human-readable name used in the error message.
    """
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def representative_laps(df: pd.DataFrame) -> pd.DataFrame:
    """Return only representative laps if the validity flag exists, else all rows.

    Replaces the buggy pattern ``df[df.get(col, True)]`` which raises
    ``KeyError: True`` when the column is absent.
    """
    if VALID_LAP_COL in df.columns:
        return df[df[VALID_LAP_COL].astype(bool)].copy()
    return df.copy()


def pace_column(df: pd.DataFrame, use_fuel_corrected: bool) -> str:
    """Pick the lap-time column to analyse, falling back to raw times."""
    if use_fuel_corrected and "fuel_corrected_lap_sec" in df.columns:
        return "fuel_corrected_lap_sec"
    return "lap_time_sec"
