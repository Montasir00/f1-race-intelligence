"""Unit tests for data loader utilities and preprocessing pipelines."""

import numpy as np
import pandas as pd
import pytest

from src.f1_analysis.data.data_loader import time_to_seconds
from src.f1_analysis.data.preprocessing import clean_lap_times, apply_fuel_correction, merge_laps_with_stints


def test_time_to_seconds():
    """Verify conversion of F1 timing strings to float seconds."""
    assert time_to_seconds("1:32.408") == pytest.approx(92.408)
    assert time_to_seconds("24.512") == pytest.approx(24.512)
    assert time_to_seconds("1:20:15.500") == pytest.approx(4815.5)
    assert time_to_seconds(None) is None
    assert time_to_seconds("") is None
    assert time_to_seconds("invalid") is None


def test_clean_lap_times_filters_outliers():
    """Verify that slow laps (e.g., pit entry, safety car) are flagged as non-representative."""
    df = pd.DataFrame({
        "driver_id": ["verstappen"] * 5,
        "lap_number": [1, 2, 3, 4, 5],
        "lap_time_sec": [90.0, 90.2, 130.0, 90.1, 90.3],  # Lap 3 is +44% slower
    })

    cleaned = clean_lap_times(df, threshold_factor=1.15)
    assert "is_valid_representative_lap" in cleaned.columns
    # Laps 1, 2, 4, 5 should be valid; lap 3 should be flagged False
    assert cleaned.loc[cleaned["lap_number"] == 3, "is_valid_representative_lap"].values[0] == False
    assert cleaned.loc[cleaned["lap_number"] == 1, "is_valid_representative_lap"].values[0] == True


def test_apply_fuel_correction():
    """Verify fuel load mass correction math."""
    df = pd.DataFrame({
        "driver_id": ["leclerc", "leclerc"],
        "lap_number": [1, 2],
        "lap_time_sec": [90.0, 90.0],
    })

    corrected = apply_fuel_correction(df, correction_per_lap=0.035)
    # Lap 1: 90.0 + (1-1)*0.035 = 90.0
    # Lap 2: 90.0 + (2-1)*0.035 = 90.035
    assert corrected.loc[0, "fuel_corrected_lap_sec"] == pytest.approx(90.0)
    assert corrected.loc[1, "fuel_corrected_lap_sec"] == pytest.approx(90.035)
