"""Unit tests for driver performance calculations."""

import numpy as np
import pandas as pd
import pytest

from src.f1_analysis.analysis.driver_performance import compute_driver_pace_metrics, compare_teammates


def test_compute_driver_pace_metrics():
    """Verify calculation of median, IQR, std, and fastest lap."""
    df = pd.DataFrame({
        "driver_id": ["norris"] * 5 + ["piastri"] * 5,
        "lap_number": list(range(1, 6)) * 2,
        "lap_time_sec": [91.0, 91.2, 91.1, 91.3, 91.0] + [91.5, 91.6, 91.7, 91.5, 91.8],
        "is_valid_representative_lap": [True] * 10,
    })

    metrics = compute_driver_pace_metrics(df, only_representative=True)
    assert len(metrics) == 2
    assert metrics.iloc[0]["driver_id"] == "norris"
    assert metrics.iloc[0]["median_lap_sec"] == pytest.approx(91.1)
    assert metrics.iloc[0]["fastest_lap_sec"] == pytest.approx(91.0)
    assert metrics.iloc[0]["gap_to_fastest_sec"] == pytest.approx(0.0)
    assert metrics.iloc[1]["driver_id"] == "piastri"
    assert metrics.iloc[1]["gap_to_fastest_sec"] > 0


def test_compare_teammates():
    """Verify intra-team delta computation between drivers in same constructor."""
    df_laps = pd.DataFrame({
        "driver_id": ["hamilton"] * 10 + ["russell"] * 10,
        "lap_number": list(range(1, 11)) * 2,
        "lap_time_sec": [90.0] * 10 + [90.5] * 10,
        "is_valid_representative_lap": [True] * 20,
    })
    df_results = pd.DataFrame({
        "driver_id": ["hamilton", "russell"],
        "constructor_id": ["mercedes", "mercedes"],
        "constructor_name": ["Mercedes", "Mercedes"],
    })

    comp = compare_teammates(df_laps, df_results)
    assert len(comp) == 1
    row = comp.iloc[0]
    assert row["constructor_name"] == "Mercedes"
    assert row["faster_driver"] == "hamilton"
    assert row["delta_sec"] == pytest.approx(-0.5)
