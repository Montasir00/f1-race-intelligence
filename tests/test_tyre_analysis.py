"""Unit tests for tyre degradation regression analysis."""

import numpy as np
import pandas as pd
import pytest

from src.f1_analysis.analysis.tyre_analysis import compute_compound_degradation, _fit_linear_regression


def test_fit_linear_regression():
    """Verify OLS linear regression calculations."""
    x = np.array([1, 2, 3, 4, 5], dtype=float)
    # y = 100 + 0.1 * x
    y = 100.0 + 0.1 * x

    fit = _fit_linear_regression(x, y)
    assert fit["intercept"] == pytest.approx(100.0)
    assert fit["slope"] == pytest.approx(0.1)
    assert fit["r_squared"] == pytest.approx(1.0)


def test_compute_compound_degradation():
    """Verify compound degradation slope calculation across compounds."""
    df = pd.DataFrame({
        "compound": ["SOFT"] * 10 + ["HARD"] * 10,
        "tyre_age": list(range(1, 11)) * 2,
        # Soft degrades faster (+0.2s/lap), Hard degrades slower (+0.05s/lap)
        "lap_time_sec": [88.0 + 0.2 * i for i in range(1, 11)] + [90.0 + 0.05 * i for i in range(1, 11)],
        "fuel_corrected_lap_sec": [88.0 + 0.2 * i for i in range(1, 11)] + [90.0 + 0.05 * i for i in range(1, 11)],
        "is_valid_representative_lap": [True] * 20,
    })

    deg = compute_compound_degradation(df, min_stint_laps=5, use_fuel_corrected=True)
    assert len(deg) == 2
    soft_row = deg[deg["compound"] == "SOFT"].iloc[0]
    hard_row = deg[deg["compound"] == "HARD"].iloc[0]

    assert soft_row["degradation_slope_sec_per_lap"] == pytest.approx(0.2)
    assert hard_row["degradation_slope_sec_per_lap"] == pytest.approx(0.05)
    assert soft_row["r_squared"] == pytest.approx(1.0)
