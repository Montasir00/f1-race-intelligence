"""Tests for the within-stint tyre degradation estimator."""

import numpy as np
import pandas as pd
import pytest

from f1_analysis.analysis.tyre_analysis import compute_compound_degradation, compute_driver_stint_summary


def _stint(driver, stint, compound, base, slope, n=8):
    age = np.arange(1, n + 1)
    return pd.DataFrame({"driver_id": driver, "stint_number": stint, "compound": compound,
                         "tyre_age": age, "lap_number": age + 10 * stint,
                         "lap_time_sec": base + slope * age})


def test_within_stint_slope_is_not_confounded_by_car_pace():
    # Fast car on old tyres, slow car on new tyres: a pooled regression would
    # produce a NEGATIVE slope. The true degradation is +0.1 s/lap for both.
    fast = _stint("fast", 1, "HARD", base=88.0, slope=0.1)
    fast["tyre_age"] += 15
    slow = _stint("slow", 1, "HARD", base=92.0, slope=0.1)
    df = pd.concat([fast, slow], ignore_index=True)
    df["lap_time_sec"] = df.groupby("driver_id")["lap_time_sec"].transform(lambda s: s)  # unchanged
    deg = compute_compound_degradation(df, use_fuel_corrected=False)
    assert deg.loc[0, "degradation_sec_per_lap"] == pytest.approx(0.1)
    assert deg.loc[0, "n_stints"] == 2


def test_compounds_estimated_separately():
    df = pd.concat([_stint("a", 1, "SOFT", 88, 0.2), _stint("a", 2, "HARD", 90, 0.05)], ignore_index=True)
    deg = compute_compound_degradation(df, use_fuel_corrected=False).set_index("compound")
    assert deg.loc["SOFT", "degradation_sec_per_lap"] == pytest.approx(0.2)
    assert deg.loc["HARD", "degradation_sec_per_lap"] == pytest.approx(0.05)


def test_short_stints_and_unknown_compound_excluded():
    df = pd.concat([_stint("a", 1, "SOFT", 88, 0.2, n=3), _stint("b", 1, np.nan, 88, 0.2)], ignore_index=True)
    assert compute_compound_degradation(df, min_stint_laps=5).empty


def test_stint_summary():
    df = pd.concat([_stint("a", 1, "SOFT", 88, 0.2), _stint("a", 2, "HARD", 90, 0.05)], ignore_index=True)
    s = compute_driver_stint_summary(df, use_fuel_corrected=False)
    assert s["stint_length_laps"].tolist() == [8, 8]
    assert s["degradation_sec_per_lap"].tolist() == pytest.approx([0.2, 0.05])
