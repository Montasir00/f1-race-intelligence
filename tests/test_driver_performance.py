"""Tests for driver performance metrics."""

import pandas as pd
import pytest

from f1_analysis.analysis.driver_performance import compare_teammates, compute_driver_pace_metrics


def test_compute_driver_pace_metrics():
    df = pd.DataFrame({
        "driver_id": ["nor"] * 5 + ["pia"] * 5,
        "lap_time_sec": [91.0, 91.2, 91.1, 91.3, 91.0, 91.5, 91.6, 91.7, 91.5, 91.8],
        "is_valid_representative_lap": [True] * 9 + [False],
    })
    m = compute_driver_pace_metrics(df).set_index("driver_id")
    assert m.loc["nor", "median_lap_sec"] == pytest.approx(91.1)
    assert m.loc["nor", "fastest_lap_sec"] == pytest.approx(91.0)
    assert m.loc["nor", "gap_to_fastest_sec"] == pytest.approx(0.0)
    assert m.loc["pia", "representative_laps"] == 4 and m.loc["pia", "total_laps"] == 5
    assert m.loc["pia", "iqr_lap_sec"] == pytest.approx(91.625 - 91.5)


def test_metrics_without_validity_column_do_not_crash():
    df = pd.DataFrame({"driver_id": ["a", "a"], "lap_time_sec": [90.0, 91.0]})
    assert compute_driver_pace_metrics(df).loc[0, "median_lap_sec"] == pytest.approx(90.5)


def test_compare_teammates(results):
    laps = pd.DataFrame({
        "driver_id": ["ver"] * 8 + ["per"] * 8,
        "lap_time_sec": [90.0, 90.1, 89.9, 90.2, 90.0, 90.1, 89.8, 90.0] +
                        [90.5, 90.6, 90.4, 90.7, 90.5, 90.6, 90.3, 90.5],
    })
    comp = compare_teammates(laps, results)
    assert len(comp) == 1                                   # ferrari has no laps -> skipped
    row = comp.iloc[0]
    assert row["faster_driver"] == "ver"
    assert row["delta_sec"] == pytest.approx(-0.5)
    assert 0 <= row["mannwhitney_p_value"] < 0.05
