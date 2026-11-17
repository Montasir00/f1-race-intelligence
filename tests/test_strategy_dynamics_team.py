"""Tests for pit-stop, race-dynamics and constructor modules."""

import numpy as np
import pandas as pd
import pytest

from f1_analysis.analysis.pitstop_analysis import analyze_undercut, compute_pit_stop_summary, flag_anomalous_stops
from f1_analysis.analysis.race_dynamics import compute_gap_to_leader, compute_net_position_changes
from f1_analysis.analysis.team_performance import compute_constructor_points_summary


def _pits():
    return pd.DataFrame({
        "driver_id": ["ver", "per", "lec", "sai"],
        "lap": [10, 12, 15, 20], "stop_number": [1, 1, 1, 1],
        "pit_lane_time_sec": [22.0, 23.0, 24.0, 1500.0],     # sai: red-flag stop
    })


def test_anomalous_stop_flagged_and_excluded_from_timing(results):
    assert flag_anomalous_stops(_pits())["is_anomalous"].tolist() == [False, False, False, True]
    s = compute_pit_stop_summary(_pits(), results).set_index("driver_id")
    assert s.loc["sai", "total_stops"] == 1 and s.loc["sai", "anomalous_stops"] == 1
    assert np.isnan(s.loc["sai", "median_pit_lane_sec"])


def test_undercut_net_gain():
    # A is 1.0 s behind B; A pits on lap 3 and gains 1.5 s overall by lap 6.
    laps = pd.DataFrame({
        "driver_id": ["A"] * 6 + ["B"] * 6,
        "lap_number": list(range(1, 7)) * 2,
        "lap_time_sec": [91.0, 90.0, 110.0, 88.0, 89.0, 89.0,
                         90.0, 90.0, 90.0, 90.0, 108.5, 89.0],
    })
    pits = pd.DataFrame({"driver_id": ["A", "B"], "lap": [3, 5], "stop_number": [1, 1]})
    r = analyze_undercut(laps, pits, attacker="A", defender="B")
    assert r["gap_before_sec"] == pytest.approx(1.0)
    assert r["gap_after_sec"] == pytest.approx(-0.5)
    assert r["net_gain_sec"] == pytest.approx(1.5)
    assert r["undercut_worked"] is True


def test_undercut_invalid_order_returns_same_keys():
    pits = pd.DataFrame({"driver_id": ["A", "B"], "lap": [5, 3], "stop_number": [1, 1]})
    r = analyze_undercut(pd.DataFrame(columns=["driver_id", "lap_number", "lap_time_sec"]), pits, "A", "B")
    assert np.isnan(r["net_gain_sec"]) and r["undercut_worked"] is None


def test_gap_to_leader_propagates_missing_laps():
    laps = pd.DataFrame({"driver_id": ["A", "A", "A", "B", "B", "B"], "lap_number": [1, 2, 3] * 2,
                         "lap_time_sec": [90, 90, 90, 91, np.nan, 89]})
    g = compute_gap_to_leader(laps).set_index(["driver_id", "lap_number"])
    assert g.loc[("B", 1), "gap_to_leader_sec"] == pytest.approx(1.0)
    assert np.isnan(g.loc[("B", 3), "gap_to_leader_sec"])


def test_net_position_changes_pit_lane_start(results):
    out = compute_net_position_changes(results).set_index("driver_id")
    assert out.loc["per", "pit_lane_start"] and out.loc["per", "effective_grid"] == 4
    assert out.loc["per", "positions_gained"] == 2
    assert out.loc["lec", "finished"] and not out.loc["sai", "finished"]


def test_constructor_points(results):
    out = compute_constructor_points_summary(results)
    assert out.iloc[0]["constructor_name"] == "Red Bull" and out.iloc[0]["total_points"] == 43
    assert out.iloc[0]["double_top10_finish"]
