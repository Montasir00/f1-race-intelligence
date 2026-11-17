"""Tests for parsing helpers and the preprocessing pipeline."""

import numpy as np
import pandas as pd
import pytest

from f1_analysis.data.data_loader import time_to_seconds
from f1_analysis.data.preprocessing import (
    apply_fuel_correction,
    clean_lap_times,
    flag_pit_laps,
    merge_laps_with_stints,
)


@pytest.mark.parametrize("text, expected", [
    ("1:32.408", 92.408), ("24.512", 24.512), ("1:20:15.500", 4815.5),
    (None, None), ("", None), ("invalid", None), ("1:2:3:4", None),
])
def test_time_to_seconds(text, expected):
    result = time_to_seconds(text)
    assert result == (pytest.approx(expected) if expected is not None else None)


def _laps():
    return pd.DataFrame({
        "driver_id": ["ver"] * 6,
        "lap_number": [1, 2, 3, 4, 5, 6],
        "lap_time_sec": [98.0, 90.0, 90.2, 112.0, 90.1, np.nan],
    })


def test_flag_pit_laps():
    pits = pd.DataFrame({"driver_id": ["ver"], "lap": [3]})
    df = flag_pit_laps(_laps(), pits)
    assert df.loc[df.lap_number == 3, "is_pit_in_lap"].item()
    assert df.loc[df.lap_number == 4, "is_pit_out_lap"].item()
    assert df["is_pit_in_lap"].sum() == 1 and df["is_pit_out_lap"].sum() == 1


def test_clean_lap_times_reasons_and_keeps_rows():
    pits = pd.DataFrame({"driver_id": ["ver"], "lap": [3]})
    df = clean_lap_times(flag_pit_laps(_laps(), pits))
    assert len(df) == 6                                   # nothing deleted
    reasons = df.set_index("lap_number")["exclusion_reason"]
    assert reasons[1] == "first_lap"
    assert reasons[3] == "pit_in_lap"
    assert reasons[4] == "pit_out_lap"
    assert reasons[6] == "missing_time"
    assert df.set_index("lap_number").loc[[2, 5], "is_valid_representative_lap"].all()


def test_clean_lap_times_flags_slow_lap():
    df = pd.DataFrame({"driver_id": ["a"] * 5, "lap_number": [2, 3, 4, 5, 6],
                       "lap_time_sec": [90.0, 90.1, 130.0, 90.2, 90.0]})
    out = clean_lap_times(df)
    assert out.loc[out.lap_number == 4, "exclusion_reason"].item() == "slow_lap"


def test_fuel_correction_uses_race_distance():
    df = pd.DataFrame({"lap_number": [1, 2, 50], "lap_time_sec": [90.0, 90.0, 90.0]})
    out = apply_fuel_correction(df, fuel_effect_sec_per_kg=0.03, start_fuel_kg=100.0)
    per_lap = 0.03 * 100.0 / 50                           # 0.06 s/lap
    assert out["fuel_corrected_lap_sec"].tolist() == pytest.approx([90.0, 90.0 + per_lap, 90.0 + 49 * per_lap])


def test_merge_laps_with_stints(results, stints):
    laps = pd.DataFrame({"driver_id": ["ver"] * 6 + ["lec"], "lap_number": [1, 2, 3, 4, 5, 6, 1]})
    out = merge_laps_with_stints(laps, stints, results)
    ver = out[out.driver_id == "ver"].set_index("lap_number")
    assert ver.loc[1, "compound"] == "SOFT" and ver.loc[1, "tyre_age"] == 1
    assert ver.loc[3, "tyre_age"] == 3
    assert ver.loc[4, "compound"] == "HARD" and ver.loc[4, "tyre_age"] == 3  # used set: age 2 at start
    assert ver.loc[6, "compound"] == "HARD"                                  # open-ended last stint
    lec = out[out.driver_id == "lec"]
    assert lec["compound"].isna().all() and lec["tyre_age"].isna().all()     # no invented values
    assert len(out) == len(laps)


def test_merge_without_stints_leaves_nan(results):
    laps = pd.DataFrame({"driver_id": ["ver"], "lap_number": [1]})
    out = merge_laps_with_stints(laps, pd.DataFrame(), results)
    assert out["compound"].isna().all() and out["tyre_age"].isna().all()
