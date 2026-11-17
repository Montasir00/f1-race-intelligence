"""Shared test fixtures: small, hand-checkable race tables."""

import pandas as pd
import pytest


@pytest.fixture
def results() -> pd.DataFrame:
    return pd.DataFrame({
        "position": [1, 2, 3, 4],
        "driver_id": ["ver", "per", "lec", "sai"],
        "driver_number": [1, 11, 16, 55],
        "driver_name": ["Max V", "Sergio P", "Charles L", "Carlos S"],
        "constructor_id": ["red_bull", "red_bull", "ferrari", "ferrari"],
        "constructor_name": ["Red Bull", "Red Bull", "Ferrari", "Ferrari"],
        "grid": [1, 0, 2, 4],
        "status": ["Finished", "Finished", "+1 Lap", "Retired"],
        "points": [25.0, 18.0, 15.0, 0.0],
    })


@pytest.fixture
def stints() -> pd.DataFrame:
    # ver: SOFT laps 1-3 (new), HARD laps 4-6 (lap_end unknown -> race end)
    return pd.DataFrame({
        "driver_number": [1, 1],
        "stint_number": [1, 2],
        "compound": ["SOFT", "HARD"],
        "tyre_age_at_start": [0, 2],
        "lap_start": [1, 4],
        "lap_end": [3, None],
    })
