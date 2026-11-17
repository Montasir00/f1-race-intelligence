"""Race dynamics and progression module for F1 Race Intelligence.

Analyzes lap-by-lap track position evolution, net position changes from grid,
pace trends over the race distance, and gap intervals to the leader.
"""

import logging
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_position_progression(df_laps: pd.DataFrame) -> pd.DataFrame:
    """Generate a pivot table of lap-by-lap track positions for all drivers.

    Index: lap_number
    Columns: driver_id
    Values: position (1 to 20)

    Args:
        df_laps: DataFrame containing ['lap_number', 'driver_id', 'position'].

    Returns:
        DataFrame pivot table suitable for position progression charts.
    """
    if df_laps.empty or "position" not in df_laps.columns:
        return pd.DataFrame()

    valid = df_laps[df_laps["position"].notna()].copy()
    pivot = valid.pivot(index="lap_number", columns="driver_id", values="position")
    return pivot


def compute_net_position_changes(df_results: pd.DataFrame) -> pd.DataFrame:
    """Calculate positions gained or lost by each driver relative to their starting grid slot.

    Formula: positions_gained = grid_position - final_position
    (Positive number indicates positions gained; negative indicates positions lost).

    Args:
        df_results: DataFrame with ['driver_id', 'driver_name', 'grid', 'position'].

    Returns:
        DataFrame sorted by positions gained descending.
    """
    if df_results.empty or "grid" not in df_results.columns or "position" not in df_results.columns:
        return pd.DataFrame()

    df = df_results.copy()
    valid = df[df["grid"].notna() & df["position"].notna() & (df["grid"] > 0)].copy()

    valid["positions_gained"] = valid["grid"] - valid["position"]
    valid.sort_values(by="positions_gained", ascending=False, inplace=True)
    valid.reset_index(drop=True, inplace=True)

    return valid[["driver_id", "driver_name", "constructor_name", "grid", "position", "positions_gained"]]


def compute_gap_to_leader(df_laps: pd.DataFrame) -> pd.DataFrame:
    """Calculate cumulative elapsed race time and time gap to the race leader for every lap.

    Args:
        df_laps: DataFrame with ['lap_number', 'driver_id', 'lap_time_sec'].

    Returns:
        DataFrame with ['lap_number', 'driver_id', 'elapsed_time_sec', 'gap_to_leader_sec'].
    """
    if df_laps.empty or "lap_time_sec" not in df_laps.columns:
        return pd.DataFrame()

    df = df_laps.copy().sort_values(by=["driver_id", "lap_number"])
    # Cumulative elapsed time per driver
    df["elapsed_time_sec"] = df.groupby("driver_id")["lap_time_sec"].cumsum()

    # Find the minimum elapsed time per lap (leader's time)
    leader_times = df.groupby("lap_number")["elapsed_time_sec"].transform("min")
    df["gap_to_leader_sec"] = df["elapsed_time_sec"] - leader_times

    return df
