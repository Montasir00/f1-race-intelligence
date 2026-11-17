"""Module D - race dynamics: positions, gaps and position changes."""

import logging

import numpy as np
import pandas as pd

from f1_analysis.utils.validation import require_columns

logger = logging.getLogger(__name__)


def compute_position_progression(df_laps: pd.DataFrame) -> pd.DataFrame:
    """Pivot table: index lap_number, columns driver_id, values position."""
    require_columns(df_laps, ["lap_number", "driver_id", "position"], "df_laps")
    valid = df_laps.dropna(subset=["position"])
    return valid.pivot_table(index="lap_number", columns="driver_id", values="position", aggfunc="first")


def compute_net_position_changes(df_results: pd.DataFrame) -> pd.DataFrame:
    """Positions gained = effective grid - finishing position.

    * Pit-lane starters have ``grid == 0`` in Jolpica; they are kept, flagged,
      and given an effective grid equal to the number of starters.
    * ``finished`` is False for retirements, whose "position" is a
      classification order, not a race result - interpret those rows carefully.
    """
    require_columns(df_results, ["driver_id", "grid", "position", "status"], "df_results")
    df = df_results.dropna(subset=["position", "grid"]).copy()
    df["pit_lane_start"] = df["grid"] == 0
    df["effective_grid"] = df["grid"].where(~df["pit_lane_start"], len(df))
    status = df["status"].fillna("")
    df["finished"] = status.isin(["Finished", "Lapped"]) | status.str.startswith("+")
    df["positions_gained"] = df["effective_grid"] - df["position"]
    cols = ["driver_id", "driver_name", "constructor_name", "grid", "pit_lane_start",
            "effective_grid", "position", "finished", "status", "positions_gained"]
    return df[[c for c in cols if c in df.columns]].sort_values("positions_gained", ascending=False).reset_index(drop=True)


def compute_gap_to_leader(df_laps: pd.DataFrame) -> pd.DataFrame:
    """Add ``elapsed_time_sec`` and ``gap_to_leader_sec`` for each driver and lap.

    Elapsed time is the cumulative sum of lap times. If a lap time is missing,
    all later elapsed times for that driver become NaN (``skipna=False``) instead
    of silently shrinking the gap. The leader on lap n is the driver with the
    smallest elapsed time after n laps. Grid-slot offsets at the start are ignored.
    """
    require_columns(df_laps, ["lap_number", "driver_id", "lap_time_sec"], "df_laps")
    df = df_laps.sort_values(["driver_id", "lap_number"]).copy()
    df["elapsed_time_sec"] = df.groupby("driver_id")["lap_time_sec"].transform(lambda s: s.cumsum(skipna=False))
    leader = df.groupby("lap_number")["elapsed_time_sec"].transform("min")
    df["gap_to_leader_sec"] = df["elapsed_time_sec"] - leader
    return df
