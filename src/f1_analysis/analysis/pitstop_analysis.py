"""Pit-stop strategy and time loss analysis module for F1 Race Intelligence.

Analyzes stationary pit-stop durations, pit-lane transit loss, undercut/overcut
strategic timing, and compound stint sequences.
"""

import logging
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_pit_stop_summary(
    df_pit_stops: pd.DataFrame,
    df_results: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Compute per-driver and race-wide pit-stop duration statistics.

    Args:
        df_pit_stops: DataFrame containing ['driver_id', 'lap', 'stop_number', 'duration_sec'].
        df_results: Optional race classification results to enrich with team and driver names.

    Returns:
        DataFrame summarizing total stops, fastest stop, median duration, and total stationary time.
    """
    if df_pit_stops.empty:
        return pd.DataFrame()

    df = df_pit_stops.copy()
    valid_stops = df[df["duration_sec"].notna() & (df["duration_sec"] > 0)]

    records = []
    for driver_id, group in valid_stops.groupby("driver_id"):
        durations = group["duration_sec"].values
        stops_count = len(durations)
        fastest_stop = float(np.min(durations))
        median_stop = float(np.median(durations))
        total_time = float(np.sum(durations))

        records.append({
            "driver_id": driver_id,
            "total_stops": stops_count,
            "fastest_stop_sec": fastest_stop,
            "median_stop_sec": median_stop,
            "total_pit_time_sec": total_time,
            "pit_laps": list(group["lap"].values),
        })

    summary_df = pd.DataFrame(records)
    if not summary_df.empty:
        if df_results is not None and not df_results.empty:
            summary_df = summary_df.merge(
                df_results[["driver_id", "driver_name", "constructor_name"]],
                on="driver_id",
                how="left",
            )
        summary_df.sort_values(by="median_stop_sec", inplace=True)
        summary_df.reset_index(drop=True, inplace=True)

    return summary_df


def analyze_undercut_potential(
    df_laps: pd.DataFrame,
    driver_ahead_id: str,
    driver_behind_id: str,
    pit_lap: int,
) -> Dict[str, Optional[float]]:
    """Analyze the efficacy of an undercut attempt between two drivers.

    The undercut works when the pursuing car pits early onto fresh tyres,
    delivering a faster 'out-lap' than the old tyres of the car staying out.

    Args:
        df_laps: Cleaned lap times DataFrame.
        driver_ahead_id: Driver leading on track.
        driver_behind_id: Driver making the early pit call (the undercut attacker).
        pit_lap: The lap the attacker pitted.

    Returns:
        Dictionary detailing in-lap delta, out-lap delta, and net track time advantage.
    """
    col = "lap_time_sec"
    laps_ahead = df_laps[df_laps["driver_id"] == driver_ahead_id].set_index("lap_number")
    laps_behind = df_laps[df_laps["driver_id"] == driver_behind_id].set_index("lap_number")

    out_lap_attacker = pit_lap + 1
    if out_lap_attacker not in laps_ahead.index or out_lap_attacker not in laps_behind.index:
        return {"attacker": driver_behind_id, "defender": driver_ahead_id, "delta_sec": None}

    # Net difference on the critical out-lap
    attacker_time = laps_behind.loc[out_lap_attacker, col]
    defender_time = laps_ahead.loc[out_lap_attacker, col]

    delta = defender_time - attacker_time if (pd.notna(attacker_time) and pd.notna(defender_time)) else None

    return {
        "attacker": driver_behind_id,
        "defender": driver_ahead_id,
        "pit_lap": pit_lap,
        "out_lap": out_lap_attacker,
        "attacker_time_sec": float(attacker_time) if pd.notna(attacker_time) else None,
        "defender_time_sec": float(defender_time) if pd.notna(defender_time) else None,
        "undercut_gain_sec": float(delta) if delta is not None else None,
        "undercut_successful": bool(delta > 0) if delta is not None else False,
    }
