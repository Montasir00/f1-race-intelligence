"""Data preprocessing and feature engineering module for F1 Race Intelligence.

Implements cleaning pipelines, outlier detection, fuel load mass correction,
and dataset mergers between timing and stint data.
"""

import logging
from typing import Optional
import numpy as np
import pandas as pd

from src.f1_analysis.config import ESTIMATED_FUEL_CORRECTION_SEC_PER_LAP

logger = logging.getLogger(__name__)


def clean_lap_times(
    df_laps: pd.DataFrame,
    threshold_factor: float = 1.15,
) -> pd.DataFrame:
    """Filter out non-representative laps (e.g. out-laps, in-laps, Safety Car, accidents).

    Uses a robust threshold based on the driver's median lap time. Any lap slower than
    `median * threshold_factor` is flagged as an outlier / abnormal lap.

    Args:
        df_laps: DataFrame containing at least ['driver_id', 'lap_number', 'lap_time_sec'].
        threshold_factor: Multiplier over the driver's median lap time (default 1.15 = +15%).

    Returns:
        DataFrame with additional boolean column 'is_valid_representative_lap'.
    """
    df = df_laps.copy()
    if df.empty or "lap_time_sec" not in df.columns:
        df["is_valid_representative_lap"] = False
        return df

    # Drop null lap times
    valid_mask = df["lap_time_sec"].notna() & (df["lap_time_sec"] > 0)
    df["is_valid_representative_lap"] = False

    # Compute driver-specific median pace
    medians = df[valid_mask].groupby("driver_id")["lap_time_sec"].transform("median")
    cutoffs = medians * threshold_factor

    # Representative lap if valid and within cutoff
    df["is_valid_representative_lap"] = valid_mask & (df["lap_time_sec"] <= cutoffs)
    return df


def apply_fuel_correction(
    df_laps: pd.DataFrame,
    total_race_laps: Optional[int] = None,
    correction_per_lap: float = ESTIMATED_FUEL_CORRECTION_SEC_PER_LAP,
) -> pd.DataFrame:
    """Apply physics-based fuel correction to lap times.

    As fuel burns off during a Grand Prix, the car's weight decreases by ~100 kg,
    making lap times naturally faster (~0.035s / lap).
    To reveal true tyre degradation without confounding weight reduction, we add back
    the time gained by lighter weight:
        fuel_corrected_lap_time = raw_lap_time + (lap_number - 1) * correction_per_lap

    Args:
        df_laps: DataFrame with ['lap_number', 'lap_time_sec'].
        total_race_laps: Total laps in the Grand Prix (optional reference).
        correction_per_lap: Seconds gained per lap due to fuel burn (default from config).

    Returns:
        DataFrame with new column 'fuel_corrected_lap_sec'.
    """
    df = df_laps.copy()
    if df.empty or "lap_time_sec" not in df.columns or "lap_number" not in df.columns:
        df["fuel_corrected_lap_sec"] = np.nan
        return df

    # Vectorized fuel correction calculation
    df["fuel_corrected_lap_sec"] = df["lap_time_sec"] + (df["lap_number"] - 1) * correction_per_lap
    return df


def merge_laps_with_stints(
    df_laps: pd.DataFrame,
    df_stints: pd.DataFrame,
    driver_mapping: Optional[dict] = None,
) -> pd.DataFrame:
    """Merge lap timing data with tyre compound stint info.

    Calculates the exact 'tyre_age' for every individual lap.

    Args:
        df_laps: Lap times DataFrame.
        df_stints: Tyre stints DataFrame.
        driver_mapping: Optional mapping between driver_id string and driver_number.

    Returns:
        Consolidated DataFrame with ['compound', 'stint_number', 'tyre_age'].
    """
    df = df_laps.copy()
    if df_stints.empty or df.empty:
        df["compound"] = "UNKNOWN"
        df["stint_number"] = 1
        df["tyre_age"] = df["lap_number"]
        return df

    stints = df_stints.copy()
    if driver_mapping and "driver_id" in df.columns:
        # Map driver_id to driver_number if needed
        df["driver_number"] = df["driver_id"].map(driver_mapping)

    merged_rows = []
    for _, lap in df.iterrows():
        d_num = lap.get("driver_number")
        lap_n = lap.get("lap_number")

        # Find matching stint
        if d_num is not None:
            match = stints[
                (stints["driver_number"] == d_num)
                & (stints["lap_start"] <= lap_n)
                & (stints["lap_end"] >= lap_n)
            ]
        else:
            match = pd.DataFrame()

        if not match.empty:
            row_stint = match.iloc[0]
            compound = row_stint["compound"]
            stint_num = row_stint["stint_number"]
            tyre_age = (lap_n - row_stint["lap_start"]) + row_stint.get("tyre_age_at_start", 0) + 1
        else:
            compound = "UNKNOWN"
            stint_num = 1
            tyre_age = lap_n

        lap_dict = lap.to_dict()
        lap_dict["compound"] = compound
        lap_dict["stint_number"] = stint_num
        lap_dict["tyre_age"] = tyre_age
        merged_rows.append(lap_dict)

    return pd.DataFrame(merged_rows)
