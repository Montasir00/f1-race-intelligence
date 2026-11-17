"""Constructor (team) performance analytics module for F1 Race Intelligence.

Aggregates driver performances into constructor-level metrics: overall team race pace,
points haul, double-points finishes, and intra-team consistency.
"""

import logging
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_constructor_pace_ranking(
    df_laps: pd.DataFrame,
    df_results: pd.DataFrame,
    use_fuel_corrected: bool = True,
) -> pd.DataFrame:
    """Aggregate lap times across both drivers to rank constructors by race pace.

    Args:
        df_laps: Cleaned lap times DataFrame.
        df_results: Race results containing ['driver_id', 'constructor_id', 'constructor_name'].
        use_fuel_corrected: Whether to base rankings on fuel-corrected times.

    Returns:
        DataFrame ranking constructors by median pace, standard deviation, and gap to fastest team.
    """
    if df_laps.empty or df_results.empty:
        return pd.DataFrame()

    col = (
        "fuel_corrected_lap_sec"
        if use_fuel_corrected and "fuel_corrected_lap_sec" in df_laps.columns
        else "lap_time_sec"
    )

    valid_laps = df_laps[df_laps.get("is_valid_representative_lap", True)].copy()

    # Merge constructor info onto lap times
    merged = valid_laps.merge(
        df_results[["driver_id", "constructor_id", "constructor_name"]],
        on="driver_id",
        how="inner",
    )

    records = []
    for constructor_id, group in merged.groupby("constructor_id"):
        times = group[col].dropna().values
        if len(times) == 0:
            continue

        team_name = group["constructor_name"].iloc[0]
        q75, q25 = np.percentile(times, [75, 25])

        records.append({
            "constructor_id": constructor_id,
            "constructor_name": team_name,
            "representative_laps": len(times),
            "median_lap_sec": float(np.median(times)),
            "mean_lap_sec": float(np.mean(times)),
            "std_lap_sec": float(np.std(times, ddof=1)) if len(times) > 1 else 0.0,
            "iqr_lap_sec": float(q75 - q25),
        })

    summary_df = pd.DataFrame(records)
    if not summary_df.empty:
        summary_df.sort_values(by="median_lap_sec", inplace=True)
        summary_df.reset_index(drop=True, inplace=True)
        fastest_median = summary_df["median_lap_sec"].iloc[0]
        summary_df["gap_to_fastest_sec"] = summary_df["median_lap_sec"] - fastest_median

    return summary_df


def compute_constructor_points_summary(df_results: pd.DataFrame) -> pd.DataFrame:
    """Summarize constructor points scoring and finishing positions for a Grand Prix.

    Args:
        df_results: Race results containing ['constructor_id', 'constructor_name', 'points', 'position'].

    Returns:
        DataFrame with total points scored, best finish, and double-points finish flag.
    """
    if df_results.empty or "constructor_id" not in df_results.columns:
        return pd.DataFrame()

    records = []
    for constructor_id, group in df_results.groupby("constructor_id"):
        team_name = group["constructor_name"].iloc[0]
        total_points = float(group["points"].sum())
        best_pos = int(group["position"].min()) if group["position"].notna().any() else None

        # Both drivers finished in top 10 (points positions)
        points_positions = group[group["position"] <= 10]
        both_in_points = len(points_positions) == 2

        records.append({
            "constructor_id": constructor_id,
            "constructor_name": team_name,
            "total_points": total_points,
            "best_finish": best_pos,
            "drivers_in_points": len(points_positions),
            "double_points_finish": both_in_points,
        })

    summary_df = pd.DataFrame(records)
    if not summary_df.empty:
        summary_df.sort_values(by=["total_points", "best_finish"], ascending=[False, True], inplace=True)
        summary_df.reset_index(drop=True, inplace=True)

    return summary_df
