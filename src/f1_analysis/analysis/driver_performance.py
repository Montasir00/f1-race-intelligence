"""Driver performance analytics module for F1 Race Intelligence.

Provides statistical summaries of driver pace, consistency metrics (standard deviation,
interquartile range), teammate comparisons, and fastest lap analysis using Pandas
and NumPy (with optional SciPy if environment policies permit C-extensions).
"""

import logging
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Check if scipy is available without DLL blocks
_HAS_SCIPY = False
try:
    from scipy import stats as sp_stats
    _HAS_SCIPY = True
except (ImportError, Exception):
    _HAS_SCIPY = False


def compute_driver_pace_metrics(
    df_laps: pd.DataFrame,
    use_fuel_corrected: bool = False,
    only_representative: bool = True,
) -> pd.DataFrame:
    """Compute comprehensive statistical pace metrics for each driver in a race.

    Metrics computed:
    - total_laps: Count of all recorded laps.
    - representative_laps: Count of clean representative laps evaluated.
    - mean_lap_sec: Arithmetic mean of lap times.
    - median_lap_sec: 50th percentile (robust to outliers).
    - std_lap_sec: Sample standard deviation (lower = more consistent).
    - iqr_lap_sec: Interquartile Range (Q75 - Q25), robust non-parametric consistency measure.
    - fastest_lap_sec: Absolute fastest lap recorded.

    Args:
        df_laps: DataFrame containing lap records with ['driver_id', 'lap_time_sec'].
        use_fuel_corrected: If True, uses 'fuel_corrected_lap_sec' instead of raw lap times.
        only_representative: If True, filters using 'is_valid_representative_lap'.

    Returns:
        DataFrame sorted by median lap time ascending with gap to fastest driver.
    """
    df = df_laps.copy()
    col = (
        "fuel_corrected_lap_sec"
        if use_fuel_corrected and "fuel_corrected_lap_sec" in df.columns
        else "lap_time_sec"
    )

    if only_representative and "is_valid_representative_lap" in df.columns:
        valid_df = df[df["is_valid_representative_lap"]].copy()
    else:
        valid_df = df[df[col].notna() & (df[col] > 0)].copy()

    if valid_df.empty:
        return pd.DataFrame()

    records = []
    for driver_id, group in valid_df.groupby("driver_id"):
        times = group[col].dropna().values
        if len(times) == 0:
            continue

        q75, q25 = np.percentile(times, [75, 25])
        iqr = float(q75 - q25)

        total_driver_laps = len(df[df["driver_id"] == driver_id])
        records.append({
            "driver_id": driver_id,
            "total_laps": total_driver_laps,
            "representative_laps": len(times),
            "mean_lap_sec": float(np.mean(times)),
            "median_lap_sec": float(np.median(times)),
            "std_lap_sec": float(np.std(times, ddof=1)) if len(times) > 1 else 0.0,
            "iqr_lap_sec": iqr,
            "fastest_lap_sec": float(np.min(times)),
        })

    summary_df = pd.DataFrame(records)
    if not summary_df.empty:
        summary_df.sort_values(by="median_lap_sec", inplace=True)
        summary_df.reset_index(drop=True, inplace=True)
        fastest_median = summary_df["median_lap_sec"].iloc[0]
        summary_df["gap_to_fastest_sec"] = summary_df["median_lap_sec"] - fastest_median

    return summary_df


def compare_teammates(
    df_laps: pd.DataFrame,
    df_results: pd.DataFrame,
    use_fuel_corrected: bool = False,
) -> pd.DataFrame:
    """Perform statistical pace comparison between teammates in the same constructor.

    Compares median pace, standard deviation, and performs two-sample significance
    testing (Welch's t-test and Mann-Whitney U test) when available.

    Args:
        df_laps: Cleaned lap times DataFrame.
        df_results: Race results containing ['driver_id', 'constructor_id', 'constructor_name'].
        use_fuel_corrected: Whether to base comparison on fuel-corrected times.

    Returns:
        DataFrame detailing teammate pairs, pace deltas, and statistical significance p-values.
    """
    if df_results.empty or df_laps.empty:
        return pd.DataFrame()

    col = (
        "fuel_corrected_lap_sec"
        if use_fuel_corrected and "fuel_corrected_lap_sec" in df_laps.columns
        else "lap_time_sec"
    )
    valid_laps = df_laps[df_laps.get("is_valid_representative_lap", True)].copy()

    records = []
    for constructor_id, team_group in df_results.groupby("constructor_id"):
        drivers = team_group["driver_id"].unique()
        if len(drivers) < 2:
            continue

        d1, d2 = drivers[0], drivers[1]
        team_name = team_group["constructor_name"].iloc[0]

        d1_times = valid_laps[valid_laps["driver_id"] == d1][col].dropna().values
        d2_times = valid_laps[valid_laps["driver_id"] == d2][col].dropna().values

        if len(d1_times) < 3 or len(d2_times) < 3:
            continue

        d1_median = float(np.median(d1_times))
        d2_median = float(np.median(d2_times))
        delta_median = d1_median - d2_median

        p_val_ttest = np.nan
        p_val_mw = np.nan
        is_sig = False

        if _HAS_SCIPY:
            try:
                _, p_val_ttest = sp_stats.ttest_ind(d1_times, d2_times, equal_var=False)
                _, p_val_mw = sp_stats.mannwhitneyu(d1_times, d2_times, alternative="two-sided")
                is_sig = bool(p_val_mw < 0.05)
            except Exception:
                pass

        records.append({
            "constructor_id": constructor_id,
            "constructor_name": team_name,
            "driver_1": d1,
            "driver_2": d2,
            "driver_1_median_sec": d1_median,
            "driver_2_median_sec": d2_median,
            "delta_sec": delta_median,  # Negative means driver 1 is faster
            "faster_driver": d1 if delta_median < 0 else d2,
            "ttest_p_value": float(p_val_ttest) if not np.isnan(p_val_ttest) else None,
            "mannwhitney_p_value": float(p_val_mw) if not np.isnan(p_val_mw) else None,
            "is_statistically_significant": is_sig,
        })

    return pd.DataFrame(records)
