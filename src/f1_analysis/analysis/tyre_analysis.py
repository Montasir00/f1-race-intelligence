"""Tyre degradation and compound analysis module for F1 Race Intelligence.

Applies linear regression (OLS via NumPy / SciPy) to quantify lap-time degradation rates
per lap of tyre age across compounds (Soft, Medium, Hard), accounting for fuel load burn-off.
"""

import logging
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Check if scipy is available
_HAS_SCIPY = False
try:
    from scipy import stats as sp_stats
    _HAS_SCIPY = True
except (ImportError, Exception):
    _HAS_SCIPY = False


def _fit_linear_regression(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """Fit OLS linear regression y = intercept + slope * x using NumPy or SciPy."""
    if _HAS_SCIPY:
        try:
            reg = sp_stats.linregress(x, y)
            return {
                "intercept": float(reg.intercept),
                "slope": float(reg.slope),
                "r_squared": float(reg.rvalue ** 2),
                "p_value": float(reg.pvalue),
                "std_err": float(reg.stderr),
            }
        except Exception:
            pass

    # Pure NumPy analytical OLS fallback: slope = Cov(x,y) / Var(x)
    n = len(x)
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    ss_xy = np.sum((x - x_mean) * (y - y_mean))
    ss_xx = np.sum((x - x_mean) ** 2)

    if ss_xx == 0:
        return {"intercept": float(y_mean), "slope": 0.0, "r_squared": 0.0, "p_value": 1.0, "std_err": 0.0}

    slope = float(ss_xy / ss_xx)
    intercept = float(y_mean - slope * x_mean)

    y_pred = intercept + slope * x
    ss_tot = np.sum((y - y_mean) ** 2)
    ss_res = np.sum((y - y_pred) ** 2)
    r_squared = float(1 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0

    return {
        "intercept": intercept,
        "slope": slope,
        "r_squared": max(0.0, r_squared),
        "p_value": 0.0,
        "std_err": 0.0,
    }


def compute_compound_degradation(
    df_laps_with_stints: pd.DataFrame,
    min_stint_laps: int = 5,
    use_fuel_corrected: bool = True,
) -> pd.DataFrame:
    """Calculate the tyre degradation slope (seconds lost per lap of tyre age) by compound.

    Fits Ordinary Least Squares (OLS) regression:
        lap_time = alpha + beta * tyre_age

    where beta is the degradation coefficient in seconds/lap.

    Args:
        df_laps_with_stints: DataFrame containing ['compound', 'tyre_age', 'lap_time_sec']
                             and optionally ['is_valid_representative_lap', 'fuel_corrected_lap_sec'].
        min_stint_laps: Minimum data points required to fit a regression model.
        use_fuel_corrected: Whether to model fuel-corrected lap times.

    Returns:
        DataFrame summarizing compound metrics, slope (beta), and r_squared.
    """
    df = df_laps_with_stints.copy()
    col = (
        "fuel_corrected_lap_sec"
        if use_fuel_corrected and "fuel_corrected_lap_sec" in df.columns
        else "lap_time_sec"
    )

    if "is_valid_representative_lap" in df.columns:
        df = df[df["is_valid_representative_lap"]].copy()

    df = df[df[col].notna() & df["tyre_age"].notna() & (df["tyre_age"] > 0)]
    if df.empty or "compound" not in df.columns:
        return pd.DataFrame()

    results = []
    for compound, group in df.groupby("compound"):
        if compound == "UNKNOWN" or len(group) < min_stint_laps:
            continue

        x = group["tyre_age"].values.astype(float)
        y = group[col].values.astype(float)

        fit = _fit_linear_regression(x, y)

        results.append({
            "compound": compound,
            "sample_size_laps": len(x),
            "mean_tyre_age": float(np.mean(x)),
            "max_tyre_age": float(np.max(x)),
            "base_pace_intercept_sec": fit["intercept"],
            "degradation_slope_sec_per_lap": fit["slope"],
            "r_squared": fit["r_squared"],
            "p_value": fit["p_value"],
        })

    summary_df = pd.DataFrame(results)
    if not summary_df.empty:
        summary_df.sort_values(by="base_pace_intercept_sec", inplace=True)
        summary_df.reset_index(drop=True, inplace=True)

    return summary_df


def compute_driver_stint_summary(
    df_laps_with_stints: pd.DataFrame,
    use_fuel_corrected: bool = True,
) -> pd.DataFrame:
    """Summarize individual driver stints, duration, and observed degradation rates.

    Args:
        df_laps_with_stints: DataFrame containing lap and stint metadata.
        use_fuel_corrected: Whether to measure degradation using fuel-corrected times.

    Returns:
        DataFrame breakdown per driver and stint.
    """
    df = df_laps_with_stints.copy()
    col = (
        "fuel_corrected_lap_sec"
        if use_fuel_corrected and "fuel_corrected_lap_sec" in df.columns
        else "lap_time_sec"
    )

    if "is_valid_representative_lap" in df.columns:
        valid_df = df[df["is_valid_representative_lap"]].copy()
    else:
        valid_df = df.copy()

    records = []
    for (driver_id, stint_num), group in valid_df.groupby(["driver_id", "stint_number"]):
        times = group[col].dropna().values
        tyre_ages = group["tyre_age"].dropna().values

        if len(times) == 0:
            continue

        compound = group["compound"].iloc[0] if "compound" in group.columns else "UNKNOWN"
        start_lap = int(group["lap_number"].min())
        end_lap = int(group["lap_number"].max())
        stint_length = len(group)

        if len(times) >= 4 and len(np.unique(tyre_ages)) > 1:
            fit = _fit_linear_regression(tyre_ages.astype(float), times.astype(float))
            deg_slope = fit["slope"]
            r_sq = fit["r_squared"]
        else:
            deg_slope = np.nan
            r_sq = np.nan

        records.append({
            "driver_id": driver_id,
            "stint_number": stint_num,
            "compound": compound,
            "start_lap": start_lap,
            "end_lap": end_lap,
            "stint_length_laps": stint_length,
            "median_pace_sec": float(np.median(times)),
            "degradation_sec_per_lap": deg_slope,
            "r_squared": r_sq,
        })

    stint_df = pd.DataFrame(records)
    if not stint_df.empty:
        stint_df.sort_values(by=["driver_id", "stint_number"], inplace=True)
        stint_df.reset_index(drop=True, inplace=True)

    return stint_df
