"""Module B - tyre degradation.

Degradation is estimated *within stints* so that differences between cars and
drivers do not leak into the slope (a pooled regression would mix them).
"""

import logging

import numpy as np
import pandas as pd
from scipy import stats

from f1_analysis.utils.validation import pace_column, representative_laps, require_columns

logger = logging.getLogger(__name__)


def _prepare(df: pd.DataFrame, use_fuel_corrected: bool) -> tuple[pd.DataFrame, str]:
    """Representative laps with known compound, stint and tyre age."""
    require_columns(df, ["driver_id", "compound", "stint_number", "tyre_age", "lap_time_sec"], "df_laps")
    col = pace_column(df, use_fuel_corrected)
    laps = representative_laps(df)
    laps = laps[laps[col].notna() & laps["tyre_age"].notna() & laps["compound"].notna()]
    laps = laps[laps["compound"] != "UNKNOWN"]
    return laps, col


def compute_compound_degradation(
    df_laps_with_stints: pd.DataFrame,
    min_stint_laps: int = 5,
    use_fuel_corrected: bool = True,
) -> pd.DataFrame:
    """Within-stint (fixed-effects) degradation slope per compound.

    For every stint s (driver + stint number) we subtract the stint means:
        y' = lap_time - mean_s(lap_time),  x' = tyre_age - mean_s(tyre_age)
    then fit y' = beta * x' by OLS (scipy.stats.linregress) on all stints of a
    compound. beta is the average seconds lost per extra lap of tyre age,
    comparing each car only with itself.

    Caveats: p-values assume independent laps (they are autocorrelated) and the
    linregress degrees of freedom ignore the removed stint means, so p-values
    are optimistic. Fuel correction is an assumption; without it slopes are
    biased downwards (cars get lighter as tyres age). This is association, not
    a controlled experiment: traffic and track evolution remain confounders.
    """
    laps, col = _prepare(df_laps_with_stints, use_fuel_corrected)
    if laps.empty:
        return pd.DataFrame()

    keys = ["driver_id", "stint_number"]
    sizes = laps.groupby(keys)[col].transform("size")
    laps = laps[sizes >= min_stint_laps].copy()
    if laps.empty:
        return pd.DataFrame()

    laps["y_dm"] = laps[col] - laps.groupby(keys)[col].transform("mean")
    laps["x_dm"] = laps["tyre_age"] - laps.groupby(keys)["tyre_age"].transform("mean")

    records = []
    for compound, g in laps.groupby("compound"):
        if g["x_dm"].nunique() < 2:
            continue
        reg = stats.linregress(g["x_dm"].to_numpy(float), g["y_dm"].to_numpy(float))
        records.append({
            "compound": compound,
            "n_stints": g.groupby(keys).ngroups,
            "n_laps": len(g),
            "median_lap_sec": float(g[col].median()),
            "max_tyre_age": float(g["tyre_age"].max()),
            "degradation_sec_per_lap": float(reg.slope),
            "slope_std_err": float(reg.stderr),
            "slope_ci95_low": float(reg.slope - 1.96 * reg.stderr),
            "slope_ci95_high": float(reg.slope + 1.96 * reg.stderr),
            "r_squared_within": float(reg.rvalue ** 2),
            "p_value": float(reg.pvalue),
        })
    out = pd.DataFrame(records)
    return out.sort_values("degradation_sec_per_lap").reset_index(drop=True) if not out.empty else out


def compute_driver_stint_summary(
    df_laps_with_stints: pd.DataFrame,
    use_fuel_corrected: bool = True,
    min_laps_for_slope: int = 4,
) -> pd.DataFrame:
    """One row per driver stint with length, median pace and its own OLS slope.

    ``stint_length_laps`` counts all laps of the stint; slope and median use
    representative laps only.
    """
    require_columns(df_laps_with_stints, ["driver_id", "stint_number", "lap_number"], "df_laps")
    all_laps = df_laps_with_stints.dropna(subset=["stint_number"])
    laps, col = _prepare(df_laps_with_stints, use_fuel_corrected)

    records = []
    for (driver_id, stint), g_all in all_laps.groupby(["driver_id", "stint_number"]):
        g = laps[(laps["driver_id"] == driver_id) & (laps["stint_number"] == stint)]
        slope, r2 = np.nan, np.nan
        if len(g) >= min_laps_for_slope and g["tyre_age"].nunique() > 1:
            reg = stats.linregress(g["tyre_age"].to_numpy(float), g[col].to_numpy(float))
            slope, r2 = float(reg.slope), float(reg.rvalue ** 2)
        records.append({
            "driver_id": driver_id,
            "stint_number": int(stint),
            "compound": g_all["compound"].iloc[0],
            "start_lap": int(g_all["lap_number"].min()),
            "end_lap": int(g_all["lap_number"].max()),
            "stint_length_laps": len(g_all),
            "representative_laps": len(g),
            "median_pace_sec": float(g[col].median()) if len(g) else np.nan,
            "degradation_sec_per_lap": slope,
            "r_squared": r2,
        })
    out = pd.DataFrame(records)
    return out.sort_values(["driver_id", "stint_number"]).reset_index(drop=True) if not out.empty else out
