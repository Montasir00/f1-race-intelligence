"""Module A - driver pace and consistency."""

import logging

import numpy as np
import pandas as pd
from scipy import stats

from f1_analysis.utils.validation import pace_column, representative_laps, require_columns

logger = logging.getLogger(__name__)


def compute_driver_pace_metrics(
    df_laps: pd.DataFrame,
    use_fuel_corrected: bool = False,
    only_representative: bool = True,
) -> pd.DataFrame:
    """Per-driver pace statistics.

    Columns: driver_id, total_laps, representative_laps, mean_lap_sec,
    median_lap_sec, std_lap_sec (sample, ddof=1), iqr_lap_sec, fastest_lap_sec,
    gap_to_fastest_sec (median vs. the best median).

    Median and IQR are preferred for comparisons because lap-time
    distributions are right-skewed (traffic, mistakes).
    """
    require_columns(df_laps, ["driver_id", "lap_time_sec"], "df_laps")
    col = pace_column(df_laps, use_fuel_corrected)
    laps = representative_laps(df_laps) if only_representative else df_laps.copy()
    laps = laps[laps[col].notna() & (laps[col] > 0)]
    if laps.empty:
        return pd.DataFrame()

    grouped = laps.groupby("driver_id")[col]
    summary = pd.DataFrame({
        "representative_laps": grouped.count(),
        "mean_lap_sec": grouped.mean(),
        "median_lap_sec": grouped.median(),
        "std_lap_sec": grouped.std(ddof=1),
        "iqr_lap_sec": grouped.quantile(0.75) - grouped.quantile(0.25),
        "fastest_lap_sec": grouped.min(),
    })
    summary.insert(0, "total_laps", df_laps.groupby("driver_id").size().reindex(summary.index))
    summary = summary.sort_values("median_lap_sec").reset_index()
    summary["gap_to_fastest_sec"] = summary["median_lap_sec"] - summary["median_lap_sec"].iloc[0]
    return summary


def compare_teammates(
    df_laps: pd.DataFrame,
    df_results: pd.DataFrame,
    use_fuel_corrected: bool = False,
    min_laps: int = 5,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Compare the two drivers of each constructor.

    Reports the median delta plus Welch's t-test and the Mann-Whitney U test.

    Statistical caveat (important): both tests assume independent observations.
    Consecutive laps are autocorrelated (tyre wear, fuel, traffic) and teammates
    may run different strategies, so p-values are optimistic. Treat them as a
    descriptive indication, not proof of a pace difference.

    ``delta_sec`` = median(driver_1) - median(driver_2); negative means driver_1 faster.
    """
    require_columns(df_laps, ["driver_id", "lap_time_sec"], "df_laps")
    if df_results.empty or df_laps.empty:
        return pd.DataFrame()
    col = pace_column(df_laps, use_fuel_corrected)
    laps = representative_laps(df_laps)

    records = []
    for constructor_id, team in df_results.groupby("constructor_id"):
        drivers = team.sort_values("position")["driver_id"].tolist()
        if len(drivers) < 2:
            continue
        d1, d2 = drivers[:2]
        t1 = laps.loc[laps["driver_id"] == d1, col].dropna().to_numpy()
        t2 = laps.loc[laps["driver_id"] == d2, col].dropna().to_numpy()
        if len(t1) < min_laps or len(t2) < min_laps:
            continue

        _, p_t = stats.ttest_ind(t1, t2, equal_var=False)
        _, p_mw = stats.mannwhitneyu(t1, t2, alternative="two-sided")
        delta = float(np.median(t1) - np.median(t2))
        records.append({
            "constructor_id": constructor_id,
            "constructor_name": team["constructor_name"].iloc[0],
            "driver_1": d1,
            "driver_2": d2,
            "n_laps_1": len(t1),
            "n_laps_2": len(t2),
            "driver_1_median_sec": float(np.median(t1)),
            "driver_2_median_sec": float(np.median(t2)),
            "delta_sec": delta,
            "faster_driver": d1 if delta < 0 else d2,
            "welch_t_p_value": float(p_t),
            "mannwhitney_p_value": float(p_mw),
            "significant_at_alpha": bool(p_mw < alpha),
        })
    return pd.DataFrame(records)
