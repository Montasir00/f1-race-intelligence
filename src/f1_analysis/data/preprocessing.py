"""Cleaning and feature engineering for lap-level data.

Pipeline order used by the notebook and dashboard:
    flag_pit_laps -> clean_lap_times -> apply_fuel_correction -> merge_laps_with_stints
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from f1_analysis.config import (
    FUEL_EFFECT_SEC_PER_KG,
    SLOW_LAP_THRESHOLD_FACTOR,
    START_FUEL_KG,
)
from f1_analysis.utils.validation import require_columns

logger = logging.getLogger(__name__)


def flag_pit_laps(df_laps: pd.DataFrame, df_pitstops: pd.DataFrame) -> pd.DataFrame:
    """Add boolean columns ``is_pit_in_lap`` and ``is_pit_out_lap``.

    Jolpica records a stop on the lap the driver ENTERS the pits (in-lap);
    the following lap starts in the pit lane (out-lap). Both are slow for
    reasons unrelated to pace, so they must be excluded from pace analysis.
    """
    require_columns(df_laps, ["driver_id", "lap_number"], "df_laps")
    df = df_laps.copy()
    if df_pitstops is None or df_pitstops.empty:
        df["is_pit_in_lap"] = False
        df["is_pit_out_lap"] = False
        return df

    stops = df_pitstops.dropna(subset=["driver_id", "lap"])
    in_keys = set(zip(stops["driver_id"], stops["lap"].astype(int)))
    out_keys = {(d, lap + 1) for d, lap in in_keys}
    keys = list(zip(df["driver_id"], df["lap_number"]))
    df["is_pit_in_lap"] = [k in in_keys for k in keys]
    df["is_pit_out_lap"] = [k in out_keys for k in keys]
    return df


def clean_lap_times(
    df_laps: pd.DataFrame,
    threshold_factor: float = SLOW_LAP_THRESHOLD_FACTOR,
    exclude_first_lap: bool = True,
) -> pd.DataFrame:
    """Add ``is_valid_representative_lap`` and ``exclusion_reason`` columns.

    A lap is excluded (not deleted - every row is kept) when it is:
      * missing / non-positive,
      * lap 1 (standing start, not comparable),
      * a pit in-lap or out-lap (needs ``flag_pit_laps`` first),
      * slower than ``threshold_factor`` x the driver's median lap.

    The last rule is a heuristic proxy for Safety Car / VSC / yellow-flag laps
    and incidents: the APIs used here do not give lap-level track status, so
    some neutralised laps may remain and some genuine slow laps may be removed.
    """
    require_columns(df_laps, ["driver_id", "lap_number", "lap_time_sec"], "df_laps")
    df = df_laps.copy()
    reason = pd.Series(pd.NA, index=df.index, dtype="object")

    missing = df["lap_time_sec"].isna() | (df["lap_time_sec"] <= 0)
    reason[missing] = "missing_time"

    if exclude_first_lap:
        reason[reason.isna() & (df["lap_number"] == 1)] = "first_lap"

    for col, label in (("is_pit_in_lap", "pit_in_lap"), ("is_pit_out_lap", "pit_out_lap")):
        if col in df.columns:
            reason[reason.isna() & df[col].astype(bool)] = label

    # Median of the laps still considered valid, per driver
    candidate = reason.isna()
    medians = df.loc[candidate].groupby("driver_id")["lap_time_sec"].median()
    cutoff = df["driver_id"].map(medians) * threshold_factor
    reason[candidate & (df["lap_time_sec"] > cutoff)] = "slow_lap"

    df["exclusion_reason"] = reason
    df["is_valid_representative_lap"] = reason.isna()
    return df


def apply_fuel_correction(
    df_laps: pd.DataFrame,
    total_race_laps: Optional[int] = None,
    fuel_effect_sec_per_kg: float = FUEL_EFFECT_SEC_PER_KG,
    start_fuel_kg: float = START_FUEL_KG,
) -> pd.DataFrame:
    """Add ``fuel_corrected_lap_sec``: lap time normalised to the lap-1 fuel load.

    Model (an assumption, see config.py):
        burn_per_lap   = start_fuel_kg / total_race_laps        [kg/lap]
        correction     = fuel_effect_sec_per_kg * burn_per_lap  [s/lap]
        corrected_time = raw_time + (lap_number - 1) * correction

    ``total_race_laps`` defaults to the maximum lap number in the data.
    The correction is linear and identical for all cars, so it does not change
    comparisons between drivers on the same lap - it matters for trends over a race.
    """
    require_columns(df_laps, ["lap_number", "lap_time_sec"], "df_laps")
    df = df_laps.copy()
    if df.empty:
        df["fuel_corrected_lap_sec"] = np.nan
        return df
    if total_race_laps is None:
        total_race_laps = int(df["lap_number"].max())
    if total_race_laps <= 0:
        raise ValueError("total_race_laps must be positive")

    correction_per_lap = fuel_effect_sec_per_kg * start_fuel_kg / total_race_laps
    df["fuel_corrected_lap_sec"] = df["lap_time_sec"] + (df["lap_number"] - 1) * correction_per_lap
    df.attrs["fuel_correction_sec_per_lap"] = correction_per_lap
    return df


def merge_laps_with_stints(
    df_laps: pd.DataFrame,
    df_stints: pd.DataFrame,
    df_results: pd.DataFrame,
) -> pd.DataFrame:
    """Attach ``compound``, ``stint_number`` and ``tyre_age`` to each lap.

    Jolpica identifies drivers by ``driver_id`` and OpenF1 by car number, so
    ``df_results`` (which has both) is used as the bridge.

    ``tyre_age`` = laps completed on that tyre set at the end of the lap
    = tyre_age_at_start + (lap_number - lap_start) + 1.

    Laps that match no stint keep NaN values - they are NOT filled with guesses.
    """
    require_columns(df_laps, ["driver_id", "lap_number"], "df_laps")
    df = df_laps.copy()
    df["_row"] = np.arange(len(df))

    if df_stints is None or df_stints.empty or df_results is None or df_results.empty:
        logger.warning("No stint data: compound and tyre_age left as NaN")
        df["compound"], df["stint_number"], df["tyre_age"] = np.nan, np.nan, np.nan
        return df.drop(columns="_row")

    mapping = df_results.dropna(subset=["driver_number"]).set_index("driver_id")["driver_number"]
    df["driver_number"] = df["driver_id"].map(mapping)

    stints = df_stints.copy()
    # The final stint may have no lap_end yet; treat it as running to the end
    stints["lap_end"] = stints["lap_end"].fillna(df["lap_number"].max())
    stints["tyre_age_at_start"] = stints["tyre_age_at_start"].fillna(0)

    joined = df[["_row", "driver_number", "lap_number"]].merge(stints, on="driver_number", how="inner")
    joined = joined[(joined["lap_number"] >= joined["lap_start"]) & (joined["lap_number"] <= joined["lap_end"])]
    # If two stints overlap on a boundary lap, keep the later stint
    joined = joined.sort_values("stint_number").drop_duplicates("_row", keep="last")
    joined["tyre_age"] = joined["tyre_age_at_start"] + (joined["lap_number"] - joined["lap_start"]) + 1

    df = df.merge(joined[["_row", "compound", "stint_number", "tyre_age"]], on="_row", how="left")
    n_unmatched = int(df["compound"].isna().sum())
    if n_unmatched:
        logger.info("%d laps did not match any stint", n_unmatched)
    return df.drop(columns="_row")
