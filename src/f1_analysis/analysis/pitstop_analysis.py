"""Module C - pit-stop strategy (observed data only, no simulation)."""

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from f1_analysis.config import PIT_OUTLIER_FACTOR
from f1_analysis.utils.validation import require_columns

logger = logging.getLogger(__name__)


def flag_anomalous_stops(df_pitstops: pd.DataFrame, factor: float = PIT_OUTLIER_FACTOR) -> pd.DataFrame:
    """Add ``is_anomalous`` = pit-lane time > factor x race median.

    Catches red-flag "stops" (cars wait in the pit lane for many minutes) and
    long repairs, which would otherwise distort averages. Rows are kept.
    """
    require_columns(df_pitstops, ["pit_lane_time_sec"], "df_pitstops")
    df = df_pitstops.copy()
    median = df["pit_lane_time_sec"].median()
    df["is_anomalous"] = df["pit_lane_time_sec"].isna() | (df["pit_lane_time_sec"] > factor * median)
    return df


def compute_pit_stop_summary(
    df_pitstops: pd.DataFrame,
    df_results: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Per-driver pit summary.

    Columns: driver_id, total_stops, anomalous_stops, pit_laps,
    fastest_pit_lane_sec, median_pit_lane_sec, total_pit_lane_sec
    (+ driver_name, constructor_name, position if results are given).
    Timing statistics exclude anomalous stops; the stop count includes them.
    """
    if df_pitstops.empty:
        return pd.DataFrame()
    df = flag_anomalous_stops(df_pitstops)
    normal = df[~df["is_anomalous"]]

    summary = df.groupby("driver_id").agg(
        total_stops=("stop_number", "count"),
        anomalous_stops=("is_anomalous", "sum"),
        pit_laps=("lap", lambda s: sorted(s.dropna().astype(int).tolist())),
    )
    timing = normal.groupby("driver_id")["pit_lane_time_sec"].agg(
        fastest_pit_lane_sec="min", median_pit_lane_sec="median", total_pit_lane_sec="sum"
    )
    summary = summary.join(timing).reset_index()

    if df_results is not None and not df_results.empty:
        summary = summary.merge(
            df_results[["driver_id", "driver_name", "constructor_name", "position"]],
            on="driver_id", how="left",
        )
    return summary.sort_values("median_pit_lane_sec").reset_index(drop=True)


def strategy_vs_result(
    df_pitstops: pd.DataFrame,
    df_results: pd.DataFrame,
    df_laps_with_stints: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """One row per classified driver: number of stops, compound sequence, grid and finish.

    Descriptive only - the number of stops is chosen in response to race
    events, so a correlation with finishing position is not causal.
    """
    require_columns(df_results, ["driver_id", "grid", "position"], "df_results")
    stops = df_pitstops.groupby("driver_id").size().rename("n_stops") if not df_pitstops.empty else pd.Series(dtype=int, name="n_stops")
    out = df_results[["driver_id", "driver_name", "constructor_name", "grid", "position", "status"]].merge(
        stops, left_on="driver_id", right_index=True, how="left"
    )
    out["n_stops"] = out["n_stops"].fillna(0).astype(int)

    if df_laps_with_stints is not None and "compound" in df_laps_with_stints:
        seq = (
            df_laps_with_stints.dropna(subset=["stint_number"])
            .sort_values("stint_number")
            .groupby(["driver_id", "stint_number"])["compound"].first()
            .groupby("driver_id").agg(lambda s: " -> ".join(s.astype(str)))
            .rename("compound_sequence")
        )
        out = out.merge(seq, left_on="driver_id", right_index=True, how="left")
    return out.sort_values("position").reset_index(drop=True)


def analyze_undercut(
    df_laps: pd.DataFrame,
    df_pitstops: pd.DataFrame,
    attacker: str,
    defender: str,
    stop_number: int = 1,
) -> Dict[str, Any]:
    """Observed effect of a pit-stop sequence between two drivers.

    The attacker is the driver who pitted first. Using cumulative race time:
        gap_before = elapsed(attacker) - elapsed(defender) at the lap before the attacker's stop
        gap_after  = same quantity at the lap after the defender's stop
        net_gain   = gap_before - gap_after   (> 0: attacker gained time)

    Returns NaN fields (same keys) when the comparison is not possible.
    """
    result: Dict[str, Any] = {
        "attacker": attacker, "defender": defender, "stop_number": stop_number,
        "attacker_pit_lap": np.nan, "defender_pit_lap": np.nan,
        "gap_before_sec": np.nan, "gap_after_sec": np.nan,
        "net_gain_sec": np.nan, "undercut_worked": None,
    }
    stops = df_pitstops[df_pitstops["stop_number"] == stop_number].set_index("driver_id")["lap"]
    if attacker not in stops or defender not in stops:
        return result
    a_lap, d_lap = int(stops[attacker]), int(stops[defender])
    result.update(attacker_pit_lap=a_lap, defender_pit_lap=d_lap)
    if a_lap >= d_lap:
        return result  # attacker did not pit first: not an undercut

    laps = df_laps[df_laps["driver_id"].isin([attacker, defender])].sort_values("lap_number")
    elapsed = laps.pivot(index="lap_number", columns="driver_id", values="lap_time_sec").cumsum(skipna=False)
    before, after = a_lap - 1, d_lap + 1
    if before not in elapsed.index or after not in elapsed.index:
        return result

    gap_before = elapsed.loc[before, attacker] - elapsed.loc[before, defender]
    gap_after = elapsed.loc[after, attacker] - elapsed.loc[after, defender]
    if pd.isna(gap_before) or pd.isna(gap_after):
        return result
    result.update(
        gap_before_sec=float(gap_before),
        gap_after_sec=float(gap_after),
        net_gain_sec=float(gap_before - gap_after),
        undercut_worked=bool(gap_before > 0 and gap_after < 0),
    )
    return result
