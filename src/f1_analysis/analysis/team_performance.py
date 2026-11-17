"""Module E - constructor performance."""

import logging
from typing import Dict, List

import numpy as np
import pandas as pd

from f1_analysis.utils.validation import pace_column, representative_laps, require_columns

logger = logging.getLogger(__name__)


def compute_constructor_pace_ranking(
    df_laps: pd.DataFrame,
    df_results: pd.DataFrame,
    use_fuel_corrected: bool = True,
) -> pd.DataFrame:
    """Rank teams by the median of all representative laps of both drivers."""
    require_columns(df_laps, ["driver_id", "lap_time_sec"], "df_laps")
    if df_laps.empty or df_results.empty:
        return pd.DataFrame()
    col = pace_column(df_laps, use_fuel_corrected)
    laps = representative_laps(df_laps).merge(
        df_results[["driver_id", "constructor_id", "constructor_name"]], on="driver_id", how="inner"
    )
    laps = laps.dropna(subset=[col])
    if laps.empty:
        return pd.DataFrame()

    g = laps.groupby(["constructor_id", "constructor_name"])[col]
    out = pd.DataFrame({
        "representative_laps": g.count(),
        "median_lap_sec": g.median(),
        "mean_lap_sec": g.mean(),
        "std_lap_sec": g.std(ddof=1),
        "iqr_lap_sec": g.quantile(0.75) - g.quantile(0.25),
    }).reset_index().sort_values("median_lap_sec").reset_index(drop=True)
    out["gap_to_fastest_sec"] = out["median_lap_sec"] - out["median_lap_sec"].iloc[0]
    return out


def compute_constructor_points_summary(df_results: pd.DataFrame) -> pd.DataFrame:
    """Points, best finish and drivers in the top 10 per team for one race."""
    require_columns(df_results, ["constructor_id", "constructor_name", "points", "position"], "df_results")
    if df_results.empty:
        return pd.DataFrame()
    g = df_results.groupby(["constructor_id", "constructor_name"])
    out = pd.DataFrame({
        "total_points": g["points"].sum(),
        "best_finish": g["position"].min(),
        "drivers_in_top10": g["position"].apply(lambda s: int((s <= 10).sum())),
    }).reset_index()
    out["double_top10_finish"] = out["drivers_in_top10"] == 2
    return out.sort_values(["total_points", "best_finish"], ascending=[False, True]).reset_index(drop=True)


def compute_season_constructor_points(loader, year: int, rounds: List[int]) -> pd.DataFrame:
    """Cumulative constructor points over several rounds (season trend).

    Args:
        loader: An ``F1DataLoader``.
        year: Season.
        rounds: Round numbers to include (each requires one cached/API request).

    Returns:
        Long DataFrame: round, constructor_name, points, cumulative_points.
        Sprint points are not included (race results only).
    """
    frames = []
    for rnd in rounds:
        res = loader.load_race_results(year, rnd)
        if res.empty:
            logger.warning("No results for %s round %s - skipped", year, rnd)
            continue
        pts = res.groupby("constructor_name")["points"].sum().reset_index()
        pts["round"] = rnd
        frames.append(pts)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True).sort_values(["constructor_name", "round"])
    df["cumulative_points"] = df.groupby("constructor_name")["points"].cumsum()
    return df.reset_index(drop=True)
