"""Charts built with Matplotlib/Seaborn (static) and Plotly (interactive).

Every function returns a Figure and never calls ``plt.show()``, so the same
code works in Jupyter (figures display inline) and in Streamlit (``st.pyplot``).
The Matplotlib backend is NOT forced here: Jupyter/Streamlit choose it.
"""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns

from f1_analysis.analysis.race_dynamics import compute_gap_to_leader
from f1_analysis.config import COMPOUND_COLORS, CONSTRUCTOR_COLORS
from f1_analysis.utils.validation import pace_column, representative_laps

sns.set_theme(style="whitegrid")


def team_color(constructor_name: str) -> str:
    """Return the team colour for a Jolpica constructor name (grey if unknown)."""
    name = str(constructor_name).lower()
    for key, color in CONSTRUCTOR_COLORS.items():
        if key in name:
            return color
    return "#888888"


def _empty_figure(message: str, figsize=(8, 3)) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)
    ax.text(0.5, 0.5, message, ha="center", va="center")
    ax.set_axis_off()
    return fig


def _pace_label(col: str) -> str:
    return "Fuel-corrected lap time (s)" if col == "fuel_corrected_lap_sec" else "Lap time (s)"


def plot_driver_pace_distributions(
    df_laps: pd.DataFrame,
    top_n: int = 10,
    use_fuel_corrected: bool = False,
    figsize: tuple = (12, 6),
) -> plt.Figure:
    """Box plot + individual laps for the ``top_n`` drivers by median pace."""
    col = pace_column(df_laps, use_fuel_corrected)
    laps = representative_laps(df_laps).dropna(subset=[col])
    if laps.empty:
        return _empty_figure("No representative laps available")

    order = laps.groupby("driver_id")[col].median().nsmallest(top_n).index.tolist()
    data = laps[laps["driver_id"].isin(order)]

    fig, ax = plt.subplots(figsize=figsize)
    sns.boxplot(data=data, x="driver_id", y=col, order=order, hue="driver_id",
                palette="Blues_r", legend=False, showfliers=False, ax=ax)
    sns.stripplot(data=data, x="driver_id", y=col, order=order, color="black",
                  alpha=0.3, size=3, jitter=0.2, ax=ax)
    ax.set_title(f"Lap-time distribution, top {len(order)} drivers by median pace")
    ax.set_xlabel("Driver")
    ax.set_ylabel(_pace_label(col))
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    return fig


def plot_tyre_degradation(
    df_laps_with_stints: pd.DataFrame,
    use_fuel_corrected: bool = True,
    figsize: tuple = (11, 6),
) -> plt.Figure:
    """Lap time vs. tyre age, coloured by compound, with a per-compound linear trend.

    The trend lines are a pooled visual guide; the numeric slopes reported by
    ``compute_compound_degradation`` use the within-stint estimator.
    """
    col = pace_column(df_laps_with_stints, use_fuel_corrected)
    laps = representative_laps(df_laps_with_stints)
    if "compound" not in laps:
        return _empty_figure("No tyre data available")
    laps = laps.dropna(subset=[col, "tyre_age", "compound"])
    laps = laps[laps["compound"] != "UNKNOWN"]
    if laps.empty:
        return _empty_figure("No tyre data available")

    fig, ax = plt.subplots(figsize=figsize)
    for compound, g in laps.groupby("compound"):
        color = COMPOUND_COLORS.get(compound, COMPOUND_COLORS["UNKNOWN"])
        ax.scatter(g["tyre_age"], g[col], color=color, edgecolors="black",
                   linewidths=0.3, alpha=0.6, s=25, label=f"{compound} (n={len(g)})")
        if len(g) >= 5 and g["tyre_age"].nunique() > 1:
            slope, intercept = np.polyfit(g["tyre_age"], g[col], deg=1)
            xs = np.linspace(g["tyre_age"].min(), g["tyre_age"].max(), 50)
            ax.plot(xs, intercept + slope * xs, color="black", linestyle="--", linewidth=1)
            ax.plot(xs, intercept + slope * xs, color=color, linewidth=2.5, alpha=0.8)
    ax.set_title("Lap time vs. tyre age by compound")
    ax.set_xlabel("Tyre age (laps completed on the set)")
    ax.set_ylabel(_pace_label(col))
    ax.legend(title="Compound")
    fig.tight_layout()
    return fig


def plot_position_progression(df_laps: pd.DataFrame, top_n: int = 10, figsize: tuple = (13, 7)) -> plt.Figure:
    """Lap chart for the drivers who finished in the top ``top_n`` positions."""
    if df_laps.empty or "position" not in df_laps:
        return _empty_figure("No position data available")
    last = df_laps.sort_values("lap_number").groupby("driver_id").tail(1)
    drivers = last.sort_values(["lap_number", "position"], ascending=[False, True]).head(top_n)["driver_id"]

    fig, ax = plt.subplots(figsize=figsize)
    palette = sns.color_palette("tab20", n_colors=len(drivers))
    for color, driver in zip(palette, drivers):
        d = df_laps[df_laps["driver_id"] == driver].sort_values("lap_number")
        ax.plot(d["lap_number"], d["position"], label=driver, color=color, linewidth=2)
    max_pos = int(df_laps["position"].max())
    ax.set_ylim(max_pos + 0.5, 0.5)            # P1 at the top
    ax.set_yticks(range(1, max_pos + 1))
    ax.set_title("Position by lap")
    ax.set_xlabel("Lap")
    ax.set_ylabel("Position")
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", title="Driver")
    fig.tight_layout()
    return fig


def plot_pit_stop_times(pit_summary: pd.DataFrame, figsize: tuple = (11, 5)) -> plt.Figure:
    """Horizontal bars of median pit-lane time per driver, coloured by team."""
    if pit_summary.empty or "median_pit_lane_sec" not in pit_summary:
        return _empty_figure("No pit-stop data available")
    data = pit_summary.dropna(subset=["median_pit_lane_sec"]).sort_values("median_pit_lane_sec")
    colors = [team_color(c) for c in data.get("constructor_name", pd.Series([""] * len(data)))]
    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(data["driver_id"], data["median_pit_lane_sec"], color=colors, edgecolor="black")
    ax.invert_yaxis()
    ax.set_xlim(data["median_pit_lane_sec"].min() * 0.95, data["median_pit_lane_sec"].max() * 1.02)
    ax.set_title("Median pit-lane time per driver (entry to exit, anomalous stops excluded)")
    ax.set_xlabel("Pit-lane time (s)")
    fig.tight_layout()
    return fig


def plot_constructor_pace(team_pace: pd.DataFrame, figsize: tuple = (11, 5)) -> plt.Figure:
    """Bar chart of each team's median-lap gap to the fastest team."""
    if team_pace.empty:
        return _empty_figure("No constructor data available")
    data = team_pace.sort_values("gap_to_fastest_sec")
    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(data["constructor_name"], data["gap_to_fastest_sec"],
            color=[team_color(c) for c in data["constructor_name"]], edgecolor="black")
    ax.invert_yaxis()
    ax.set_title("Constructor race pace: gap to fastest team (median representative lap)")
    ax.set_xlabel("Gap (s)")
    fig.tight_layout()
    return fig


def create_interactive_gap_chart(df_laps: pd.DataFrame, top_n: int = 8) -> go.Figure:
    """Interactive Plotly line chart of the gap to the leader."""
    gaps = compute_gap_to_leader(df_laps)
    if gaps.empty:
        return go.Figure()
    last_lap = gaps.groupby("driver_id")["lap_number"].max()
    final = gaps[gaps["lap_number"] == gaps["driver_id"].map(last_lap)]
    drivers = final.sort_values(["lap_number", "position"], ascending=[False, True]).head(top_n)["driver_id"]
    fig = px.line(
        gaps[gaps["driver_id"].isin(drivers)], x="lap_number", y="gap_to_leader_sec", color="driver_id",
        labels={"lap_number": "Lap", "gap_to_leader_sec": "Gap to leader (s)", "driver_id": "Driver"},
        title="Gap to race leader", template="plotly_white",
    )
    fig.update_layout(hovermode="x unified")
    return fig


def create_season_points_chart(season_points: pd.DataFrame) -> go.Figure:
    """Interactive cumulative constructor points across rounds."""
    if season_points.empty:
        return go.Figure()
    teams = season_points["constructor_name"].unique()
    return px.line(
        season_points, x="round", y="cumulative_points", color="constructor_name", markers=True,
        color_discrete_map={t: team_color(t) for t in teams},
        labels={"round": "Round", "cumulative_points": "Cumulative points", "constructor_name": "Team"},
        title="Constructor points over the season (race results only)", template="plotly_white",
    )
