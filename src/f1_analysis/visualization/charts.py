"""Publication-quality visualization module for F1 Race Intelligence.

Generates broadcast-grade figures using Matplotlib, Seaborn, and Plotly, adhering
to official F1 compound and team color conventions.
"""

import os
import tempfile
from typing import Dict, List, Optional
import matplotlib
# Use non-interactive backend for headless/server execution
matplotlib.use("Agg")
# Set a writable temporary config directory to prevent Windows permission warnings
os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

from src.f1_analysis.config import COMPOUND_COLORS, CONSTRUCTOR_COLORS

# Configure professional dark/light styling
sns.set_theme(style="whitegrid", font="sans-serif")


def plot_driver_pace_distributions(
    df_laps: pd.DataFrame,
    df_results: Optional[pd.DataFrame] = None,
    top_n: int = 10,
    use_fuel_corrected: bool = False,
    figsize: tuple = (12, 6),
) -> plt.Figure:
    """Generate publication boxplot + stripplot showing driver pace distribution and consistency.

    Args:
        df_laps: DataFrame containing lap times with ['driver_id', 'lap_time_sec'].
        df_results: Optional race classification to sort drivers by finishing order.
        top_n: Number of drivers to display.
        use_fuel_corrected: If True, uses 'fuel_corrected_lap_sec'.
        figsize: Matplotlib figure size tuple.

    Returns:
        Matplotlib Figure object.
    """
    col = (
        "fuel_corrected_lap_sec"
        if use_fuel_corrected and "fuel_corrected_lap_sec" in df_laps.columns
        else "lap_time_sec"
    )

    valid = df_laps[df_laps.get("is_valid_representative_lap", True)].copy()
    valid = valid[valid[col].notna() & (valid[col] > 0)]

    if valid.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No data available", ha="center", va="center")
        return fig

    # Determine driver ordering (by median pace or finish position)
    order = (
        valid.groupby("driver_id")[col]
        .median()
        .sort_values()
        .head(top_n)
        .index.tolist()
    )
    plot_data = valid[valid["driver_id"].isin(order)]

    fig, ax = plt.subplots(figsize=figsize, dpi=120)

    # Boxplot for quartiles and IQR consistency
    sns.boxplot(
        data=plot_data,
        x="driver_id",
        y=col,
        order=order,
        palette="Blues_r",
        showmeans=True,
        meanprops={"marker": "o", "markerfacecolor": "red", "markeredgecolor": "red"},
        ax=ax,
        fliersize=0,
        boxprops=dict(alpha=0.7),
    )

    # Overlay strip points for transparency of sample density
    sns.stripplot(
        data=plot_data,
        x="driver_id",
        y=col,
        order=order,
        color="black",
        alpha=0.25,
        size=4,
        jitter=0.2,
        ax=ax,
    )

    title_metric = "Fuel-Corrected Lap Time (s)" if use_fuel_corrected else "Raw Lap Time (s)"
    ax.set_title(f"Driver Pace Distribution & Consistency (Top {top_n})", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Driver ID", fontsize=11, fontweight="bold")
    ax.set_ylabel(title_metric, fontsize=11, fontweight="bold")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()

    return fig


def plot_tyre_degradation_regression(
    df_laps_with_stints: pd.DataFrame,
    figsize: tuple = (11, 6),
) -> plt.Figure:
    """Generate scatter plot with OLS linear regression fits per tyre compound.

    Visualizes degradation curves: lap time vs. tyre age, color-coded by compound.

    Args:
        df_laps_with_stints: Cleaned DataFrame with ['compound', 'tyre_age', 'lap_time_sec']
                             or 'fuel_corrected_lap_sec'.

    Returns:
        Matplotlib Figure object.
    """
    col = (
        "fuel_corrected_lap_sec"
        if "fuel_corrected_lap_sec" in df_laps_with_stints.columns
        else "lap_time_sec"
    )

    df = df_laps_with_stints[df_laps_with_stints.get("is_valid_representative_lap", True)].copy()
    df = df[df[col].notna() & df["tyre_age"].notna() & (df["compound"] != "UNKNOWN")]

    fig, ax = plt.subplots(figsize=figsize, dpi=120)

    for compound, group in df.groupby("compound"):
        color = COMPOUND_COLORS.get(compound.upper(), "#333333")
        # In dark or light background, make white hard visible
        edge_color = "black" if color == "#FFFFFF" else color
        fill_color = "#E0E0E0" if color == "#FFFFFF" else color

        # Scatter
        ax.scatter(
            group["tyre_age"],
            group[col],
            label=f"{compound} (n={len(group)})",
            color=fill_color,
            edgecolors=edge_color,
            alpha=0.6,
            s=35,
        )

        # Regression line
        if len(group) >= 5 and group["tyre_age"].nunique() > 1:
            x = group["tyre_age"].values.astype(float)
            y = group[col].values.astype(float)
            p = np.polyfit(x, y, deg=1)
            x_range = np.linspace(x.min(), x.max(), 50)
            ax.plot(
                x_range,
                np.polyval(p, x_range),
                color=edge_color,
                linewidth=2.5,
                linestyle="--",
            )

    ax.set_title("Tyre Compound Degradation (Fuel-Corrected Pace vs. Tyre Age)", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Tyre Age (Laps Completed on Set)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Fuel-Corrected Lap Time (s)", fontsize=11, fontweight="bold")
    ax.legend(title="Compound", frameon=True)
    plt.tight_layout()

    return fig


def plot_position_progression_chart(
    df_laps: pd.DataFrame,
    top_n: int = 10,
    figsize: tuple = (13, 7),
) -> plt.Figure:
    """Generate broadcast lap chart showing track position evolution across the Grand Prix.

    Args:
        df_laps: Lap times DataFrame with ['lap_number', 'driver_id', 'position'].
        top_n: Display top N drivers at race finish.

    Returns:
        Matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=120)

    # Determine top drivers
    max_lap = df_laps["lap_number"].max()
    final_positions = (
        df_laps[df_laps["lap_number"] == max_lap]
        .sort_values(by="position")
        .head(top_n)["driver_id"]
        .tolist()
    )

    palette = sns.color_palette("tab10", n_colors=len(final_positions))

    for idx, driver_id in enumerate(final_positions):
        driver_data = df_laps[df_laps["driver_id"] == driver_id].sort_values(by="lap_number")
        ax.plot(
            driver_data["lap_number"],
            driver_data["position"],
            label=driver_id,
            color=palette[idx],
            linewidth=2.2,
            marker="o",
            markersize=3,
            alpha=0.85,
        )

    # Invert y-axis so P1 is at the top
    ax.invert_yaxis()
    ax.set_yticks(range(1, top_n + 1))
    ax.set_title("Race Position Evolution (Lap Chart)", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Lap Number", fontsize=11, fontweight="bold")
    ax.set_ylabel("Track Position (P1 = Leader)", fontsize=11, fontweight="bold")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", title="Driver")
    plt.tight_layout()

    return fig


def create_interactive_gap_chart(df_laps: pd.DataFrame, top_n: int = 8) -> go.Figure:
    """Create an interactive Plotly chart of time gap to the leader over the race distance.

    Args:
        df_laps: DataFrame with ['lap_number', 'driver_id', 'lap_time_sec'].
        top_n: Number of drivers to display.

    Returns:
        Plotly Figure object.
    """
    from src.f1_analysis.analysis.race_dynamics import compute_gap_to_leader

    gap_df = compute_gap_to_leader(df_laps)
    if gap_df.empty:
        return go.Figure()

    # Filter to top N drivers by minimum average gap
    top_drivers = (
        gap_df.groupby("driver_id")["gap_to_leader_sec"]
        .mean()
        .sort_values()
        .head(top_n)
        .index.tolist()
    )
    plot_data = gap_df[gap_df["driver_id"].isin(top_drivers)]

    fig = px.line(
        plot_data,
        x="lap_number",
        y="gap_to_leader_sec",
        color="driver_id",
        title="Interval Gap to Race Leader Over Distance",
        labels={"lap_number": "Lap Number", "gap_to_leader_sec": "Gap to Leader (seconds)", "driver_id": "Driver"},
        template="plotly_white",
    )
    fig.update_layout(hovermode="x unified", legend_title_text="Driver")
    return fig
