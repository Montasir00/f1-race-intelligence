"""Tests for visualization functions in charts.py."""

import matplotlib.pyplot as plt
import pandas as pd

from f1_analysis.analysis.pitstop_analysis import compute_pit_stop_summary
from f1_analysis.analysis.team_performance import compute_constructor_pace_ranking
from f1_analysis.visualization.charts import (
    create_interactive_gap_chart,
    create_season_points_chart,
    plot_constructor_pace,
    plot_driver_pace_distributions,
    plot_pit_stop_times,
    plot_position_progression,
    plot_tyre_degradation,
)


def _sample_laps() -> pd.DataFrame:
    return pd.DataFrame({
        "driver_id": ["ver"] * 6 + ["lec"] * 6,
        "lap_number": list(range(1, 7)) * 2,
        "position": [1] * 6 + [2] * 6,
        "lap_time_sec": [90.0 + 0.1 * i for i in range(6)] + [90.5 + 0.12 * i for i in range(6)],
        "fuel_corrected_lap_sec": [90.0 + 0.15 * i for i in range(6)] + [90.5 + 0.17 * i for i in range(6)],
        "compound": ["SOFT"] * 6 + ["HARD"] * 6,
        "stint_number": [1] * 12,
        "tyre_age": list(range(1, 7)) * 2,
        "is_valid_representative_lap": [True] * 12,
    })


def test_matplotlib_charts_return_non_empty_figures(results):
    laps = _sample_laps()
    pits = pd.DataFrame({
        "driver_id": ["ver", "lec"],
        "lap": [3, 4],
        "stop_number": [1, 1],
        "pit_lane_time_sec": [22.1, 22.8],
    })
    pit_summary = compute_pit_stop_summary(pits, results)
    team_pace = compute_constructor_pace_ranking(laps, results)

    for fig in (
        plot_driver_pace_distributions(laps),
        plot_tyre_degradation(laps),
        plot_position_progression(laps),
        plot_pit_stop_times(pit_summary),
        plot_constructor_pace(team_pace),
    ):
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) >= 1
        plt.close(fig)


def test_plotly_charts_return_traces():
    laps = _sample_laps()
    gap_fig = create_interactive_gap_chart(laps)
    assert len(gap_fig.data) == 2

    season_df = pd.DataFrame({
        "round": [1, 2, 1, 2],
        "constructor_name": ["Red Bull", "Red Bull", "Ferrari", "Ferrari"],
        "points": [43.0, 43.0, 27.0, 30.0],
        "cumulative_points": [43.0, 86.0, 27.0, 57.0],
    })
    pts_fig = create_season_points_chart(season_df)
    assert len(pts_fig.data) == 2
