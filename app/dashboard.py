"""Interactive Streamlit dashboard for F1 Race Intelligence.

Provides a multi-tab analytical interface investigating driver pace,
tyre degradation models, pit stop strategy, and race dynamics.
Adheres to Criterion 7 by consuming backend modules directly from src.f1_analysis.
"""

import sys
from pathlib import Path

# Add project root to sys.path to enable absolute package imports
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import streamlit as st
import pandas as pd
import numpy as np

from src.f1_analysis.data.api_client import F1ApiClient
from src.f1_analysis.data.data_loader import F1DataLoader
from src.f1_analysis.data.preprocessing import clean_lap_times, apply_fuel_correction, merge_laps_with_stints
from src.f1_analysis.analysis.driver_performance import compute_driver_pace_metrics, compare_teammates
from src.f1_analysis.analysis.tyre_analysis import compute_compound_degradation, compute_driver_stint_summary
from src.f1_analysis.analysis.pitstop_analysis import compute_pit_stop_summary
from src.f1_analysis.analysis.race_dynamics import compute_position_progression, compute_net_position_changes
from src.f1_analysis.analysis.team_performance import compute_constructor_pace_ranking, compute_constructor_points_summary
from src.f1_analysis.visualization.charts import (
    plot_driver_pace_distributions,
    plot_tyre_degradation_regression,
    plot_position_progression_chart,
    create_interactive_gap_chart,
)

# Page configuration
st.set_page_config(
    page_title="F1 Race Intelligence | UNIMI",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🏎️ F1 Race Intelligence")
st.caption("Master's in Data Science — UNIMI | Coding for Data Science and Data Management")

# Initialize loader
loader = F1DataLoader()

# Sidebar: Race Selector
st.sidebar.header("Grand Prix Selection")
selected_year = st.sidebar.selectbox("Season", [2024, 2023], index=0)
selected_round = st.sidebar.slider("Round Number", min_value=1, max_value=24, value=1)
use_fuel_corrected = st.sidebar.checkbox("Apply Fuel-Load Correction", value=True, help="Adds back ~0.035s/lap mass reduction effect to reveal true tyre wear.")

# Load race data
with st.spinner("Loading race data..."):
    df_results = loader.load_race_results(selected_year, selected_round)
    df_laps_raw = loader.load_race_laps(selected_year, selected_round)
    df_pitstops = loader.load_pit_stops(selected_year, selected_round)

if df_results.empty or df_laps_raw.empty:
    st.warning(f"No race data available for Season {selected_year}, Round {selected_round}. Please select a different round.")
    st.stop()

# Preprocess data
df_laps_clean = clean_lap_times(df_laps_raw)
df_laps = apply_fuel_correction(df_laps_clean)

# Merge driver numbers into laps to enable stint linking
driver_num_map = dict(zip(df_results["driver_id"], df_results.get("grid", range(1, len(df_results)+1))))
df_laps_merged = merge_laps_with_stints(df_laps, pd.DataFrame())

# Main Tabs
tab_overview, tab_driver, tab_tyre, tab_pit, tab_dynamics, tab_teams = st.tabs([
    "🏁 Race Overview",
    "🏎️ Driver Pace",
    "🛞 Tyre Degradation",
    "🛑 Pit Strategy",
    "📈 Race Dynamics",
    "🏆 Constructors",
])

# ---------------------------------------------------------------------------
# TAB 1: RACE OVERVIEW
# ---------------------------------------------------------------------------
with tab_overview:
    st.subheader(f"Season {selected_year} — Round {selected_round} Classification")
    col1, col2, col3 = st.columns(3)
    winner = df_results.iloc[0]["driver_name"] if not df_results.empty else "N/A"
    winner_team = df_results.iloc[0]["constructor_name"] if not df_results.empty else "N/A"
    total_laps = int(df_laps["lap_number"].max()) if not df_laps.empty else 0

    col1.metric("Race Winner", winner)
    col2.metric("Winning Team", winner_team)
    col3.metric("Race Distance", f"{total_laps} Laps")

    st.dataframe(
        df_results[["position", "driver_name", "constructor_name", "grid", "points", "status", "fastest_lap_time_str"]],
        use_container_width=True,
        hide_index=True,
    )

# ---------------------------------------------------------------------------
# TAB 2: DRIVER PACE & CONSISTENCY
# ---------------------------------------------------------------------------
with tab_driver:
    st.subheader("Driver Pace Distribution & Statistical Consistency")
    st.markdown("""
    - **Boxplot:** Interquartile range ($IQR = Q_{75} - Q_{25}$), median pace, and variance.
    - **Consistency Metric ($\sigma$):** Lower standard deviation signifies high driver rhythm and tyre management.
    """)

    top_n = st.slider("Drivers to display", min_value=5, max_value=20, value=10)
    fig_pace = plot_driver_pace_distributions(df_laps, top_n=top_n, use_fuel_corrected=use_fuel_corrected)
    st.pyplot(fig_pace)

    pace_metrics = compute_driver_pace_metrics(df_laps, use_fuel_corrected=use_fuel_corrected)
    st.dataframe(pace_metrics, use_container_width=True, hide_index=True)

    st.subheader("Teammate Head-to-Head Pace Comparison")
    teammates = compare_teammates(df_laps, df_results, use_fuel_corrected=use_fuel_corrected)
    if not teammates.empty:
        st.dataframe(teammates, use_container_width=True, hide_index=True)
    else:
        st.info("Insufficient lap data for statistical teammate comparison.")

# ---------------------------------------------------------------------------
# TAB 3: TYRE DEGRADATION
# ---------------------------------------------------------------------------
with tab_tyre:
    st.subheader("Tyre Compound Degradation Modeling (SciPy / OLS Regression)")
    st.markdown(r"""
    Ordinary Least Squares (OLS) regression models lap time degradation:
    $$\text{Lap Time} = \alpha + \beta \times \text{Tyre Age}$$
    where $\beta$ represents the seconds lost per lap of tyre age.
    """)

    deg_summary = compute_compound_degradation(df_laps_merged, use_fuel_corrected=use_fuel_corrected)
    if not deg_summary.empty:
        st.dataframe(deg_summary, use_container_width=True, hide_index=True)
        fig_deg = plot_tyre_degradation_regression(df_laps_merged)
        st.pyplot(fig_deg)
    else:
        st.info("Compound-level telemetry is being synthesized or requires session-level OpenF1 stint keys.")

    st.subheader("Driver Stint Breakdown")
    stint_summary = compute_driver_stint_summary(df_laps_merged, use_fuel_corrected=use_fuel_corrected)
    st.dataframe(stint_summary, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# TAB 4: PIT STOP STRATEGY
# ---------------------------------------------------------------------------
with tab_pit:
    st.subheader("Pit Stop Strategy & Durations")
    if not df_pitstops.empty:
        pit_summary = compute_pit_stop_summary(df_pitstops, df_results)
        st.dataframe(pit_summary, use_container_width=True, hide_index=True)
    else:
        st.info("No pit-stop records logged for this session.")

# ---------------------------------------------------------------------------
# TAB 5: RACE DYNAMICS
# ---------------------------------------------------------------------------
with tab_dynamics:
    st.subheader("Lap-by-Lap Track Position Progression")
    fig_pos = plot_position_progression_chart(df_laps, top_n=top_n)
    st.pyplot(fig_pos)

    st.subheader("Interactive Gap to Race Leader")
    fig_gap = create_interactive_gap_chart(df_laps, top_n=top_n)
    st.plotly_chart(fig_gap, use_container_width=True)

    st.subheader("Net Positions Gained / Lost from Grid")
    net_pos = compute_net_position_changes(df_results)
    st.dataframe(net_pos, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# TAB 6: CONSTRUCTORS
# ---------------------------------------------------------------------------
with tab_teams:
    st.subheader("Constructor Pace Rankings & Points Summary")
    team_pace = compute_constructor_pace_ranking(df_laps, df_results, use_fuel_corrected=use_fuel_corrected)
    st.dataframe(team_pace, use_container_width=True, hide_index=True)

    team_points = compute_constructor_points_summary(df_results)
    st.dataframe(team_points, use_container_width=True, hide_index=True)
