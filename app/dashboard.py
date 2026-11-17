"""Interactive Streamlit dashboard for F1 Race Intelligence.

Run from the project root:
    streamlit run app/dashboard.py

All analytical logic lives in ``src/f1_analysis``; this file only wires UI
controls to those functions and renders the returned tables and figures.
"""

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from f1_analysis.analysis.driver_performance import (
    compare_teammates,
    compute_driver_pace_metrics,
)
from f1_analysis.analysis.pitstop_analysis import (
    analyze_undercut,
    compute_pit_stop_summary,
    strategy_vs_result,
)
from f1_analysis.analysis.race_dynamics import compute_net_position_changes
from f1_analysis.analysis.team_performance import (
    compute_constructor_pace_ranking,
    compute_constructor_points_summary,
    compute_season_constructor_points,
)
from f1_analysis.analysis.tyre_analysis import (
    compute_compound_degradation,
    compute_driver_stint_summary,
)
from f1_analysis.data.api_client import DataFetchError
from f1_analysis.data.data_loader import F1DataLoader
from f1_analysis.pipeline import RaceDataset, build_race_dataset
from f1_analysis.visualization.charts import (
    create_interactive_gap_chart,
    create_season_points_chart,
    plot_constructor_pace,
    plot_driver_pace_distributions,
    plot_pit_stop_times,
    plot_position_progression,
    plot_tyre_degradation,
)

st.set_page_config(
    page_title="F1 Race Intelligence | UNIMI",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🏎️ F1 Race Intelligence")
st.caption(
    "Box, Box — Formula 1 Race Analytics | Master's in Data Science, University of Milan (UNIMI)"
)


@st.cache_data(show_spinner=False)
def cached_calendar(year: int) -> pd.DataFrame:
    return F1DataLoader().load_season_calendar(year)


@st.cache_data(show_spinner=False)
def cached_race_dataset(year: int, round_num: int) -> RaceDataset:
    return build_race_dataset(year, round_num)


@st.cache_data(show_spinner=False)
def cached_season_points(year: int, rounds: tuple[int, ...]) -> pd.DataFrame:
    return compute_season_constructor_points(F1DataLoader(), year, list(rounds))


# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
st.sidebar.header("Grand Prix Selection")
selected_year = st.sidebar.selectbox("Season", [2024, 2023], index=0)

try:
    calendar_df = cached_calendar(selected_year)
except DataFetchError:
    calendar_df = pd.DataFrame()

if not calendar_df.empty:
    round_options = calendar_df["round"].tolist()
    labels = {
        int(r["round"]): f"R{int(r['round']):02d} — {r['race_name']}"
        for _, r in calendar_df.iterrows()
    }
    selected_round = st.sidebar.selectbox(
        "Grand Prix",
        options=round_options,
        index=0,
        format_func=lambda r: labels.get(r, f"Round {r}"),
    )
else:
    selected_round = st.sidebar.number_input("Round Number", min_value=1, max_value=24, value=1)

use_fuel_corrected = st.sidebar.checkbox(
    "Apply Fuel-Load Correction",
    value=True,
    help=(
        "Adds back an estimated ~0.03 s/kg fuel mass effect across the race distance "
        "so that lap-time trends reflect tyre wear rather than car weight reduction."
    ),
)
top_n = st.sidebar.slider("Drivers shown in plots", min_value=5, max_value=20, value=10)

# ---------------------------------------------------------------------------
# Load and preprocess race dataset
# ---------------------------------------------------------------------------
try:
    with st.spinner("Loading race dataset (using local cache if available)..."):
        ds = cached_race_dataset(int(selected_year), int(selected_round))
except DataFetchError as exc:
    st.error(
        f"Could not load data for Season {selected_year}, Round {selected_round}: {exc}\n\n"
        f"Run `python scripts/download_data.py --year {selected_year} --rounds {selected_round}` "
        "when connected to the internet to cache this race locally."
    )
    st.stop()

df_results = ds.results
df_laps = ds.laps
df_pitstops = ds.pitstops

if df_results.empty or df_laps.empty:
    st.warning(
        f"No race classification or lap timing data found for {selected_year} Round {selected_round}."
    )
    st.stop()

for note in ds.notes:
    st.info(note)

race_title = ds.info.get("race_name", f"Round {selected_round}")
circuit_name = ds.info.get("circuit", "")

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
    st.subheader(f"{selected_year} {race_title} ({circuit_name})")
    col1, col2, col3, col4 = st.columns(4)
    winner = df_results.iloc[0]["driver_name"]
    winner_team = df_results.iloc[0]["constructor_name"]
    total_laps = int(df_laps["lap_number"].max())
    rep_ratio = df_laps["is_valid_representative_lap"].mean()

    col1.metric("Race Winner", winner)
    col2.metric("Winning Team", winner_team)
    col3.metric("Race Distance", f"{total_laps} laps")
    col4.metric("Representative Laps", f"{rep_ratio:.1%}")

    st.dataframe(
        df_results[
            [
                "position",
                "driver_name",
                "driver_code",
                "constructor_name",
                "grid",
                "laps",
                "status",
                "points",
                "fastest_lap_time_str",
            ]
        ],
        width="stretch",
        hide_index=True,
    )

    with st.expander("Data Cleaning Breakdown (Excluded Laps by Reason)"):
        reasons = (
            df_laps["exclusion_reason"]
            .fillna("representative")
            .value_counts()
            .rename_axis("category")
            .reset_index(name="lap_count")
        )
        st.dataframe(reasons, width="stretch", hide_index=True)

# ---------------------------------------------------------------------------
# TAB 2: DRIVER PACE & CONSISTENCY
# ---------------------------------------------------------------------------
with tab_driver:
    st.subheader("Driver Pace Distribution & Consistency")
    st.markdown(
        "- **Median & IQR ($Q_{75} - Q_{25}$)** are robust to traffic and single-lap mistakes.\n"
        "- **Standard Deviation ($s$)** measures lap-to-lap variability on representative laps."
    )

    fig_pace = plot_driver_pace_distributions(
        df_laps, top_n=top_n, use_fuel_corrected=use_fuel_corrected
    )
    st.pyplot(fig_pace)
    plt.close(fig_pace)

    pace_metrics = compute_driver_pace_metrics(df_laps, use_fuel_corrected=use_fuel_corrected)
    st.dataframe(pace_metrics, width="stretch", hide_index=True)

    st.subheader("Teammate Head-to-Head Comparison")
    st.caption(
        "Note: Welch's t-test and Mann-Whitney U test treat laps as independent observations. "
        "Because consecutive laps are autocorrelated and strategies can differ, treat p-values "
        "as descriptive indicators rather than strict causal proof."
    )
    teammates = compare_teammates(df_laps, df_results, use_fuel_corrected=use_fuel_corrected)
    if not teammates.empty:
        st.dataframe(teammates, width="stretch", hide_index=True)
    else:
        st.info("Insufficient representative laps for teammate comparison.")

# ---------------------------------------------------------------------------
# TAB 3: TYRE DEGRADATION
# ---------------------------------------------------------------------------
with tab_tyre:
    st.subheader("Within-Stint Tyre Degradation Modeling")
    st.markdown(
        r"""
        To prevent faster cars on harder tyres from biasing compound comparisons (Simpson's paradox),
        degradation is estimated **within each stint** by demeaning lap time and tyre age per driver-stint:
        $$y_{i} - \bar{y}_{s} = \beta \, (a_{i} - \bar{a}_{s}) + \varepsilon_{i}$$
        where $\beta$ is the average time lost (in seconds) per extra lap of tyre age.
        """
    )

    deg_summary = compute_compound_degradation(df_laps, use_fuel_corrected=use_fuel_corrected)
    if not deg_summary.empty:
        st.dataframe(deg_summary, width="stretch", hide_index=True)
        fig_deg = plot_tyre_degradation(df_laps, use_fuel_corrected=use_fuel_corrected)
        st.pyplot(fig_deg)
        plt.close(fig_deg)
    else:
        st.info("No tyre compound stint data available for this race.")

    st.subheader("Individual Driver Stint Summary")
    stint_summary = compute_driver_stint_summary(df_laps, use_fuel_corrected=use_fuel_corrected)
    if not stint_summary.empty:
        st.dataframe(stint_summary, width="stretch", hide_index=True)

# ---------------------------------------------------------------------------
# TAB 4: PIT STOP STRATEGY
# ---------------------------------------------------------------------------
with tab_pit:
    st.subheader("Pit-Lane Time & Stint Strategy Summary")
    st.caption(
        "Jolpica reports total pit-lane transit time (pit entry to pit exit), not stationary jack time. "
        "Stops exceeding 2× the race median (e.g. red-flag periods or wing repairs) are flagged as anomalous."
    )
    if not df_pitstops.empty:
        pit_summary = compute_pit_stop_summary(df_pitstops, df_results)
        fig_pit = plot_pit_stop_times(pit_summary)
        st.pyplot(fig_pit)
        plt.close(fig_pit)
        st.dataframe(pit_summary, width="stretch", hide_index=True)

        st.subheader("Observed Strategy vs. Classification")
        strat_df = strategy_vs_result(df_pitstops, df_results, df_laps)
        st.dataframe(strat_df, width="stretch", hide_index=True)

        st.subheader("Undercut Evaluation (Observed Pit Sequence)")
        drivers_list = df_results["driver_id"].tolist()
        if len(drivers_list) >= 2:
            c1, c2, c3 = st.columns(3)
            attacker = c1.selectbox("Attacker (pitted earlier)", drivers_list, index=min(1, len(drivers_list) - 1))
            defender = c2.selectbox("Defender (pitted later)", drivers_list, index=0)
            stop_num = c3.number_input("Stop Number", min_value=1, max_value=5, value=1)
            uc = analyze_undercut(df_laps, df_pitstops, attacker, defender, int(stop_num))
            st.json(uc)
    else:
        st.info("No pit-stop records found for this race.")

# ---------------------------------------------------------------------------
# TAB 5: RACE DYNAMICS
# ---------------------------------------------------------------------------
with tab_dynamics:
    st.subheader("Lap-by-Lap Track Position Chart")
    fig_pos = plot_position_progression(df_laps, top_n=top_n)
    st.pyplot(fig_pos)
    plt.close(fig_pos)

    st.subheader("Interactive Gap to Race Leader")
    fig_gap = create_interactive_gap_chart(df_laps, top_n=min(top_n, 10))
    st.plotly_chart(fig_gap, width="stretch")

    st.subheader("Net Positions Gained / Lost vs. Starting Grid")
    net_pos = compute_net_position_changes(df_results)
    st.dataframe(net_pos, width="stretch", hide_index=True)

# ---------------------------------------------------------------------------
# TAB 6: CONSTRUCTORS
# ---------------------------------------------------------------------------
with tab_teams:
    st.subheader("Constructor Race Pace Ranking")
    team_pace = compute_constructor_pace_ranking(
        df_laps, df_results, use_fuel_corrected=use_fuel_corrected
    )
    if not team_pace.empty:
        fig_team = plot_constructor_pace(team_pace)
        st.pyplot(fig_team)
        plt.close(fig_team)
        st.dataframe(team_pace, width="stretch", hide_index=True)

    st.subheader("Race Points Summary by Constructor")
    team_points = compute_constructor_points_summary(df_results)
    st.dataframe(team_points, width="stretch", hide_index=True)

    st.subheader("Multi-Round Constructor Points Trend")
    max_r = int(selected_round)
    default_rounds = tuple(range(1, max(2, min(max_r + 1, 4))))
    try:
        season_pts = cached_season_points(int(selected_year), default_rounds)
        if not season_pts.empty:
            st.plotly_chart(create_season_points_chart(season_pts), width="stretch")
    except DataFetchError:
        st.info("Additional rounds not cached locally for multi-round trend.")
