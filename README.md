# F1 Race Intelligence: Driver Performance, Tyre Degradation, and Pit-Stop Strategy Analytics

An advanced, reproducible Formula 1 analytics and scientific computing application developed for the **Master's Degree in Data Science** (Course: *Coding for Data Science and Data Management*, Python Module) at the **University of Milan (UNIMI)**.

Based on the official examination project specification: **"Box, Box — Formula 1 Race Analytics"**.

---

## 1. Project Overview

Formula 1 races are dynamic events where tactical decisions, vehicle dynamics, tyre degradation, and pit-stop timings determine outcomes. This project implements a modular Python software package that ingests, cleans, processes, and statistically analyzes timing and telemetry data from real Grand Prix events without relying on obsolete or subscription-gated libraries (such as FastF1).

### Core Analytical Objectives:
- **Driver Performance:** Lap time distributions, pace consistency metrics (IQR, standard deviation), and teammate head-to-head comparisons.
- **Tyre Degradation Modeling:** Fuel-load corrected lap-time evolution, compound deltas (Soft, Medium, Hard), and linear/polynomial degradation estimation using SciPy.
- **Pit-Stop Strategy:** In-lap/out-lap pace impacts, undercut vs. overcut efficacy, and compound lifecycle dynamics.
- **Race Dynamics & Gaps:** Lap chart progression, gap evolution relative to the leader, and safety car/VSC neutralization effects.
- **Constructor Pace:** Team-level pace aggregates and championship trajectories.

---

## 2. Project Architecture

```
f1-race-intelligence/
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/                  # Cached API responses (JSON/CSV, Git-ignored)
│   ├── processed/            # Cleaned, standardized analytical datasets
│   └── README.md
│
├── notebooks/
│   └── exploratory_analysis.ipynb
│
├── src/
│   └── f1_analysis/
│       ├── __init__.py
│       ├── config.py
│       │
│       ├── data/             # API client, caching & data ingestion
│       │   ├── __init__.py
│       │   ├── api_client.py
│       │   ├── data_loader.py
│       │   └── preprocessing.py
│       │
│       ├── analysis/         # Statistical & scientific computing modules
│       │   ├── __init__.py
│       │   ├── driver_performance.py
│       │   ├── tyre_analysis.py
│       │   ├── pitstop_analysis.py
│       │   ├── race_dynamics.py
│       │   └── team_performance.py
│       │
│       ├── visualization/    # Matplotlib, Seaborn & Plotly charts
│       │   ├── __init__.py
│       │   └── charts.py
│       │
│       └── utils/            # Data validation and math helpers
│           ├── __init__.py
│           └── validation.py
│
├── app/
│   └── dashboard.py          # Interactive Streamlit application
│
└── tests/                    # Unit and integration tests (pytest)
    ├── __init__.py
    ├── test_preprocessing.py
    ├── test_driver_performance.py
    └── test_tyre_analysis.py
```

---

## 3. Data Sources

In strict accordance with examination instructions, data is retrieved via open APIs and cached locally:
1. **Jolpica F1 API** (`https://api.jolpi.ca/ergast/f1/`): Race results, season calendars, lap times, pit stops, and championship standings.
2. **OpenF1 API** (`https://openf1.org/`): High-resolution session telemetry, stint intervals, tyre compound usage, and track status.

---

## 4. Setup and Installation

### Prerequisites
- Python 3.10+
- Git

### Installation Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Montasir00/f1-race-intelligence.git
   cd f1-race-intelligence
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   # Windows (PowerShell)
   .\.venv\Scripts\Activate.ps1
   # Linux / macOS
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 5. Development Status

- [x] Phase 1: Environment & repository architecture setup.
- [ ] Phase 2: Ingestion & API caching client (Jolpica + OpenF1).
- [ ] Phase 3: Preprocessing, fuel-load correction & outlier filtering.
- [ ] Phase 4: Core statistical & scientific computing modules.
- [ ] Phase 5: Publication visualizations.
- [ ] Phase 6: Streamlit dashboard & Jupyter presentation notebook.
- [ ] Phase 7: Pytest test suite & examination defense preparation.
