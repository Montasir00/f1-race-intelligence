# F1 Race Intelligence: Driver Performance, Tyre Degradation, and Pit-Stop Strategy Analytics

A modular Formula 1 analytics and scientific computing application developed for the **Master's Degree in Data Science** (*Coding for Data Science and Data Management*, Python Module) at the **University of Milan (UNIMI)**.

Based on the official examination project specification: **"Box, Box — Formula 1 Race Analytics"**.

---

## 1. Project Overview

Formula 1 races are governed by tyre degradation, fuel mass reduction, pit-lane time loss, and track position dynamics. This project retrieves, validates, cleans, analyzes, and visualizes lap-by-lap timing and stint data from real Grand Prix weekends without depending on `FastF1`.

### Analytical Modules
- **Module A — Driver Performance (`driver_performance.py`):** Parametric (mean, sample standard deviation) and non-parametric (median, interquartile range) pace metrics, plus intra-team teammate comparisons (Welch's $t$-test and Mann-Whitney $U$ test).
- **Module B — Tyre Degradation (`tyre_analysis.py`):** Within-stint (fixed-effects demeaned) Ordinary Least Squares regression (`scipy.stats.linregress`) estimating seconds lost per lap of tyre age by compound, isolating tyre wear from car-performance differences and fuel burn-off.
- **Module C — Pit-Stop Strategy (`pitstop_analysis.py`):** Pit-lane transit time summaries, anomalous stop detection (e.g. red-flag periods or damage repairs), compound stint sequences, and observed undercut net time gain calculations.
- **Module D — Race Dynamics (`race_dynamics.py`):** Lap-by-lap position matrices, cumulative elapsed-time gaps to the race leader, and net positions gained/lost relative to starting grid slots (with explicit handling of pit-lane starts and retirements).
- **Module E — Constructor Performance (`team_performance.py`):** Team-level pace rankings, points summaries, double top-10 finishes, and multi-round cumulative points progression.

---

## 2. Project Architecture

```text
f1-race-intelligence/
├── README.md
├── LICENSE
├── pyproject.toml
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/                        # Cached API responses (JSON, Git-ignored)
│   ├── processed/                  # Cleaned & merged CSV tables (Git-ignored)
│   └── README.md                   # Reproducible data download instructions
│
├── scripts/
│   └── download_data.py            # CLI script to fetch, validate, and cache race data
│
├── notebooks/
│   └── exploratory_analysis.ipynb  # Executed step-by-step analysis & verification notebook
│
├── src/
│   └── f1_analysis/
│       ├── __init__.py
│       ├── config.py               # Paths, API constants, modelling parameters, palettes
│       ├── pipeline.py             # End-to-end RaceDataset builder shared by UI & notebook
│       │
│       ├── data/
│       │   ├── __init__.py
│       │   ├── api_client.py       # Paginated Jolpica + OpenF1 client with cache validation
│       │   ├── data_loader.py      # JSON-to-DataFrame normalization & session matching
│       │   └── preprocessing.py    # Pit-lap flagging, outlier tagging, fuel correction, stint join
│       │
│       ├── analysis/
│       │   ├── __init__.py
│       │   ├── driver_performance.py
│       │   ├── tyre_analysis.py
│       │   ├── pitstop_analysis.py
│       │   ├── race_dynamics.py
│       │   └── team_performance.py
│       │
│       ├── visualization/
│       │   ├── __init__.py
│       │   └── charts.py           # Matplotlib, Seaborn, and Plotly figures
│       │
│       └── utils/
│           ├── __init__.py
│           └── validation.py       # Schema checks and representative-lap helpers
│
├── app/
│   └── dashboard.py                # Multi-tab Streamlit dashboard importing f1_analysis
│
└── tests/
    ├── __init__.py
    ├── conftest.py
    ├── test_api_client.py
    ├── test_preprocessing.py
    ├── test_driver_performance.py
    ├── test_tyre_analysis.py
    └── test_strategy_dynamics_team.py
```

---

## 3. Data Sources & Methodological Notes

### Data Sources
From the official *Box, Box* specification (Jolpica F1 API, OpenF1 API, TracingInsights):
1. **Jolpica F1 API** (`https://api.jolpi.ca/ergast/f1`): Season calendars, final classifications, full lap-by-lap timing (`limit=100` pagination with `MRData.total` completeness verification), and pit-stop records.
2. **OpenF1 API** (`https://api.openf1.org/v1`): Race session metadata and driver tyre stints (`compound`, `stint_number`, `lap_start`, `lap_end`, `tyre_age_at_start`), linked to Jolpica via each driver's permanent racing number (`driver_number`). Available from the 2023 season onwards.

### Key Methodological Assumptions & Limitations
- **Fuel-Load Correction:** Cars burn roughly $100\text{ kg}$ of fuel over a race distance. Using a standard engineering approximation of $0.03\text{ s/kg}$, lap times are normalized to lap-1 mass via:
  $$\text{LapTime}_{\text{corrected}} = \text{LapTime}_{\text{raw}} + (\text{LapNumber} - 1) \times \frac{0.03 \times 100}{N_{\text{laps}}}$$
  Without this adjustment, cars getting lighter over a stint bias raw tyre degradation slopes downward.
- **Within-Stint Tyre Degradation:** Pooling laps across all drivers confounds car speed with compound choice (e.g., faster teams running harder compounds longer). `compute_compound_degradation` subtracts each driver-stint's mean lap time and mean tyre age before fitting OLS, estimating degradation strictly within stints.
- **Autocorrelation in Lap-Time Tests:** Welch's $t$-test and Mann-Whitney $U$ test treat laps as independent observations. Because consecutive laps experience shared fuel, traffic, and track evolution, $p$-values are optimistic and should be interpreted alongside effect sizes (median delta and IQR).
- **Pit-Lane Time vs. Stationary Time:** Jolpica's `duration` measures total time between pit-lane entry and exit timing loops (~20–25 s), not stationary tyre-change time (~2–3 s).

---

## 4. Setup, Data Download, and Execution

### 1. Install Dependencies
```bash
git clone https://github.com/Montasir00/f1-race-intelligence.git
cd f1-race-intelligence
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Download & Cache Race Data
```bash
python scripts/download_data.py --year 2024 --rounds 1 2 3
```

### 3. Run the Test Suite
```bash
pytest -v
```

### 4. Launch the Presentations
- **Jupyter Notebook:** Open `notebooks/exploratory_analysis.ipynb` in Jupyter Notebook / JupyterLab.
- **Streamlit Dashboard:**
  ```bash
  streamlit run app/dashboard.py
  ```
