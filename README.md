# AgriTwin AI — Intelligent Crop Digital Twin & Multi-Model AI Simulation

End-to-end simulation of the **AgriTwin AI** pipeline: synthetic multi-sensor telemetry → Digital Twin State Mapping (DTSM) feature fusion → Multi-Model AI prediction engine (Random Forest, SVM, pure NumPy LSTM) → Monte Carlo confidence quantification → Next-Gen 5-Tab Interactive Dashboard.

```mermaid
flowchart LR
    A[Synthetic Telemetry\n5-min Sensors + Noise] --> B[DTSM Feature Fusion\nEWMA + Multi-Horizon Lags + VPD]
    B --> C1[Random Forest\nIrrigation Trigger]
    B --> C2[SVM (RBF Kernel)\nCrop Stress Anomaly]
    B --> C3[NumPy LSTM\n3-Day Moisture Trend]
    C1 & C2 & C3 --> D[Confidence Calibration\nTree-Votes, Platt, MC-Dropout]
    D --> E1[Interactive Streamlit Dashboard\n5-Tab Analytics & Sandbox]
    D --> E2[Static Executive Report\nSelf-Contained HTML Backup]
```

> **Important:** This is a **software simulation**, using synthetic data generated with realistic agronomic physics (evapotranspiration, diurnal cycles, vapor pressure deficit, decay curves). All model metrics are computed live by actually training on this dataset — genuine numbers from executable code.

---

## 1. Setup & Installation

Requires Python 3.9+.

```bash
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # Mac/Linux

pip install -r requirements.txt
```

---

## 2. Running the Pipeline

### Option A: One-Click Master Runner (Recommended)
Run all 5 stages end-to-end with progress timing:
```bash
python run_pipeline.py
```

### Option B: Step-by-Step Execution
```bash
python generate_data.py          # Step 1: Synthetic telemetry + preview_20days.png
python feature_engineering.py    # Step 2: DTSM fusion -> features.csv, feature_importance.png
python train_models.py           # Step 3: Train RF / SVM / LSTM + model_validation_summary.png
python confidence_scoring.py     # Step 4: MC-Dropout & calibration -> confidence_calibration.png
python build_static_report.py    # Step 5: Builds AgriTwin_Simulation_Report.html (Backup)
```

---

## 3. Live Interactive Dashboard

Launch the Streamlit web dashboard:
```bash
streamlit run dashboard.py
```

### 🌟 5 Dashboard Tabs Included:
1. **🛰️ Live Twin & Inference**: Real-time multi-sensor telemetry, RF irrigation trigger, SVM crop stress anomaly alert, LSTM 3-day forecast with $\pm\sigma$ Monte Carlo confidence bounds, and live moisture crash injection.
2. **🚜 Prescriptive Agronomic Advisory**: Automated computation of root-zone water deficit in $mm$ and $m^3/\text{hectare}$, pump operating duration, optimal dawn/dusk scheduling, canopy cooling alerts, and pH guidance.
3. **🔬 Digital Twin State Fusion (DTSM)**: Comparison of raw noisy sensor signals against smoothed EWMA digital twin state, plus Gini importance ranking of features.
4. **📊 Model Performance & Monte Carlo Diagnostics**: 100-iteration Monte Carlo accuracy distribution, precision/recall stats, Expected Calibration Error (ECE), and Brier score.
5. **🎮 What-If Scenario Sandbox**: Interactively tweak temperature, soil moisture, and rainfall sliders to test AI model resilience and decision triggers under extreme weather shocks.

---

## 4. Static Backup Report for Presentations

If Streamlit cannot be launched on presentation day, simply double-click **`AgriTwin_Simulation_Report.html`** in any web browser. It is a single, self-contained, responsive HTML file containing all computed metrics, summary tables, and embedded diagnostic charts.

---

## 5. File Guide

| File | Role / Description |
|---|---|
| `run_pipeline.py` | Master orchestrator running all pipeline stages sequentially |
| `generate_data.py` | Step 1 — Correlated 5-min sensor simulator & `preview_20days.png` |
| `feature_engineering.py` | Step 2 — DTSM adaptive EWMA filtering, VPD calculation, lag extraction & `feature_importance.png` |
| `train_models.py` | Step 3 — Trains RF, SVM, and pure NumPy LSTM via BPTT with 100-iter Monte Carlo validation & `model_validation_summary.png` |
| `confidence_scoring.py` | Step 4 — MC-Dropout uncertainty, Brier score, ECE & `confidence_calibration.png` |
| `dashboard.py` | Step 5 — Modern 5-tab interactive Streamlit dashboard with prescriptive advisory & what-if simulator |
| `build_static_report.py` | Step 6 — Generates the publication-grade `AgriTwin_Simulation_Report.html` |
| `sensor_data.csv` | Synthetic sensor readings (90 days, 5-minute resolution) |
| `features.csv` | Fused digital twin state and engineered features |
| `models.pkl` | Serialized Random Forest + SVM models |
| `lstm_weights.npz` | Trained LSTM weights (pure NumPy, zero framework dependencies) |
| `model_results.json` | Real computed validation metrics across 100 Monte Carlo iterations |
| `confidence_results.json` | Empirical calibration metrics (quartile accuracy, ECE, Brier score) |

---

## 6. Technical Highlights

- **Dependency-Free NumPy LSTM**: Implemented from scratch with forward pass, Backpropagation-Through-Time (BPTT), Adam optimizer, and 50-pass Monte Carlo Dropout inference.
- **Physics-Grounded DTSM**: Accounts for diurnal thermal cycles, Vapor Pressure Deficit (VPD), exponential moisture decay, and precipitation infiltration.
- **Uncertainty Quantification**: Model-appropriate confidence estimators (tree-vote agreement for RF, Platt probability scaling for SVM, and stochastic dropout forward passes for LSTM).

"# AgriTwin-AI" 
