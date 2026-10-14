"""
AgriTwin AI - Static Simulation Report Generator (Step 6 / Backup)
------------------------------------------------------------------
Builds a single self-contained HTML file (all diagnostic plots embedded
as base64) summarizing the whole simulated pipeline.
"""
import base64
import json
import os

def b64(path):
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

with open("model_results.json") as f:
    r = json.load(f)
with open("confidence_results.json") as f:
    c = json.load(f)

imgs = {
    "preview": b64("preview_20days.png"),
    "importance": b64("feature_importance.png"),
    "validation": b64("model_validation_summary.png"),
    "calibration": b64("confidence_calibration.png"),
}

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AgriTwin AI — Software Simulation Pipeline Report</title>
<style>
  :root {{
    --primary: #1b4332;
    --primary-light: #2d6a4f;
    --accent: #52b788;
    --bg: #f8fafc;
    --card-bg: #ffffff;
    --text: #1e293b;
    --text-muted: #64748b;
    --border: #e2e8f0;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    background-color: var(--bg);
    color: var(--text);
    line-height: 1.6;
    padding: 30px 20px;
  }}
  .container {{ max-width: 1040px; margin: 0 auto; }}
  header {{
    background: linear-gradient(135deg, #1b4332 0%, #2d6a4f 100%);
    color: white;
    padding: 36px 32px;
    border-radius: 16px;
    box-shadow: 0 10px 25px rgba(27, 67, 50, 0.15);
    margin-bottom: 30px;
  }}
  header h1 {{ font-size: 2.2rem; font-weight: 800; margin-bottom: 8px; letter-spacing: -0.5px; }}
  header p {{ font-size: 1.05rem; opacity: 0.9; margin-bottom: 16px; }}
  .badge-container {{ display: flex; gap: 8px; flex-wrap: wrap; }}
  .badge {{
    background: rgba(255, 255, 255, 0.2);
    color: white;
    padding: 5px 12px;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 600;
    backdrop-filter: blur(4px);
  }}
  .section {{
    background: var(--card-bg);
    border-radius: 14px;
    border: 1px solid var(--border);
    padding: 28px;
    margin-bottom: 28px;
    box-shadow: 0 4px 6px -1px rgba(0,0,0,0.03);
  }}
  .section h2 {{
    color: var(--primary);
    font-size: 1.4rem;
    font-weight: 700;
    margin-bottom: 12px;
    border-bottom: 2px solid #e8f5e9;
    padding-bottom: 8px;
  }}
  .metric-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 16px;
    margin: 20px 0;
  }}
  .metric-card {{
    background: #f1f8f5;
    border-left: 4px solid var(--primary-light);
    border-radius: 10px;
    padding: 16px;
  }}
  .metric-card .label {{ font-size: 0.8rem; font-weight: 600; color: var(--text-muted); text-transform: uppercase; }}
  .metric-card .val {{ font-size: 1.7rem; font-weight: 800; color: var(--primary); margin-top: 4px; }}
  .chart-img {{
    width: 100%;
    border-radius: 10px;
    border: 1px solid var(--border);
    margin-top: 16px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
  }}
  .pipeline-flow {{
    background: #f8fafc;
    border: 1px dashed #cbd5e1;
    border-radius: 10px;
    padding: 18px;
    margin: 16px 0;
    text-align: center;
    font-family: monospace;
    font-size: 0.9rem;
    color: #334155;
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    margin: 16px 0;
    font-size: 0.95rem;
  }}
  th, td {{
    padding: 10px 14px;
    text-align: left;
    border-bottom: 1px solid var(--border);
  }}
  th {{ background-color: #f8fafc; color: var(--primary); font-weight: 700; }}
  .notice {{
    background-color: #fffbeb;
    border-left: 4px solid #f59e0b;
    padding: 14px 18px;
    border-radius: 8px;
    margin-bottom: 24px;
    font-size: 0.95rem;
    color: #92400e;
  }}
</style>
</head>
<body>
<div class="container">
  <header>
    <h1>🌱 AgriTwin AI — Simulation Pipeline Report</h1>
    <p>Physics-Guided Crop Digital Twin with Multi-Model AI Inferences & Monte Carlo Calibration</p>
    <div class="badge-container">
      <span class="badge">Simulation Mode</span>
      <span class="badge">90-Day Synthetic Telemetry</span>
      <span class="badge">DTSM Fusion</span>
      <span class="badge">100-Run Monte Carlo</span>
      <span class="badge">Pure NumPy LSTM</span>
    </div>
  </header>

  <div class="notice">
    <strong>📋 Presentation Backup Notice:</strong> This standalone HTML file contains genuine computed metrics from the trained models and pre-rendered diagnostic charts. For live real-time interaction and stress-injection, launch the interactive app via <code>streamlit run dashboard.py</code>.
  </div>

  <!-- Section 1 -->
  <div class="section">
    <h2>1. System Architecture & End-to-End Flow</h2>
    <div class="pipeline-flow">
      [Synthetic 5-Min Sensors] ➔ [DTSM State Fusion & Lags] ➔ [AI Multi-Model Inferences (RF/SVM/LSTM)] ➔ [Monte Carlo Uncertainty] ➔ [Prescriptive Dashboard]
    </div>
    <p>The simulation simulates correlated field variables (soil moisture, canopy temperature, rainfall, humidity, soil pH, and vapor pressure deficit) over a 90-day season at 5-minute intervals.</p>
    {"<img class='chart-img' src='data:image/png;base64," + imgs['preview'] + "'>" if imgs['preview'] else ""}
  </div>

  <!-- Section 2 -->
  <div class="section">
    <h2>2. Digital Twin State Mapping (DTSM) Feature Selection</h2>
    <p>High-frequency sensor telemetry undergoes EWMA noise filtering, multi-scale lag calculation (1h, 6h, 24h), and Gini importance ranking to extract the most predictive agronomic indicators.</p>
    {"<img class='chart-img' src='data:image/png;base64," + imgs['importance'] + "'>" if imgs['importance'] else ""}
  </div>

  <!-- Section 3 -->
  <div class="section">
    <h2>3. Multi-Model AI Validation Results</h2>
    <div class="metric-grid">
      <div class="metric-card">
        <div class="label">RF Accuracy</div>
        <div class="val">{r['random_forest']['accuracy_mean']:.1%}</div>
      </div>
      <div class="metric-card">
        <div class="label">RF Irrigation Recall</div>
        <div class="val">{r['random_forest']['recall_mean']:.1%}</div>
      </div>
      <div class="metric-card">
        <div class="label">SVM Anomaly Accuracy</div>
        <div class="val">{r['svm']['accuracy_mean']:.1%}</div>
      </div>
      <div class="metric-card">
        <div class="label">SVM Anomaly Recall</div>
        <div class="val">{r['svm']['recall_mean']:.1%}</div>
      </div>
      <div class="metric-card">
        <div class="label">LSTM 3-Day Forecast MAE</div>
        <div class="val">{r['lstm']['test_mae_soil_moisture_pct']:.2f}%</div>
      </div>
    </div>

    <table>
      <thead>
        <tr>
          <th>Model</th>
          <th>Agronomic Role</th>
          <th>Validation Methodology</th>
          <th>Primary Performance Metric</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td><strong>Random Forest</strong></td>
          <td>Irrigation Trigger Classification</td>
          <td>100-Iteration Monte Carlo Resampling (70:30)</td>
          <td>Accuracy: <strong>{r['random_forest']['accuracy_mean']:.1%}</strong> (F1: {r['random_forest']['f1_mean']:.3f})</td>
        </tr>
        <tr>
          <td><strong>Support Vector Machine (RBF)</strong></td>
          <td>Crop-Stress Anomaly Detection</td>
          <td>30-Iteration Stratified Monte Carlo</td>
          <td>Recall: <strong>{r['svm']['recall_mean']:.1%}</strong> (Precision: {r['svm']['precision_mean']:.3f})</td>
        </tr>
        <tr>
          <td><strong>Numpy LSTM</strong></td>
          <td>Short-Horizon Trend Forecast (3-Day)</td>
          <td>Backpropagation-Through-Time + Adam</td>
          <td>Test MAE: <strong>{r['lstm']['test_mae_soil_moisture_pct']:.2f}%</strong> (RMSE: {r['lstm']['test_rmse_soil_moisture_pct']:.2f}%)</td>
        </tr>
      </tbody>
    </table>
    {"<img class='chart-img' src='data:image/png;base64," + imgs['validation'] + "'>" if imgs['validation'] else ""}
  </div>

  <!-- Section 4 -->
  <div class="section">
    <h2>4. Confidence Calibration & Uncertainty Quantification</h2>
    <p>Predictions are coupled with empirical uncertainty quantification. The Random Forest demonstrates strong calibration where lowest confidence predictions (Q1) reflect noticeably lower accuracy ({c['rf_accuracy_by_confidence_quartile'].get('Q1 (lowest conf)', 0):.1%}) compared to high confidence predictions (Q4: {c['rf_accuracy_by_confidence_quartile'].get('Q4 (highest conf)', 0):.1%}).</p>
    {"<img class='chart-img' src='data:image/png;base64," + imgs['calibration'] + "'>" if imgs['calibration'] else ""}
  </div>

  <footer style="text-align: center; color: var(--text-muted); font-size: 0.85rem; margin-top: 30px; padding: 20px;">
    AgriTwin AI Simulation Module • Built with Python, NumPy, Scikit-Learn & Streamlit
  </footer>
</div>
</body>
</html>
"""

with open("AgriTwin_Simulation_Report.html", "w", encoding="utf-8") as f:
    f.write(html)

print("Saved: AgriTwin_Simulation_Report.html")

