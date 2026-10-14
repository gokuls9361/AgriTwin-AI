"""
AgriTwin AI - DTSM Feature Engineering (Step 2)
--------------------------------------------------
Digital Twin State Mapping (DTSM) feature fusion pipeline:
  - EWMA adaptive filtering (sensor noise reduction)
  - Multi-horizon lag features (1h, 6h, 24h)
  - Agronomic metrics (VPD, soil moisture depletion velocity, rolling stats)
  - Gini importance feature selection via Random Forest probe
"""

import numpy as np
import pandas as pd
import json
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("sensor_data.csv", parse_dates=["timestamp"])

# 1. EWMA smoothing
ewma_alphas = {
    "soil_moisture_pct": 0.05,
    "temperature_c": 0.15,
    "humidity_pct": 0.15,
    "soil_ph": 0.03,
    "rainfall_mm": 0.4,
}
for col, alpha in ewma_alphas.items():
    df[f"{col}_ewma"] = df[col].ewm(alpha=alpha, adjust=False).mean()

# 2. Multi-horizon lag features
lag_steps = {"1h": 12, "6h": 72, "24h": 288}
lag_cols = ["soil_moisture_pct", "temperature_c", "humidity_pct", "rainfall_mm"]
for col in lag_cols:
    for label, steps in lag_steps.items():
        df[f"{col}_lag{label}"] = df[col].shift(steps)

# 3. Dynamic Agronomic & Rate-of-change Indicators
# Depletion velocity (% moisture change per hour / 6 hours)
df["moisture_depletion_1h"] = df["soil_moisture_pct"] - df["soil_moisture_pct_lag1h"]
df["moisture_depletion_6h"] = df["soil_moisture_pct"] - df["soil_moisture_pct_lag6h"]
df["soil_moisture_roll6h_std"] = df["soil_moisture_pct"].rolling(72, min_periods=1).std().fillna(0)
df["temp_roll24h_mean"] = df["temperature_c"].rolling(288, min_periods=1).mean()
df["temp_roll6h_max"] = df["temperature_c"].rolling(72, min_periods=1).max()

# If VPD column exists in sensor_data, use it; otherwise compute it
if "vpd_kpa" not in df.columns:
    e_sat = 0.61078 * np.exp((17.27 * df["temperature_c"]) / (df["temperature_c"] + 237.3))
    df["vpd_kpa"] = e_sat * (1.0 - df["humidity_pct"] / 100.0)
df["vpd_ewma"] = df["vpd_kpa"].ewm(alpha=0.15, adjust=False).mean()

df = df.dropna().reset_index(drop=True)

feature_cols = [
    c for c in df.columns 
    if c not in ("timestamp", "irrigation_need", "anomaly_flag", "phenology_stage") 
    and np.issubdtype(df[c].dtype, np.number)
]
norm_stats = {}
df_norm = df.copy()
for c in feature_cols:
    mu, sigma = float(df[c].mean()), float(df[c].std())
    sigma = sigma if sigma > 1e-9 else 1.0
    df_norm[c] = (df[c] - mu) / sigma
    norm_stats[c] = (mu, sigma)

X = df_norm[feature_cols]
y = df["irrigation_need"]
rf_probe = RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42, n_jobs=-1)
rf_probe.fit(X, y)
importances = pd.Series(rf_probe.feature_importances_, index=feature_cols).sort_values(ascending=False)

TOP_N = 15
top_features = importances.head(TOP_N).index.tolist()
print(f"Total engineered features: {len(feature_cols)}")
print(importances.head(TOP_N).round(4))

df_out = df_norm[top_features].copy()
df_out.insert(0, "timestamp", df["timestamp"].values)
df_out["irrigation_need"] = y.values
df_out["anomaly_flag"] = df["anomaly_flag"].values
df_out["soil_moisture_pct_raw"] = df["soil_moisture_pct"].values
df_out["soil_moisture_top_pct_raw"] = df["soil_moisture_top_pct"].values if "soil_moisture_top_pct" in df.columns else df["soil_moisture_pct"].values
df_out["soil_moisture_deep_pct_raw"] = df["soil_moisture_deep_pct"].values if "soil_moisture_deep_pct" in df.columns else df["soil_moisture_pct"].values
df_out["rainfall_mm_raw"] = df["rainfall_mm"].values
df_out["temperature_c_raw"] = df["temperature_c"].values
if "vpd_kpa" in df.columns:
    df_out["vpd_kpa_raw"] = df["vpd_kpa"].values
if "cumulative_gdd" in df.columns:
    df_out["cumulative_gdd_raw"] = df["cumulative_gdd"].values
if "phenology_stage" in df.columns:
    df_out["phenology_stage_raw"] = df["phenology_stage"].values
if "ndvi_canopy" in df.columns:
    df_out["ndvi_canopy_raw"] = df["ndvi_canopy"].values
if "zone1_moisture_pct" in df.columns:
    df_out["zone1_moisture_pct_raw"] = df["zone1_moisture_pct"].values
    df_out["zone3_moisture_pct_raw"] = df["zone3_moisture_pct"].values
    df_out["zone4_moisture_pct_raw"] = df["zone4_moisture_pct"].values

df_out.to_csv("features.csv", index=False)

with open("norm_stats.json", "w") as f:
    json.dump({k: list(v) for k, v in norm_stats.items()}, f, indent=2)
with open("top_features.json", "w") as f:
    json.dump(top_features, f, indent=2)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 5.5))
colors = ["#2b5c8f" if "moisture" in feat else "#e67e22" if "temp" in feat or "vpd" in feat else "#27ae60" for feat in importances.head(TOP_N).sort_values().index]
importances.head(TOP_N).sort_values().plot(kind="barh", ax=ax, color=colors, edgecolor="#2c3e50")
ax.set_xlabel("Gini Importance (Tree Split Quality)", fontsize=10, fontweight="bold")
ax.set_title(f"Top {TOP_N} Digital Twin Features (DTSM Fusion Pipeline)", fontsize=11, fontweight="bold")
ax.grid(True, axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
plt.close()

print(f"\nSaved: features.csv ({df_out.shape[0]} rows, {len(top_features)} features)")
print("Saved: feature_importance.png")


