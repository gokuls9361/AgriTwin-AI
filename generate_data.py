"""
AgriTwin AI - Synthetic Sensor Data Generator (Step 1 - Option C Suite)
------------------------------------------------------------------------
Simulates a multi-zone agricultural digital twin over a 90-day season at
5-minute intervals. Implements agronomic physics:
  - 3-Layer Soil Hydrology: Topsoil (0-10cm), Root Zone (10-30cm), Deep Subsoil (30-60cm)
  - Crop Phenology & Growing Degree Days (GDD, Base T=10°C)
  - Synthetic Multispectral NDVI Canopy Greenness Proxy
  - Multi-Zone Spatial Heterogeneity (4 Micro-Plots)
  - Anomaly Injections & Probabilistic Irrigation Labels
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 42
rng = np.random.default_rng(SEED)

DAYS = 90
FREQ_MIN = 5
STEPS_PER_DAY = int(24 * 60 / FREQ_MIN)
N = DAYS * STEPS_PER_DAY

timestamps = pd.date_range("2026-05-01", periods=N, freq=f"{FREQ_MIN}min")
t_hours = np.arange(N) * (FREQ_MIN / 60.0)
day_of_sim = t_hours / 24.0

# ---- Rainfall: sparse stochastic storm events ----
rain_event_prob_per_step = 0.0018
rain_mm = np.zeros(N)
is_raining = rng.random(N) < rain_event_prob_per_step
i = 0
while i < N:
    if is_raining[i]:
        duration = rng.integers(2, 10)
        peak = rng.uniform(1.0, 10.0)
        for k in range(duration):
            if i + k >= N:
                break
            rain_mm[i + k] += peak * np.exp(-k / 4.0) * rng.uniform(0.6, 1.0)
        i += duration
    else:
        i += 1

# ---- Temperature ----
seasonal_drift = 3.0 * np.sin(2 * np.pi * day_of_sim / 90)
daily_cycle = 7.0 * np.sin(2 * np.pi * (t_hours % 24 - 9) / 24)
temperature = 26.0 + seasonal_drift + daily_cycle + rng.normal(0, 0.8, N)
temperature -= np.minimum(rain_mm * 0.15, 3.0)

# ---- Humidity ----
humidity = 60.0 - 0.8 * daily_cycle + rng.normal(0, 3.0, N)
humidity += np.minimum(rain_mm * 1.2, 25.0)
humidity = np.clip(humidity, 20, 98)

# ---- Vapor Pressure Deficit (VPD in kPa) ----
e_sat = 0.61078 * np.exp((17.27 * temperature) / (temperature + 237.3))
vpd = e_sat * (1.0 - humidity / 100.0)
vpd = np.clip(vpd, 0.1, 5.0)

# ---- Growing Degree Days (GDD) & Phenology ----
# T_base = 10°C for warm-season crops (e.g. maize, cotton, tomato)
gdd_rate = np.maximum(0, temperature - 10.0) / STEPS_PER_DAY
cumulative_gdd = np.cumsum(gdd_rate)

# Phenological Stage Labeling
# 0-150: Emergence, 150-500: Vegetative, 500-850: Flowering (Critical), 850-1250: Grain Filling, 1250+: Maturity
phenology_stage = np.empty(N, dtype=object)
phenology_stage[cumulative_gdd < 150] = "1_Emergence"
phenology_stage[(cumulative_gdd >= 150) & (cumulative_gdd < 500)] = "2_Vegetative"
phenology_stage[(cumulative_gdd >= 500) & (cumulative_gdd < 850)] = "3_Flowering"
phenology_stage[(cumulative_gdd >= 850) & (cumulative_gdd < 1250)] = "4_GrainFilling"
phenology_stage[cumulative_gdd >= 1250] = "5_Maturity"

# ---- Multi-Layer Soil Hydrology (0-10cm, 10-30cm, 30-60cm) ----
soil_moisture_top = np.zeros(N)     # 0-10 cm: rapid evaporation & immediate rain gain
soil_moisture_root = np.zeros(N)    # 10-30 cm: primary root extraction
soil_moisture_deep = np.zeros(N)    # 30-60 cm: percolation buffer

soil_moisture_top[0] = 38.0
soil_moisture_root[0] = 40.0
soil_moisture_deep[0] = 42.0

et_rate_top = 0.035 + 0.0010 * np.maximum(temperature - 25, 0) + 0.008 * vpd
et_rate_root = 0.020 + 0.0006 * np.maximum(temperature - 25, 0) + 0.004 * vpd
et_rate_deep = 0.005 + 0.0001 * np.maximum(temperature - 25, 0)

for k in range(1, N):
    # Topsoil layer
    rain_top = rain_mm[k] * 0.85
    soil_moisture_top[k] = soil_moisture_top[k - 1] - et_rate_top[k] + rain_top
    percolation_to_root = max(0.0, (soil_moisture_top[k] - 35.0) * 0.05)
    soil_moisture_top[k] = np.clip(soil_moisture_top[k] - percolation_to_root + rng.normal(0, 0.15), 6, 50)

    # Root zone layer (receives percolation from top, loses percolation to deep)
    soil_moisture_root[k] = soil_moisture_root[k - 1] - et_rate_root[k] + percolation_to_root
    percolation_to_deep = max(0.0, (soil_moisture_root[k] - 38.0) * 0.03)
    soil_moisture_root[k] = np.clip(soil_moisture_root[k] - percolation_to_deep + rng.normal(0, 0.10), 8, 48)

    # Deep subsoil layer
    soil_moisture_deep[k] = soil_moisture_deep[k - 1] - et_rate_deep[k] + percolation_to_deep
    soil_moisture_deep[k] = np.clip(soil_moisture_deep[k] + rng.normal(0, 0.05), 12, 45)

# Primary soil moisture for standard sensing is the root zone (10-30cm)
soil_moisture = soil_moisture_root.copy()

# ---- Synthetic Multispectral NDVI Canopy Greenness Proxy ----
# Baseline vegetative curve follows cumulative GDD curve, capped around 0.85, decaying at maturity
ndvi_base = 0.20 + 0.65 / (1.0 + np.exp(-(cumulative_gdd - 400) / 120.0))
# Senescence at end of season
ndvi_base[cumulative_gdd > 1100] -= 0.0004 * (cumulative_gdd[cumulative_gdd > 1100] - 1100)
ndvi = np.clip(ndvi_base + rng.normal(0, 0.015, N), 0.15, 0.88)

# ---- Soil pH ----
ph = 6.5 + 0.3 * np.sin(2 * np.pi * day_of_sim / 60) + rng.normal(0, 0.05, N)
ph = np.clip(ph, 5.5, 7.8)

# ---- Inject synthetic crop-stress anomalies ----
n_anomalies = 20
anomaly_flag = np.zeros(N, dtype=int)
anomaly_idxs = rng.choice(np.arange(500, N - 500), size=n_anomalies, replace=False)
for idx in anomaly_idxs:
    span = rng.integers(30, 80)
    kind = rng.choice(["moisture_crash", "heat_spike"], p=[0.75, 0.25])
    for k in range(span):
        j = idx + k
        if j >= N:
            break
        if kind == "moisture_crash":
            soil_moisture_top[j] -= rng.uniform(12, 20) * np.exp(-k / 25.0)
            soil_moisture[j] -= rng.uniform(10, 18) * np.exp(-k / 25.0)
            ndvi[j] -= 0.08 * np.exp(-k / 35.0)
        else:
            temperature[j] += rng.uniform(6, 10) * np.exp(-k / 25.0)
            ndvi[j] -= 0.05 * np.exp(-k / 35.0)
        anomaly_flag[j] = 1

soil_moisture_top = np.clip(soil_moisture_top, 5, 50)
soil_moisture = np.clip(soil_moisture, 6, 50)
ndvi = np.clip(ndvi, 0.12, 0.88)

# ---- Rolling rain-in-last-24h ----
steps_24h = STEPS_PER_DAY
rain_series = pd.Series(rain_mm)
rain_last_24h = rain_series.rolling(steps_24h, min_periods=1).sum().values

# ---- Rule-derived irrigation_need label ----
MOISTURE_THRESH = 20.0
RAIN_THRESH = 3.0

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

moisture_z = (MOISTURE_THRESH - soil_moisture) / 4.5
rain_z = (RAIN_THRESH - rain_last_24h) / 4.0
vpd_z = (vpd - 1.5) / 0.8
# Stage sensitivity: During flowering (stage 3), water threshold is more sensitive
stage_boost = np.where(phenology_stage == "3_Flowering", 0.4, 0.0)
score = 1.2 * moisture_z + 0.4 * rain_z + 0.2 * vpd_z + stage_boost
prob_irrigate = sigmoid(score)
irrigation_need = (rng.random(N) < prob_irrigate).astype(int)

# label noise: ~4% flipped
flip_mask = rng.random(N) < 0.04
irrigation_need[flip_mask] = 1 - irrigation_need[flip_mask]

# ---- Multi-Zone Spatial Micro-Plots (4 Zones) ----
# Zone 1: Sandy Loam (Drier, higher drainage, -2.5% moisture)
# Zone 2: Optimal Clay Loam (Nominal baseline)
# Zone 3: Lowland Alluvial (+3.5% moisture, higher retention)
# Zone 4: Sun-Exposed Ridge (+1.2°C temp, +0.3 kPa VPD, -1.8% moisture)
zone1_moist = np.clip(soil_moisture - 2.5 + rng.normal(0, 0.2, N), 5, 48)
zone3_moist = np.clip(soil_moisture + 3.5 + rng.normal(0, 0.2, N), 8, 52)
zone4_moist = np.clip(soil_moisture - 1.8 + rng.normal(0, 0.2, N), 5, 48)

df = pd.DataFrame({
    "timestamp": timestamps,
    "temperature_c": temperature.round(2),
    "humidity_pct": humidity.round(2),
    "vpd_kpa": vpd.round(3),
    "cumulative_gdd": cumulative_gdd.round(1),
    "phenology_stage": phenology_stage,
    "ndvi_canopy": ndvi.round(3),
    "soil_moisture_pct": soil_moisture.round(2),
    "soil_moisture_top_pct": soil_moisture_top.round(2),
    "soil_moisture_deep_pct": soil_moisture_deep.round(2),
    "zone1_moisture_pct": zone1_moist.round(2),
    "zone3_moisture_pct": zone3_moist.round(2),
    "zone4_moisture_pct": zone4_moist.round(2),
    "rainfall_mm": rain_mm.round(2),
    "rain_last_24h_mm": rain_last_24h.round(2),
    "soil_ph": ph.round(2),
    "anomaly_flag": anomaly_flag,
    "irrigation_need": irrigation_need,
})

df.to_csv("sensor_data.csv", index=False)
print(f"Generated {len(df)} rows over {DAYS} days -> sensor_data.csv")
print(f"Irrigation-need positive rate: {df['irrigation_need'].mean():.2%}")
print(f"Injected anomaly steps: {df['anomaly_flag'].sum()} across {n_anomalies} events")

# ---- Generate 20-day preview plot ----
preview_n = 20 * STEPS_PER_DAY
sub_df = df.iloc[:preview_n]

fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(11, 9.0), sharex=True)

# 1. Multi-Depth Soil Moisture
ax1.plot(sub_df["timestamp"], sub_df["soil_moisture_top_pct"], color="#a0522d", lw=1.0, linestyle=":", label="Topsoil (0-10cm)")
ax1.plot(sub_df["timestamp"], sub_df["soil_moisture_pct"], color="#2d6a4f", lw=1.6, label="Root Zone (10-30cm)")
ax1.plot(sub_df["timestamp"], sub_df["soil_moisture_deep_pct"], color="#1b4332", lw=1.2, linestyle="--", label="Deep Subsoil (30-60cm)")
anom_pts = sub_df[sub_df["anomaly_flag"] == 1]
if len(anom_pts):
    ax1.scatter(anom_pts["timestamp"], anom_pts["soil_moisture_pct"], color="red", s=14, zorder=5, label="Stress Anomaly")
ax1.axhline(MOISTURE_THRESH, color="#d9534f", linestyle="--", lw=1, alpha=0.7, label=f"Threshold ({MOISTURE_THRESH}%)")
ax1.set_ylabel("Moisture (%)", fontsize=9.5)
ax1.set_title("AgriTwin AI: 20-Day Multi-Layer Telemetry, GDD Phenology & NDVI Greenness", fontsize=11, fontweight="bold")
ax1.legend(loc="upper right", fontsize=8, ncol=2)
ax1.grid(True, alpha=0.3)

# 2. Temperature & VPD
ax2.plot(sub_df["timestamp"], sub_df["temperature_c"], color="#e67e22", lw=1.2, label="Canopy Temp (°C)")
ax2.set_ylabel("Temp (°C)", fontsize=9.5)
ax2_twin = ax2.twinx()
ax2_twin.plot(sub_df["timestamp"], sub_df["vpd_kpa"], color="#8e44ad", lw=1.0, linestyle=":", label="VPD (kPa)")
ax2_twin.set_ylabel("VPD (kPa)", fontsize=9.5)
ax2.legend(loc="upper left", fontsize=8)
ax2_twin.legend(loc="upper right", fontsize=8)
ax2.grid(True, alpha=0.3)

# 3. NDVI Greenness & Cumulative GDD
ax3.plot(sub_df["timestamp"], sub_df["ndvi_canopy"], color="#27ae60", lw=1.5, label="Canopy NDVI (Greenness)")
ax3.set_ylabel("NDVI (0-1)", fontsize=9.5)
ax3_twin = ax3.twinx()
ax3_twin.plot(sub_df["timestamp"], sub_df["cumulative_gdd"], color="#d35400", lw=1.2, linestyle="--", label="Cumulative GDD (°C-day)")
ax3_twin.set_ylabel("GDD", fontsize=9.5)
ax3.legend(loc="upper left", fontsize=8)
ax3_twin.legend(loc="upper right", fontsize=8)
ax3.grid(True, alpha=0.3)

# 4. Rainfall & Irrigation Need Trigger
ax4.bar(sub_df["timestamp"], sub_df["rainfall_mm"], color="#3498db", width=0.003, label="Rainfall (mm)")
ax4.step(sub_df["timestamp"], sub_df["irrigation_need"] * 5, color="#e74c3c", lw=1.5, where="mid", label="Irrigation Trigger (5=Active)")
ax4.set_ylabel("Rain / Irrig", fontsize=9.5)
ax4.set_xlabel("Simulated Timeline", fontsize=9.5)
ax4.legend(loc="upper right", fontsize=8)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("preview_20days.png", dpi=150)
plt.close()
print("Saved preview plot: preview_20days.png")
