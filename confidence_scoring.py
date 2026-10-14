"""
AgriTwin AI - Confidence Scoring / Uncertainty Quantification (Step 4)
---------------------------------------------------------------------------
The paper describes Monte Carlo Dropout confidence scoring across all
three models. Dropout is a neural-network-specific regularization
technique, so it only applies literally to the LSTM. For RF and SVM we
use the model-appropriate equivalent of "many stochastic samples of the
same prediction":

  - Random Forest : tree-vote agreement across the 200 trees in the
                     forest (predict_proba) -- each tree is itself a
                     bootstrap sample, so vote agreement plays the same
                     role dropout-averaging plays for a neural net.
  - SVM            : Platt-scaled predict_proba (already fit with
                      probability=True in Step 3).
  - LSTM           : true Monte Carlo Dropout -- 50 forward passes with
                      dropout active at inference, mean + std of outputs.

Predictions below a confidence threshold are flagged for farmer review,
matching Section III-D of the paper. This script also empirically checks
the claim (made in the report's Discussion section) that low-confidence
predictions carry higher error -- using the actual trained models rather
than asserting it without evidence.
"""

import json
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

with open("models.pkl", "rb") as f:
    m = pickle.load(f)
rf_final, svm_final, top_features = m["rf_final"], m["svm_final"], m["top_features"]

df = pd.read_csv("features.csv", parse_dates=["timestamp"]).iloc[::12].reset_index(drop=True)
X = df[top_features].values
y_irr = df["irrigation_need"].values
y_anom = df["anomaly_flag"].values

X_tr, X_te, y_tr, y_te = train_test_split(X, y_irr, test_size=0.30, random_state=0, stratify=y_irr)

# ============================================================
# Random Forest confidence: tree-vote agreement (predict_proba)
# ============================================================
rf_proba = rf_final.predict_proba(X_te)          # (n, 2)
rf_pred = rf_final.predict(X_te)
rf_confidence = rf_proba.max(axis=1)               # winning-class vote share
rf_correct = (rf_pred == y_te)

CONF_THRESHOLD = 0.65
rf_flagged = rf_confidence < CONF_THRESHOLD
print(f"RF: {rf_flagged.sum()}/{len(rf_flagged)} predictions flagged for farmer review "
      f"(confidence < {CONF_THRESHOLD})")

# empirical check: does low confidence correlate with higher error?
quartiles = pd.qcut(rf_confidence, 4, labels=["Q1 (lowest conf)", "Q2", "Q3", "Q4 (highest conf)"])
calib = pd.DataFrame({"quartile": quartiles, "correct": rf_correct}).groupby("quartile", observed=True)["correct"].mean()
print("\nRF accuracy by confidence quartile (empirical calibration check):")
print(calib.round(4))

# ============================================================
# SVM confidence: Platt-scaled probability
# ============================================================
X_tr_a, X_te_a, y_tr_a, y_te_a = train_test_split(X, y_anom, test_size=0.30, random_state=0, stratify=y_anom)
svm_proba = svm_final.predict_proba(X_te_a)
svm_pred = svm_final.predict(X_te_a)
svm_confidence = svm_proba.max(axis=1)
svm_flagged = svm_confidence < CONF_THRESHOLD
print(f"\nSVM: {svm_flagged.sum()}/{len(svm_flagged)} predictions flagged for farmer review")

# ============================================================
# LSTM: TRUE Monte Carlo Dropout (50 forward passes)
# ============================================================
class NumpyLSTM:
    def __init__(self, hidden_size):
        self.hidden_size = hidden_size

    @staticmethod
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

    def load(self, path):
        d = np.load(path)
        self.Wf, self.bf = d["Wf"], d["bf"]
        self.Wi, self.bi = d["Wi"], d["bi"]
        self.Wc, self.bc = d["Wc"], d["bc"]
        self.Wo, self.bo = d["Wo"], d["bo"]
        self.Wy, self.by = d["Wy"], d["by"]
        self.d_mu, self.d_sigma = d["d_mu"], d["d_sigma"]
        return self

    def forward(self, X, dropout_p=0.0, rng=None):
        batch, seq_len, _ = X.shape
        H = self.hidden_size
        h = np.zeros((batch, H)); c = np.zeros((batch, H))
        for t in range(seq_len):
            xt = X[:, t, :]
            z = np.concatenate([h, xt], axis=1)
            f = self.sigmoid(z @ self.Wf + self.bf)
            i = self.sigmoid(z @ self.Wi + self.bi)
            cbar = np.tanh(z @ self.Wc + self.bc)
            o = self.sigmoid(z @ self.Wo + self.bo)
            c = f * c + i * cbar
            h = o * np.tanh(c)
            if dropout_p > 0 and rng is not None:
                mask = (rng.random(h.shape) > dropout_p).astype(np.float64) / (1 - dropout_p)
                h = h * mask
        y = h @ self.Wy + self.by
        return y


lstm = NumpyLSTM(hidden_size=12).load("lstm_weights.npz")

# rebuild the same windowed daily sequences used in Step 3
daily = df.set_index("timestamp").resample("1D").agg({
    "soil_moisture_pct_raw": "mean", "rainfall_mm_raw": "sum", "temperature_c_raw": "mean"
}).dropna()
feat_cols = ["soil_moisture_pct_raw", "rainfall_mm_raw", "temperature_c_raw"]
daily_norm = (daily[feat_cols].values - lstm.d_mu) / lstm.d_sigma

WINDOW, HORIZON = 7, 3
Xs = np.array([daily_norm[i:i + WINDOW] for i in range(len(daily_norm) - WINDOW - HORIZON + 1)])

rng = np.random.default_rng(123)
N_PASSES = 50
mc_preds = np.zeros((N_PASSES, len(Xs)))
for p in range(N_PASSES):
    y = lstm.forward(Xs, dropout_p=0.15, rng=rng)
    mc_preds[p] = y[:, 0] * lstm.d_sigma[0] + lstm.d_mu[0]

lstm_mean = mc_preds.mean(axis=0)
lstm_std = mc_preds.std(axis=0)
print(f"\nLSTM MC-Dropout ({N_PASSES} passes): mean prediction std across samples = {lstm_std.mean():.3f}% moisture")
lstm_flagged = lstm_std > np.percentile(lstm_std, 75)  # flag top-quartile uncertainty
print(f"LSTM: {lstm_flagged.sum()}/{len(lstm_flagged)} forecasts flagged for farmer review (top-quartile uncertainty)")

# Calculate Brier Score and Expected Calibration Error (ECE) for Random Forest
brier_score = float(np.mean((rf_proba[:, 1] - y_te) ** 2))
n_bins = 5
bin_boundaries = np.linspace(0, 1, n_bins + 1)
ece = 0.0
for b in range(n_bins):
    bin_mask = (rf_proba[:, 1] >= bin_boundaries[b]) & (rf_proba[:, 1] < bin_boundaries[b + 1])
    if bin_mask.sum() > 0:
        bin_acc = y_te[bin_mask].mean()
        bin_conf = rf_proba[:, 1][bin_mask].mean()
        ece += (bin_mask.sum() / len(y_te)) * np.abs(bin_acc - bin_conf)
ece = float(ece)
print(f"\nRF Calibration Metrics: Brier Score = {brier_score:.4f}, ECE = {ece:.4f}")

# ============================================================
# Save results + calibration chart
# ============================================================
out = {
    "confidence_threshold": CONF_THRESHOLD,
    "rf_brier_score": brier_score,
    "rf_ece": ece,
    "rf_flag_rate": float(rf_flagged.mean()),
    "svm_flag_rate": float(svm_flagged.mean()),
    "lstm_flag_rate": float(lstm_flagged.mean()),
    "rf_accuracy_by_confidence_quartile": {str(k): float(v) for k, v in calib.items()},
}
with open("confidence_results.json", "w") as f:
    json.dump(out, f, indent=2)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))

# 1. RF Accuracy by Confidence Quartile
bars = axes[0].bar(calib.index.astype(str), calib.values * 100, color="#3498db", edgecolor="#2980b9", width=0.55)
for bar in bars:
    yval = bar.get_height()
    axes[0].text(bar.get_x() + bar.get_width()/2.0, yval + 1.5, f"{yval:.1f}%", ha='center', va='bottom', fontsize=8.5, fontweight="bold")
axes[0].set_ylabel("Empirical Accuracy (%)", fontsize=10)
axes[0].set_title("Random Forest: Accuracy by Confidence Quartile", fontsize=10.5, fontweight="bold")
axes[0].tick_params(axis='x', rotation=15, labelsize=8)
axes[0].set_ylim(0, 115)
axes[0].grid(True, axis="y", alpha=0.3)

# 2. LSTM Forecast Uncertainty (MC-Dropout Std)
scatter = axes[1].scatter(np.arange(len(lstm_std)), lstm_std, s=24, c=lstm_std, cmap="viridis", edgecolors="#2c3e50", alpha=0.85)
threshold_val = float(np.percentile(lstm_std, 75))
axes[1].axhline(threshold_val, color="#e74c3c", ls="--", lw=1.8, label=f"75th %ile Threshold ({threshold_val:.2f}%)")
axes[1].set_xlabel("Daily Forecast Sequence Index", fontsize=10)
axes[1].set_ylabel("MC-Dropout Std (Moisture %)", fontsize=10)
axes[1].set_title("LSTM: Monte Carlo Dropout Uncertainty Distribution", fontsize=10.5, fontweight="bold")
axes[1].legend(fontsize=8, loc="upper right")
axes[1].grid(True, alpha=0.3)
cbar = plt.colorbar(scatter, ax=axes[1])
cbar.set_label("Std (%)", fontsize=8)

plt.tight_layout()
plt.savefig("confidence_calibration.png", dpi=150)
plt.close()
print("Saved: confidence_results.json, confidence_calibration.png")
