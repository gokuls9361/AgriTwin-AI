"""
AgriTwin AI - Master Pipeline Orchestrator (run_pipeline.py)
------------------------------------------------------------
Executes the entire end-to-end simulation pipeline sequentially:
  1. generate_data.py          (Synthetic sensor generation + 20-day preview)
  2. feature_engineering.py    (DTSM feature fusion + Gini importance)
  3. train_models.py           (RF, SVM, pure NumPy LSTM + MC validation)
  4. confidence_scoring.py     (Uncertainty quantification + calibration)
  5. build_static_report.py    (Self-contained HTML backup report)

Usage:
  python run_pipeline.py
"""

import os
import subprocess
import sys
import time

# Ensure stdout handles UTF-8 on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

STEPS = [
    ("Step 1: Synthetic Telemetry Simulation", "generate_data.py"),
    ("Step 2: Digital Twin State Mapping (DTSM)", "feature_engineering.py"),
    ("Step 3: Multi-Model AI Training & Monte Carlo Validation", "train_models.py"),
    ("Step 4: Confidence Scoring & Calibration", "confidence_scoring.py"),
    ("Step 5: Static Report Build (HTML Backup)", "build_static_report.py"),
]

def run_step(name, script):
    print("\n" + "=" * 70)
    print(f">> {name} ({script})")
    print("=" * 70)
    t0 = time.time()
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    res = subprocess.run([sys.executable, script], capture_output=False, env=env)
    elapsed = time.time() - t0
    if res.returncode != 0:
        print(f"\n[ERROR] Failed in {script} (exit code {res.returncode})")
        sys.exit(res.returncode)
    print(f"[OK] Completed in {elapsed:.2f}s")

def main():
    print("Starting AgriTwin AI Pipeline Execution...")
    start_total = time.time()
    for name, script in STEPS:
        run_step(name, script)
    total_time = time.time() - start_total
    print("\n" + "=" * 70)
    print(f"All pipeline steps completed successfully in {total_time:.2f}s!")
    print("=" * 70)
    print("\nTo launch the live interactive dashboard:")
    print("  streamlit run dashboard.py\n")

if __name__ == "__main__":
    main()

