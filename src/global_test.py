import joblib
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
from collections import Counter
from pathlib import Path
from datetime import datetime

results_dir = Path(f"results/{datetime.now().strftime('%Y%m%d_%H%M%S')}")
results_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(results_dir / "drift_test.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("DRIFT_TEST")

from drift_detect import DriftDetector

saved = joblib.load(r"C:\Users\RMSTVNMFST\mahesh\drift-IDS\src\models\drift_detectors\drift_detector.joblib")
scaler = joblib.load(r"C:\Users\RMSTVNMFST\mahesh\drift-IDS\src\models\trained\scaler.joblib")

detector = DriftDetector(saved["config"], logger=logger)
detector.reference_data = saved["reference_data"]
detector.feature_names = saved["feature_names"]

df = pd.read_csv(r"C:\Users\RMSTVNMFST\mahesh\drift-IDS\src\results\20260127_125638\synthetic_drift_data.csv")

X = df[detector.feature_names].values
X_scaled = scaler.transform(X)

results = detector.detect_drift_streaming(X_scaled, n_batches=50)

summary = detector.get_drift_summary()

report = pd.DataFrame({
    "batch": range(len(detector.drift_history)),
    "psi_magnitude": [d["drift_magnitude"] for d in detector.drift_history],
    "ks_drift": [d["ks_drift"] for d in detector.drift_history],
    "psi_drift": [d["psi_drift"] for d in detector.drift_history],
    "num_drift_features": [len(d["drift_features"]) for d in detector.drift_history]
})

report.to_csv(results_dir / "drift_report.csv", index=False)
df.to_csv(results_dir / "synthetic_drift_data.csv", index=False)

psi_vals = report["psi_magnitude"]
ks_flags = report["ks_drift"]
psi_flags = report["psi_drift"]

plt.figure()
plt.plot(psi_vals)
plt.axhline(detector.drift_threshold_psi, linestyle="--")
plt.title("PSI Drift Magnitude")
plt.xlabel("Batch")
plt.ylabel("PSI")
plt.savefig(results_dir / "psi_over_time.png")
plt.close()

plt.figure()
plt.plot(ks_flags)
plt.title("KS Drift Detection")
plt.xlabel("Batch")
plt.ylabel("Drift Detected")
plt.savefig(results_dir / "ks_over_time.png")
plt.close()

plt.figure()
plt.plot(ks_flags, label="KS")
plt.plot(psi_flags, label="PSI")
plt.legend()
plt.title("Drift Detection Flags")
plt.savefig(results_dir / "drift_flags.png")
plt.close()

all_feats = []
for d in detector.drift_history:
    all_feats.extend(d["drift_features"])

counter = Counter(all_feats)
top = counter.most_common(10)

if top:
    features, counts = zip(*top)
    plt.figure()
    plt.bar(features, counts)
    plt.xticks(rotation=45)
    plt.title("Most Frequently Drifted Features")
    plt.savefig(results_dir / "top_features.png")
    plt.close()
