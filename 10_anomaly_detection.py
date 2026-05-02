# 10_anomaly_detection.py
#
# Goal : find anomalous / potentially fraudulent customers in CC_GENERAL
# Algorithms : Isolation Forest, Local Outlier Factor, Elliptic Envelope
# We reuse the same preprocessing pipeline as lesson 09

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io, os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay)

# ── 1. Load & preprocess (identical to lesson 09) ────────────────────────────
FILENAME = "CC_GENERAL.csv"
if not os.path.exists(FILENAME):
    print(f"ERROR: {FILENAME} not found. Run lesson 09 first.")
    exit(1)

df      = pd.read_csv(FILENAME)
df      = df.drop(columns=["CUST_ID"], errors="ignore")
imputer = SimpleImputer(strategy="median")
df_imp  = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# engineered KPIs
df_imp["PURCHASE_TO_LIMIT"]  = df_imp["PURCHASES"]    / (df_imp["CREDIT_LIMIT"] + 1)
df_imp["PAYMENT_TO_BALANCE"] = df_imp["PAYMENTS"]     / (df_imp["BALANCE"] + 1)
df_imp["CASH_TO_PURCHASES"]  = df_imp["CASH_ADVANCE"] / (df_imp["PURCHASES"] + 1)
df_imp["INSTALLMENT_RATIO"]  = (df_imp["INSTALLMENTS_PURCHASES"]
                                 / (df_imp["PURCHASES"] + 1))

features = [
    "BALANCE", "PURCHASES", "CASH_ADVANCE",
    "CREDIT_LIMIT", "PAYMENTS", "MINIMUM_PAYMENTS",
    "PRC_FULL_PAYMENT", "PURCHASES_FREQUENCY",
    "CASH_ADVANCE_FREQUENCY", "TENURE",
    "PURCHASE_TO_LIMIT", "PAYMENT_TO_BALANCE",
    "CASH_TO_PURCHASES", "INSTALLMENT_RATIO",
]
X        = df_imp[features].values
X_log    = np.log1p(np.abs(X))
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X_log)

pca2d = PCA(n_components=2, random_state=42)
X_2d  = pca2d.fit_transform(X_scaled)
var   = pca2d.explained_variance_ratio_
print(f"Data loaded: {X_scaled.shape[0]} customers, {X_scaled.shape[1]} features")

# ── 2. Anomaly detection — three algorithms ───────────────────────────────────
# contamination = expected fraction of outliers (we assume ~5%)
CONTAMINATION = 0.05

detectors = {
    "Isolation Forest": IsolationForest(
        n_estimators=200,
        contamination=CONTAMINATION,
        random_state=42
    ),
    "Local Outlier Factor": LocalOutlierFactor(
        n_neighbors=20,
        contamination=CONTAMINATION
    ),
    "Elliptic Envelope": EllipticEnvelope(
        contamination=CONTAMINATION,
        random_state=42
    ),
}

results = {}
for name, model in detectors.items():
    if hasattr(model, "fit_predict"):
        raw = model.fit_predict(X_scaled)   # returns +1 (normal) / -1 (anomaly)
    else:
        model.fit(X_scaled)
        raw = model.predict(X_scaled)
    # convert to boolean: True = anomaly
    is_anomaly = raw == -1
    n_anomalies = is_anomaly.sum()
    results[name] = is_anomaly
    print(f"{name:<25}  anomalies={n_anomalies}  ({n_anomalies/len(raw)*100:.1f}%)")

# ── 3. Consensus — flag customers marked anomalous by 2+ detectors ───────────
votes = (results["Isolation Forest"].astype(int)
       + results["Local Outlier Factor"].astype(int)
       + results["Elliptic Envelope"].astype(int))

consensus = votes >= 2
print(f"\nConsensus anomalies (flagged by 2+ detectors): {consensus.sum()}"
      f"  ({consensus.sum()/len(votes)*100:.1f}%)")
print(f"Flagged by all 3 detectors                  : {(votes==3).sum()}"
      f"  ({(votes==3).sum()/len(votes)*100:.1f}%)")

# ── 4. Plot — all 3 detectors side by side ───────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(20, 5))

for ax, (name, is_anom) in zip(axes[:3], results.items()):
    ax.scatter(X_2d[~is_anom, 0], X_2d[~is_anom, 1],
               c='#B5D4F4', s=5, alpha=0.4, label='Normal')
    ax.scatter(X_2d[is_anom, 0],  X_2d[is_anom, 1],
               c='#E24B4A', s=20, alpha=0.8, label=f'Anomaly ({is_anom.sum()})')
    ax.set_title(name, fontsize=10)
    ax.set_xlabel(f"PC1 ({var[0]:.0%})")
    ax.set_ylabel(f"PC2 ({var[1]:.0%})")
    ax.legend(fontsize=7, markerscale=2)

# consensus plot
ax = axes[3]
colors_map = {0: '#B5D4F4', 1: '#FAC775', 2: '#D85A30', 3: '#A32D2D'}
labels_map = {0: 'Normal (0)', 1: 'Mild (1)', 2: 'Likely (2)', 3: 'High risk (3)'}
for v in [0, 1, 2, 3]:
    mask = votes == v
    ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
               c=colors_map[v], s=5 if v == 0 else 25,
               alpha=0.4 if v == 0 else 0.9,
               label=f"{labels_map[v]} n={mask.sum()}")
ax.set_title("Consensus (vote count)", fontsize=10)
ax.set_xlabel(f"PC1 ({var[0]:.0%})")
ax.set_ylabel(f"PC2 ({var[1]:.0%})")
ax.legend(fontsize=7, markerscale=2)

plt.suptitle("Anomaly detection on credit card customers", y=1.01)
plt.tight_layout()
plt.savefig("10a_anomalies.png", dpi=150)
plt.show()

# ── 5. Profile the anomalies ──────────────────────────────────────────────────
df_imp["anomaly_votes"] = votes
df_imp["is_anomaly"]    = consensus

profile_cols = ["BALANCE", "PURCHASES", "CASH_ADVANCE",
                "CREDIT_LIMIT", "PAYMENTS", "MINIMUM_PAYMENTS",
                "PRC_FULL_PAYMENT", "PURCHASES_FREQUENCY",
                "CASH_ADVANCE_FREQUENCY"]

print("\nProfile — normal customers vs consensus anomalies:")
print(df_imp.groupby("is_anomaly")[profile_cols].mean().round(1).to_string())

# ── 6. Isolation Forest anomaly score distribution ────────────────────────────
iso = IsolationForest(n_estimators=200, contamination=CONTAMINATION,
                      random_state=42)
iso.fit(X_scaled)
scores = iso.decision_function(X_scaled)   # more negative = more anomalous

fig, axes = plt.subplots(1, 2, figsize=(13, 4))

axes[0].hist(scores, bins=60, color='#85B7EB', edgecolor='white')
axes[0].axvline(x=np.percentile(scores, 5), color='#E24B4A',
                linestyle='--', label='5th percentile threshold')
axes[0].set_xlabel("Anomaly score  (lower = more anomalous)")
axes[0].set_ylabel("Count")
axes[0].set_title("Isolation Forest score distribution")
axes[0].legend()

sc = axes[1].scatter(X_2d[:, 0], X_2d[:, 1],
                     c=scores, cmap='RdYlBu', s=5, alpha=0.6)
plt.colorbar(sc, ax=axes[1], label="Anomaly score")
axes[1].set_title("Anomaly score in PCA space\n(red = most anomalous)")
axes[1].set_xlabel(f"PC1 ({var[0]:.0%})")
axes[1].set_ylabel(f"PC2 ({var[1]:.0%})")

plt.tight_layout()
plt.savefig("10b_scores.png", dpi=150)
plt.show()

# ── 7. Top 20 most anomalous customers ────────────────────────────────────────
df_imp["iso_score"] = scores
top20 = df_imp.nsmallest(20, "iso_score")[
    ["BALANCE", "PURCHASES", "CASH_ADVANCE", "CREDIT_LIMIT",
     "PAYMENTS", "MINIMUM_PAYMENTS", "PRC_FULL_PAYMENT", "iso_score"]
].round(1)
print("\nTop 20 most anomalous customers (Isolation Forest):")
print(top20.to_string())

# ── TODO tasks ────────────────────────────────────────────────────────────────
# 1. Read the profile table — how do anomalies differ from normal customers?
#    Are they high balance? High cash advance? Unusual payment patterns?
# 2. Change CONTAMINATION to 0.01 and 0.10 — how does the count change?
#    Which threshold makes more business sense for fraud detection?
# 3. Look at the top 20 anomalous customers — what do they have in common?
# 4. Try n_neighbors=5 vs n_neighbors=50 in LOF — how does it affect results?
