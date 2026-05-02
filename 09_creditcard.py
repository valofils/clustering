# 09_creditcard.py
#
# Dataset : CC_GENERAL — 9000 credit card holders, 18 behavioural variables
# Source  : Kaggle / multiple public mirrors
# Goal    : Segment customers by spending, cash advance, and payment behaviour

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

# ── 1. Load dataset ───────────────────────────────────────────────────────────
# Place CC_GENERAL.csv in the same folder as this script
FILENAME = "CC_GENERAL.csv"
if not os.path.exists(FILENAME):
    print(f"ERROR: {FILENAME} not found in current folder.")
    print("Download it from: https://www.kaggle.com/datasets/arjunbhasin2013/ccdata")
    exit(1)
print(f"Loading {FILENAME} ...")

# ── 2. Load & inspect ─────────────────────────────────────────────────────────
df = pd.read_csv(FILENAME)
print(f"\nShape : {df.shape}")
print(f"\nColumns:\n{df.columns.tolist()}")
print(f"\nMissing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
print(f"\nSample:\n{df.head(3).to_string()}")

# ── 3. Feature engineering — derived KPIs ────────────────────────────────────
df = df.drop(columns=["CUST_ID"], errors="ignore")

# fill missing values (MINIMUM_PAYMENTS and CREDIT_LIMIT have some NaN)
imputer = SimpleImputer(strategy="median")
df_imp  = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# engineer meaningful ratios
df_imp["PURCHASE_TO_LIMIT"]   = df_imp["PURCHASES"]      / (df_imp["CREDIT_LIMIT"] + 1)
df_imp["PAYMENT_TO_BALANCE"]  = df_imp["PAYMENTS"]       / (df_imp["BALANCE"] + 1)
df_imp["CASH_TO_PURCHASES"]   = df_imp["CASH_ADVANCE"]   / (df_imp["PURCHASES"] + 1)
df_imp["INSTALLMENT_RATIO"]   = (df_imp["INSTALLMENTS_PURCHASES"]
                                  / (df_imp["PURCHASES"] + 1))

# select final features
features = [
    "BALANCE", "PURCHASES", "CASH_ADVANCE",
    "CREDIT_LIMIT", "PAYMENTS", "MINIMUM_PAYMENTS",
    "PRC_FULL_PAYMENT", "PURCHASES_FREQUENCY",
    "CASH_ADVANCE_FREQUENCY", "TENURE",
    "PURCHASE_TO_LIMIT", "PAYMENT_TO_BALANCE",
    "CASH_TO_PURCHASES", "INSTALLMENT_RATIO",
]
X = df_imp[features].values
print(f"\nFeatures used ({len(features)}): {features}")

# ── 4. Scale + log-transform skewed features ─────────────────────────────────
# Credit card data is heavily right-skewed — log helps clustering
X_log = np.log1p(np.abs(X))   # log1p handles zeros safely
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X_log)
print(f"\nScaled shape : {X_scaled.shape}")

# ── 5. PCA for visualisation and dimensionality reduction ────────────────────
pca_full = PCA(random_state=42).fit(X_scaled)
cumvar   = np.cumsum(pca_full.explained_variance_ratio_)
n_comp   = np.argmax(cumvar >= 0.85) + 1
print(f"\nComponents to explain 85% variance: {n_comp}")

pca   = PCA(n_components=n_comp, random_state=42)
X_pca = pca.fit_transform(X_scaled)

pca2d   = PCA(n_components=2, random_state=42)
X_2d    = pca2d.fit_transform(X_scaled)
var2d   = pca2d.explained_variance_ratio_

# ── 6. Choose k with elbow + silhouette (on PCA-reduced data) ────────────────
inertias, sil_scores = [], []
ks = range(2, 10)
for k in ks:
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    km.fit(X_pca)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_pca, km.labels_))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(ks, inertias, 'o-')
axes[0].set_xlabel("k"); axes[0].set_ylabel("Inertia")
axes[0].set_title("Elbow curve  (PCA-reduced data)")

axes[1].plot(ks, sil_scores, 'o-', color='tab:orange')
axes[1].set_xlabel("k"); axes[1].set_ylabel("Silhouette")
axes[1].set_title("Silhouette score — pick the peak")

plt.tight_layout()
plt.savefig("09a_choosing_k.png", dpi=150)
plt.show()

best_k = list(ks)[np.argmax(sil_scores)]
print(f"\nBest k by silhouette: {best_k}")

# ── 7. Fit KMeans with best k ─────────────────────────────────────────────────
km_best = KMeans(n_clusters=best_k, n_init=20, random_state=42)
labels  = km_best.fit_predict(X_pca)
sil     = silhouette_score(X_pca, labels)
print(f"KMeans silhouette at k={best_k}: {sil:.3f}")

# ── 8. Visualise clusters in 2D ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 6))
colors = plt.cm.tab10(np.linspace(0, 1, best_k))
for i, col in enumerate(colors):
    mask = labels == i
    ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
               c=[col], s=8, alpha=0.5, label=f"C{i} (n={mask.sum()})")
ax.set_xlabel(f"PC1 ({var2d[0]:.0%})")
ax.set_ylabel(f"PC2 ({var2d[1]:.0%})")
ax.set_title(f"Credit card customers — KMeans k={best_k}  (sil={sil:.3f})")
ax.legend(markerscale=3, fontsize=9)
plt.tight_layout()
plt.savefig("09b_clusters_2d.png", dpi=150)
plt.show()

# ── 9. Cluster profiles ───────────────────────────────────────────────────────
df_imp["Cluster"] = labels
profile_cols = ["BALANCE", "PURCHASES", "CASH_ADVANCE",
                "CREDIT_LIMIT", "PAYMENTS", "PRC_FULL_PAYMENT",
                "PURCHASES_FREQUENCY", "CASH_ADVANCE_FREQUENCY"]
profile = df_imp.groupby("Cluster")[profile_cols].mean().round(1)
print("\nCluster profiles (KMeans — raw feature means):")
print(profile.to_string())

# ── 10. Radar chart — visual profile per cluster ──────────────────────────────
# normalise each feature 0-1 for radar
profile_norm = (profile - profile.min()) / (profile.max() - profile.min() + 1e-9)
categories   = [c.replace("_", "\n") for c in profile_cols]
N = len(categories)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

fig, axes = plt.subplots(1, best_k,
                          figsize=(4 * best_k, 4),
                          subplot_kw=dict(polar=True))
if best_k == 1:
    axes = [axes]
colors_hex = ['#378ADD', '#1D9E75', '#D85A30', '#BA7517',
              '#993556', '#534AB7', '#639922', '#E24B4A']
for i, ax in enumerate(axes):
    values = profile_norm.iloc[i].tolist()
    values += values[:1]
    ax.plot(angles, values, color=colors_hex[i % len(colors_hex)], lw=2)
    ax.fill(angles, values, color=colors_hex[i % len(colors_hex)], alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=7)
    ax.set_yticks([])
    n = (labels == i).sum()
    ax.set_title(f"Cluster {i}\n(n={n})", size=10, pad=12)

plt.suptitle("Cluster radar profiles", y=1.02)
plt.tight_layout()
plt.savefig("09c_radar.png", dpi=150)
plt.show()

# ── TODO tasks ────────────────────────────────────────────────────────────────
# 1. Read the profile table — name each cluster:
#    high CASH_ADVANCE + low PURCHASES = "cash borrowers"
#    high PURCHASES + high PRC_FULL_PAYMENT = "responsible big spenders"
#    low everything = "inactive customers"
# 2. Try k=4 and k=6 — do extra clusters split a meaningful group or fragment noise?
# 3. Remove the engineered KPIs (last 4 features) and re-run — do clusters change?
# 4. Add AgglomerativeClustering with the same k and compare silhouette scores

# --- force k=4 to find sub-segments ---
km4    = KMeans(n_clusters=4, n_init=20, random_state=42)
lbls4  = km4.fit_predict(X_pca)
sil4   = silhouette_score(X_pca, lbls4)

df_imp["Cluster4"] = lbls4
profile4 = df_imp.groupby("Cluster4")[profile_cols].mean().round(1)
print(f"\nk=4 silhouette: {sil4:.3f}")
print("\nCluster profiles (k=4):")
print(profile4.to_string())

fig, ax = plt.subplots(figsize=(9, 6))
colors4 = plt.cm.tab10(np.linspace(0, 1, 4))
for i, col in enumerate(colors4):
    mask = lbls4 == i
    ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
               c=[col], s=8, alpha=0.5, label=f"C{i} (n={mask.sum()})")
ax.set_xlabel(f"PC1 ({var2d[0]:.0%})")
ax.set_ylabel(f"PC2 ({var2d[1]:.0%})")
ax.set_title(f"Credit card — KMeans k=4  (sil={sil4:.3f})")
ax.legend(markerscale=3, fontsize=9)
plt.tight_layout()
plt.savefig("09d_clusters_k4.png", dpi=150)
plt.show()