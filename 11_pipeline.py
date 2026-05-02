# 11_pipeline.py
#
# Goal : wrap the full clustering workflow into reusable sklearn Pipelines
#
# Concepts covered:
#   - sklearn Pipeline  : chain steps so fit/transform/predict work end-to-end
#   - ColumnTransformer : apply different preprocessing to different columns
#   - Custom transformer: write your own sklearn-compatible step
#   - Pipeline + GridSearchCV : tune hyperparameters cleanly
#   - Saving / loading a pipeline with joblib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, io
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import silhouette_score
from sklearn.model_selection import ParameterGrid

# ── 1. Load data ──────────────────────────────────────────────────────────────
FILENAME = "CC_GENERAL.csv"
if not os.path.exists(FILENAME):
    print("ERROR: CC_GENERAL.csv not found. Run lesson 09 first.")
    exit(1)

df = pd.read_csv(FILENAME)
df = df.drop(columns=["CUST_ID"], errors="ignore")
print(f"Loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# ── 2. Custom transformer — feature engineering ───────────────────────────────
# A proper sklearn transformer:
#   - inherits BaseEstimator, TransformerMixin
#   - implements fit() and transform()
#   - is stateless here (no parameters to learn from data)
#   - can be dropped into any Pipeline

class CreditCardFeatureEngineer(BaseEstimator, TransformerMixin):
    """Add derived KPI columns to a credit card dataframe."""

    def fit(self, X, y=None):
        # nothing to learn — return self is required
        return self

    def transform(self, X):
        X = X.copy()
        X["PURCHASE_TO_LIMIT"]  = X["PURCHASES"]    / (X["CREDIT_LIMIT"] + 1)
        X["PAYMENT_TO_BALANCE"] = X["PAYMENTS"]     / (X["BALANCE"] + 1)
        X["CASH_TO_PURCHASES"]  = X["CASH_ADVANCE"] / (X["PURCHASES"] + 1)
        X["INSTALLMENT_RATIO"]  = (X["INSTALLMENTS_PURCHASES"]
                                    / (X["PURCHASES"] + 1))
        return X

    def get_feature_names_out(self, input_features=None):
        extra = ["PURCHASE_TO_LIMIT", "PAYMENT_TO_BALANCE",
                 "CASH_TO_PURCHASES", "INSTALLMENT_RATIO"]
        if input_features is not None:
            return list(input_features) + extra
        return extra


# ── 3. Build the pipeline ─────────────────────────────────────────────────────
# Step order:
#   engineer  → add KPI columns
#   impute    → fill missing values
#   log       → log1p transform (handle skew)
#   scale     → StandardScaler
#   pca       → reduce dimensions
#   cluster   → KMeans

FEATURES = [
    "BALANCE", "PURCHASES", "CASH_ADVANCE",
    "CREDIT_LIMIT", "PAYMENTS", "MINIMUM_PAYMENTS",
    "PRC_FULL_PAYMENT", "PURCHASES_FREQUENCY",
    "CASH_ADVANCE_FREQUENCY", "TENURE",
]
FINAL_FEATURES = FEATURES + [
    "PURCHASE_TO_LIMIT", "PAYMENT_TO_BALANCE",
    "CASH_TO_PURCHASES", "INSTALLMENT_RATIO",
]

# Named functions instead of lambdas — required for joblib pickling
def log_transform(X):
    return np.log1p(np.abs(X))

def select_features(df):
    return df[FINAL_FEATURES]

log_transformer = FunctionTransformer(func=log_transform, validate=True)

pipeline = Pipeline(steps=[
    ("engineer", CreditCardFeatureEngineer()),
    ("select",   FunctionTransformer(func=select_features, validate=False)),
    ("impute",   SimpleImputer(strategy="median")),
    ("log",      log_transformer),
    ("scale",    StandardScaler()),
    ("pca",      PCA(n_components=6, random_state=42)),
    ("cluster",  KMeans(n_clusters=4, n_init=20, random_state=42)),
])

print("\nPipeline steps:")
for name, step in pipeline.steps:
    print(f"  {name:<12} {step.__class__.__name__}")

# ── 4. Fit the pipeline ───────────────────────────────────────────────────────
pipeline.fit(df)
labels = pipeline.predict(df)

# get transformed data up to (not including) the cluster step
X_pca = pipeline[:-1].transform(df)
sil   = silhouette_score(X_pca, labels)
print(f"\nFitted.  k={len(set(labels))}  silhouette={sil:.3f}")
print(f"Cluster counts: {pd.Series(labels).value_counts().sort_index().to_dict()}")

# ── 5. Predict on new customers ───────────────────────────────────────────────
# The pipeline handles ALL preprocessing automatically
new_customers = pd.DataFrame({
    "BALANCE":                          [5000,  100,  200],
    "PURCHASES":                        [  50, 2000,    0],
    "ONEOFF_PURCHASES":                 [  50,  500,    0],
    "INSTALLMENTS_PURCHASES":           [   0, 1500,    0],
    "CASH_ADVANCE":                     [3000,    0, 1500],
    "PURCHASES_FREQUENCY":              [ 0.1,  0.9,  0.0],
    "ONEOFF_PURCHASES_FREQUENCY":       [ 0.1,  0.5,  0.0],
    "PURCHASES_INSTALLMENTS_FREQUENCY": [ 0.0,  0.7,  0.0],
    "CASH_ADVANCE_FREQUENCY":           [ 0.4,  0.0,  0.3],
    "CASH_ADVANCE_TRX":                 [   5,    0,    4],
    "PURCHASES_TRX":                    [   1,   20,    0],
    "CREDIT_LIMIT":                     [6000, 5000, 3000],
    "PAYMENTS":                         [2000, 1800,  800],
    "MINIMUM_PAYMENTS":                 [ 500,  200,  400],
    "PRC_FULL_PAYMENT":                 [ 0.0,  0.4,  0.0],
    "TENURE":                           [  12,   12,   12],
})

predicted = pipeline.predict(new_customers)
print("\nNew customer predictions:")
descriptions = [
    "High balance, cash borrower",
    "Active purchaser, partial payer",
    "Zero purchases, cash only",
]
for desc, cluster in zip(descriptions, predicted):
    print(f"  {desc:<35} → Cluster {cluster}")

# ── 6. Grid search over pipeline parameters ───────────────────────────────────
# Pipeline parameter names use the format: stepname__parametername
print("\nGrid searching over n_components and n_clusters ...")

param_grid = {
    "pca__n_components": [4, 6, 8],
    "cluster__n_clusters": [2, 3, 4, 5],
}

best_score, best_params = -1, {}
grid_rows = []

for params in ParameterGrid(param_grid):
    pipeline.set_params(**params)
    pipeline.fit(df)
    lbls  = pipeline.predict(df)
    X_tr  = pipeline[:-1].transform(df)
    score = silhouette_score(X_tr, lbls)
    grid_rows.append({**params, "silhouette": round(score, 3)})
    if score > best_score:
        best_score  = score
        best_params = params.copy()

grid_df = pd.DataFrame(grid_rows).sort_values("silhouette", ascending=False)
print("\nTop 5 parameter combinations:")
print(grid_df.head().to_string(index=False))
print(f"\nBest: {best_params}  silhouette={best_score:.3f}")

# ── 7. Refit with best params and visualise ───────────────────────────────────
pipeline.set_params(**best_params)
pipeline.fit(df)
labels    = pipeline.predict(df)
X_best    = pipeline[:-1].transform(df)

pca2d     = PCA(n_components=2, random_state=42)
X_2d      = pca2d.fit_transform(pipeline[:-2].transform(df))
var       = pca2d.explained_variance_ratio_
k         = best_params["cluster__n_clusters"]
sil_best  = silhouette_score(X_best, labels)

fig, ax = plt.subplots(figsize=(9, 6))
colors  = plt.cm.tab10(np.linspace(0, 1, k))
for i, col in enumerate(colors):
    mask = labels == i
    ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
               c=[col], s=8, alpha=0.5,
               label=f"C{i} (n={mask.sum()})")
ax.set_xlabel(f"PC1 ({var[0]:.0%})")
ax.set_ylabel(f"PC2 ({var[1]:.0%})")
ax.set_title(f"Best pipeline: pca={best_params['pca__n_components']} "
             f"k={k}  sil={sil_best:.3f}")
ax.legend(markerscale=3, fontsize=9)
plt.tight_layout()
plt.savefig("11a_pipeline_best.png", dpi=150)
plt.show()

# ── 8. Save & reload the pipeline ────────────────────────────────────────────
joblib.dump(pipeline, "clustering_pipeline.pkl")
print("\nPipeline saved to clustering_pipeline.pkl")

loaded = joblib.load("clustering_pipeline.pkl")
loaded_labels = loaded.predict(df)
assert (loaded_labels == labels).all(), "Mismatch after reload!"
print("Pipeline reloaded and verified — predictions match.")

# ── 9. Show pipeline diagram ──────────────────────────────────────────────────
profile_cols = ["BALANCE", "PURCHASES", "CASH_ADVANCE",
                "CREDIT_LIMIT", "PAYMENTS", "PRC_FULL_PAYMENT",
                "PURCHASES_FREQUENCY", "CASH_ADVANCE_FREQUENCY"]
df["Cluster"] = labels
print("\nFinal cluster profiles:")
print(df.groupby("Cluster")[profile_cols].mean().round(1).to_string())

# ── TODO tasks ────────────────────────────────────────────────────────────────
# 1. Add a new step to the pipeline: try inserting a MinMaxScaler after impute
#    instead of StandardScaler — does the silhouette change?
# 2. Add your own custom transformer that caps outliers at the 99th percentile
#    Hint: inherit BaseEstimator, TransformerMixin; store the cap in fit()
# 3. Extend the grid search to also include cluster__init: ['k-means++','random']
# 4. Use the saved pipeline: load clustering_pipeline.pkl and predict on
#    5 new made-up customers — the pipeline handles all preprocessing for you
