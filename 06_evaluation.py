# 06_evaluation.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (silhouette_score,
                              adjusted_rand_score,
                              davies_bouldin_score)

from sklearn.datasets import make_moons
X, y_true = make_moons(n_samples=300, noise=0.07, random_state=42)

# --- run all algorithms ---
models = [
    ('KMeans',       KMeans(n_clusters=3, n_init=10, random_state=42)),
    ('DBSCAN',       DBSCAN(eps=1.0, min_samples=5)),
    ('GMM',          GaussianMixture(n_components=3, random_state=42)),
    ('Hierarchical', AgglomerativeClustering(n_clusters=3, linkage='ward')),
]

results = {}
for name, model in models:
    if hasattr(model, 'fit_predict'):
        lbls = model.fit_predict(X)
    else:
        model.fit(X)
        lbls = model.predict(X)

    mask = lbls != -1
    X_v, v = X[mask], lbls[mask]
    sil = silhouette_score(X_v, v)    if len(set(v)) > 1 else float('nan')
    dbi = davies_bouldin_score(X_v, v) if len(set(v)) > 1 else float('nan')
    ari = adjusted_rand_score(y_true, lbls)
    n_c = len(set(lbls)) - (1 if -1 in lbls else 0)
    results[name] = dict(clusters=n_c,
                          silhouette=round(sil, 3),
                          ARI=round(ari, 3),
                          DBI=round(dbi, 3))

# --- print comparison table ---
print(f"\n{'Algorithm':<14} {'Clusters':>8} {'Silhouette':>11} {'ARI':>7} {'DBI':>7}")
print("-" * 52)
for name, m in results.items():
    print(f"{name:<14} {m['clusters']:>8} {m['silhouette']:>11} "
          f"{m['ARI']:>7} {m['DBI']:>7}")

# --- bar chart ---
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
names  = list(results.keys())
colors = ['#378ADD', '#BA7517', '#1D9E75', '#A32D2D']
for ax, (metric, note) in zip(axes, [
        ('silhouette', 'higher = better'),
        ('ARI',        'higher = better'),
        ('DBI',        'lower  = better')]):
    vals = [results[n][metric] for n in names]
    ax.bar(names, vals, color=colors)
    ax.set_title(f"{metric}  ({note})")
    ax.tick_params(axis='x', rotation=15)

plt.tight_layout()
plt.savefig("06_evaluation.png", dpi=150)
plt.show()

# --- TODO tasks ---
# 1. Run — read the printed table and compare all four algorithms
# 2. Change the dataset to make_moons — which algorithm wins on silhouette?
# 3. Which algorithm has the best ARI on blobs? Does that surprise you?
# 4. Add a 5th algorithm: DBSCAN with better-tuned eps — does it compete?
