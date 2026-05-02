# 05_hierarchical.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

X, _ = make_blobs(n_samples=100, centers=3,
                   cluster_std=0.8, random_state=42)

# --- build full linkage tree (scipy) for dendrogram ---
Z = linkage(X, method='ward')

# --- sklearn agglomerative clustering ---
agg    = AgglomerativeClustering(n_clusters=3, linkage='ward')
labels = agg.fit_predict(X)

# --- plot ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].scatter(X[:, 0], X[:, 1], c=labels,
                cmap='tab10', s=30, alpha=0.8)
axes[0].set_title("Agglomerative  (ward, k=3)")

dendrogram(Z, ax=axes[1], truncate_mode='lastp',
           p=20, leaf_rotation=45,
           color_threshold=0.7 * max(Z[:, 2]))
axes[1].set_title("Dendrogram  (cut line = 3 clusters)")
axes[1].set_xlabel("Sample index / cluster size")
axes[1].set_ylabel("Distance")

plt.tight_layout()
plt.savefig("05_hierarchical.png", dpi=150)
plt.show()
print(f"Unique clusters found: {len(set(labels))}")

# --- TODO tasks ---
# 1. Run the script — read the dendrogram top-down to count natural clusters
# 2. Change linkage to 'single', 'complete', then 'average' — compare shapes
# 3. Where would you cut the dendrogram to get 2 clusters? And 4?
# 4. Try on make_moons — which linkage handles curved shapes best?
