# 03_dbscan.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

X, _ = make_moons(n_samples=300, noise=0.07, random_state=42)

# --- k-distance graph to guide eps choice ---
nbrs = NearestNeighbors(n_neighbors=5).fit(X)
dists, _ = nbrs.kneighbors(X)
k_dists  = np.sort(dists[:, -1])[::-1]

# --- fit DBSCAN ---
db     = DBSCAN(eps=0.2, min_samples=5)
labels = db.fit_predict(X)
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise    = (labels == -1).sum()

# --- plot ---
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

ax = axes[0]
unique = set(labels)
colors = plt.cm.tab10(np.linspace(0, 1, max(len(unique), 1)))
for lbl, col in zip(sorted(unique), colors):
    mask   = labels == lbl
    marker = 'x' if lbl == -1 else 'o'
    label  = 'Noise' if lbl == -1 else f'Cluster {lbl}'
    ax.scatter(X[mask, 0], X[mask, 1],
               c=[col], marker=marker, s=25,
               alpha=0.7, label=label)
ax.set_title(f"DBSCAN  |  {n_clusters} clusters, {n_noise} noise pts")
ax.legend(fontsize=8)

axes[1].plot(range(len(k_dists)), k_dists)
axes[1].set_title("K-distance graph  (look for the elbow)")
axes[1].set_xlabel("Points sorted by distance")
axes[1].set_ylabel("5-NN distance")

plt.tight_layout()
plt.savefig("03_dbscan.png", dpi=150)
plt.show()
print(f"Clusters found : {n_clusters}  |  Noise points : {n_noise}")

# --- TODO tasks ---
# 1. Run on make_moons — DBSCAN should find both arcs perfectly
# 2. Increase eps to 0.5 — what happens to the number of clusters?
# 3. Decrease min_samples to 2 — does noise increase or decrease?
# 4. Try it on make_blobs — compare result to k-means from lesson 02
