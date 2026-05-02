# 02_kmeans.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

X, _ = make_blobs(n_samples=300, centers=3,
                   cluster_std=0.8, random_state=42)

# --- fit k-means ---
km = KMeans(n_clusters=3, init='k-means++',
            n_init=10, random_state=42)
km.fit(X)
labels    = km.labels_
centroids = km.cluster_centers_

# --- plot assignments ---
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

ax = axes[0]
ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='tab10',
           s=25, alpha=0.7)
ax.scatter(centroids[:, 0], centroids[:, 1],
           marker='X', s=200, c='black', zorder=5,
           label='Centroids')
ax.set_title("K-means  (k=3)")
ax.legend()

# --- elbow method ---
inertias = []
ks = range(1, 10)
for k in ks:
    inertias.append(
        KMeans(n_clusters=k, n_init=10, random_state=42).fit(X).inertia_
    )

axes[1].plot(ks, inertias, 'o-')
axes[1].set_xlabel("k")
axes[1].set_ylabel("Inertia")
axes[1].set_title("Elbow curve — choose k at the bend")

plt.tight_layout()
plt.savefig("02_kmeans.png", dpi=150)
plt.show()
print(f"Inertia at k=3 : {km.inertia_:.1f}")

# --- TODO tasks ---
# 1. Run the script — you will see the cluster plot and elbow curve
# 2. Change n_clusters to 2 and 5 — what happens to inertia?
# 3. Try init='random' instead of 'k-means++' — run 5 times, notice variance
# 4. Apply the same code to X_moons — does k-means work well? Why not?
