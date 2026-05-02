# 01_datasets.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons, make_circles

# --- generate datasets ---
X_blobs, y_blobs = make_blobs(n_samples=300, centers=3,
                               cluster_std=0.8, random_state=99)
X_moons, _       = make_moons(n_samples=300, noise=0.07,
                               random_state=99)
X_circles, _     = make_circles(n_samples=300, noise=0.05,
                                 factor=0.5, random_state=99)

# --- plot ---
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
datasets  = [(X_blobs,   "Blobs (3 centres)"),
             (X_moons,   "Two moons"),
             (X_circles, "Concentric circles")]

for ax, (X, title) in zip(axes, datasets):
    ax.scatter(X[:, 0], X[:, 1], s=20, alpha=0.7)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.savefig("01_datasets.png", dpi=150)
plt.show()
print("Saved 01_datasets.png")

# --- TODO tasks ---
# 1. Run the script — three scatter plots should appear
# 2. Change n_samples to 300 and re-run
# 3. Add a 4th dataset: make_circles with noise=0.05
# 4. Try random_state=0 vs random_state=99 — how different do blobs look?
