# 04_gmm.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture

X, _ = make_blobs(n_samples=300, centers=3,
                   cluster_std=0.9, random_state=42)

# --- fit GMM ---
gmm   = GaussianMixture(n_components=3, covariance_type='full',
                         random_state=42)
gmm.fit(X)
labels = gmm.predict(X)
proba  = gmm.predict_proba(X)   # shape (n, 3) — soft assignments

# --- BIC across component counts ---
bic_scores = [
    GaussianMixture(n_components=k, random_state=42).fit(X).bic(X)
    for k in range(1, 9)
]

# --- helper: draw confidence ellipse ---
def draw_ellipse(pos, cov, ax, n_std=2, **kwargs):
    vals, vecs = np.linalg.eigh(cov)
    angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    w, h  = 2 * n_std * np.sqrt(vals)
    ax.add_patch(Ellipse(pos, w, h, angle=angle,
                         fill=False, lw=1.5, **kwargs))

# --- plot ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='tab10',
           s=20, alpha=0.6)
ax.scatter(gmm.means_[:, 0], gmm.means_[:, 1],
           marker='X', s=200, c='black', zorder=5)

colors = ['tab:blue', 'tab:orange', 'tab:green']
for i, (mean, cov) in enumerate(zip(gmm.means_, gmm.covariances_)):
    draw_ellipse(mean, cov, ax, color=colors[i], alpha=0.7)
ax.set_title("GMM  (full covariance)")

axes[1].plot(range(1, 9), bic_scores, 'o-')
axes[1].set_xlabel("n_components")
axes[1].set_ylabel("BIC")
axes[1].set_title("BIC — lower is better")

plt.tight_layout()
plt.savefig("04_gmm.png", dpi=150)
plt.show()
print(f"Soft probabilities for first point: {proba[0].round(3)}")

# --- TODO tasks ---
# 1. Run the script — notice the elliptical confidence regions
# 2. Print predict_proba(X[:5]) — each row should sum to 1.0
# 3. Change covariance_type to 'spherical' or 'diag' — how do ellipses change?
# 4. Use the BIC plot to verify n_components=3 is the best choice
