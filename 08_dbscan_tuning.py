# 08_dbscan_tuning.py
#
# Goal: find the best (eps, min_samples) for DBSCAN on mall customers
# Strategy:
#   1. K-distance graph to narrow down eps range
#   2. Grid search over (eps, min_samples)
#   3. Visualise the best result and compare to k-means

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# ── 1. Data (same as lesson 07) ───────────────────────────────────────────────
RAW = """CustomerID,Gender,Age,Annual Income (k$),Spending Score (1-100)
1,Male,19,15,39
2,Male,21,15,81
3,Female,20,16,6
4,Female,23,16,77
5,Female,31,17,40
6,Female,22,17,76
7,Female,35,18,6
8,Female,23,18,94
9,Male,64,19,3
10,Female,30,19,72
11,Male,67,19,14
12,Female,35,19,99
13,Female,58,20,15
14,Female,24,20,77
15,Male,37,20,13
16,Male,22,20,79
17,Female,35,21,35
18,Male,20,21,66
19,Male,52,23,29
20,Female,35,23,98
21,Male,35,24,35
22,Male,25,24,73
23,Female,46,25,5
24,Male,31,25,73
25,Female,54,28,14
26,Male,29,28,82
27,Female,45,28,32
28,Male,35,28,61
29,Female,40,29,31
30,Female,23,29,87
31,Male,60,30,4
32,Female,21,30,73
33,Male,53,33,4
34,Male,18,33,92
35,Female,49,33,14
36,Female,21,33,81
37,Female,42,34,17
38,Female,30,34,73
39,Male,36,37,26
40,Female,20,37,75
41,Male,65,38,35
42,Male,24,38,92
43,Female,48,39,36
44,Female,31,39,61
45,Female,49,39,28
46,Female,24,39,65
47,Female,50,40,55
48,Male,27,40,47
49,Female,29,40,42
50,Male,31,40,42
51,Male,49,42,52
52,Female,33,42,60
53,Female,31,43,54
54,Male,59,43,60
55,Female,50,43,45
56,Male,47,43,41
57,Female,51,44,50
58,Male,69,44,46
59,Female,27,46,51
60,Male,53,46,46
61,Male,70,46,56
62,Female,19,46,55
63,Female,67,47,52
64,Female,54,47,59
65,Male,63,48,51
66,Male,18,48,59
67,Female,43,48,50
68,Female,68,48,48
69,Male,19,48,59
70,Female,32,48,47
71,Male,70,49,55
72,Female,47,49,42
73,Female,60,50,49
74,Female,60,50,56
75,Male,59,50,47
76,Male,24,50,54
77,Female,26,50,52
78,Male,63,50,42
79,Female,58,51,44
80,Male,67,51,46
81,Female,35,51,46
82,Female,58,51,69
83,Male,54,51,31
84,Male,29,52,55
85,Male,35,52,69
86,Female,55,53,48
87,Male,35,53,47
88,Male,38,54,42
89,Male,21,54,48
90,Female,35,54,42
91,Male,42,54,51
92,Female,40,54,55
93,Male,59,54,46
94,Male,60,54,55
95,Female,24,54,46
96,Female,31,54,55
97,Female,24,54,48
98,Male,31,54,47
99,Female,40,54,48
100,Male,42,54,52
101,Male,31,54,55
102,Female,40,55,47
103,Female,41,55,52
104,Male,43,55,47
105,Female,41,55,50
106,Male,31,55,55
107,Female,58,56,47
108,Female,26,56,55
109,Male,31,56,50
110,Female,26,56,48
111,Male,31,58,55
112,Female,31,58,48
113,Female,28,60,50
114,Male,31,60,55
115,Female,29,60,42
116,Male,31,60,47
117,Female,22,60,52
118,Male,29,60,55
119,Female,31,61,49
120,Female,29,61,56
121,Male,35,62,50
122,Female,31,62,48
123,Male,54,62,47
124,Female,29,62,55
125,Male,33,63,46
126,Female,31,63,55
127,Female,59,63,50
128,Male,50,63,48
129,Female,47,64,52
130,Male,51,64,55
131,Female,69,65,48
132,Male,27,65,59
133,Female,53,65,43
134,Male,70,65,60
135,Female,19,65,55
136,Male,67,65,46
137,Female,54,67,53
138,Male,63,67,47
139,Female,18,67,59
140,Male,68,67,53
141,Female,19,69,75
142,Male,32,69,71
143,Female,35,69,73
144,Male,47,69,72
145,Female,45,69,63
146,Male,60,70,42
147,Female,39,70,65
148,Male,24,70,70
149,Female,40,71,55
150,Male,41,71,60
151,Female,45,71,52
152,Male,23,71,59
153,Female,48,71,55
154,Male,25,72,60
155,Female,29,72,55
156,Male,55,72,60
157,Female,58,72,46
158,Male,26,72,59
159,Female,35,73,55
160,Male,35,73,60
161,Female,55,74,46
162,Male,26,74,59
163,Female,42,74,55
164,Male,42,74,60
165,Female,51,74,46
166,Male,30,74,59
167,Female,36,74,55
168,Male,30,74,46
169,Female,25,75,73
170,Male,28,75,72
171,Female,33,75,71
172,Male,56,75,73
173,Female,31,76,72
174,Male,28,76,71
175,Female,24,77,73
176,Male,29,77,72
177,Female,35,77,71
178,Male,35,77,73
179,Female,48,77,72
180,Male,20,77,91
181,Female,56,77,73
182,Male,26,77,62
183,Female,50,78,72
184,Male,29,78,91
185,Female,23,78,62
186,Male,29,78,72
187,Female,43,78,91
188,Male,54,78,62
189,Female,59,78,72
190,Male,30,78,62
191,Female,37,78,91
192,Male,37,78,72
193,Female,22,78,62
194,Male,37,78,91
195,Female,52,78,72
196,Male,28,78,62
197,Female,28,78,91
198,Male,39,78,72
199,Female,24,78,62
200,Male,44,78,91"""

df = pd.read_csv(io.StringIO(RAW))
features  = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
X         = df[features].values
scaler    = StandardScaler()
X_scaled  = scaler.fit_transform(X)

# ── 2. K-distance graph for multiple k values ─────────────────────────────────
# Rule of thumb: min_samples = 2 * n_features = 6, use k = min_samples - 1 = 5
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, k in zip(axes, [3, 5, 7]):
    nbrs    = NearestNeighbors(n_neighbors=k).fit(X_scaled)
    dists,_ = nbrs.kneighbors(X_scaled)
    k_dists = np.sort(dists[:, -1])[::-1]
    ax.plot(range(len(k_dists)), k_dists)
    ax.set_title(f"K-distance  (k={k})")
    ax.set_xlabel("Points sorted by distance")
    ax.set_ylabel(f"{k}-NN distance")
    ax.axhline(y=0.5, color='red',  linestyle='--', alpha=0.5, label='eps=0.5')
    ax.axhline(y=0.8, color='orange', linestyle='--', alpha=0.5, label='eps=0.8')
    ax.axhline(y=1.2, color='green', linestyle='--', alpha=0.5, label='eps=1.2')
    ax.legend(fontsize=8)

plt.suptitle("K-distance graphs — elbow suggests good eps range", y=1.02)
plt.tight_layout()
plt.savefig("08a_kdistance.png", dpi=150)
plt.show()

# ── 3. Grid search over (eps, min_samples) ────────────────────────────────────
eps_values        = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5]
min_samples_values = [3, 4, 5, 6, 7, 8, 10]

print(f"\n{'eps':>6} {'min_s':>6} {'clusters':>9} {'noise%':>7} {'silhouette':>11}")
print("-" * 45)

best_score  = -1
best_params = {}
grid_results = []

for eps in eps_values:
    for ms in min_samples_values:
        db   = DBSCAN(eps=eps, min_samples=ms)
        lbls = db.fit_predict(X_scaled)
        n_clusters = len(set(lbls)) - (1 if -1 in lbls else 0)
        n_noise    = (lbls == -1).sum()
        noise_pct  = round(n_noise / len(lbls) * 100, 1)

        # only score if we have at least 2 clusters and not too much noise
        valid = lbls[lbls != -1]
        X_v   = X_scaled[lbls != -1]
        if n_clusters >= 2 and len(set(valid)) >= 2 and noise_pct < 30:
            sil = silhouette_score(X_v, valid)
        else:
            sil = float("nan")

        grid_results.append((eps, ms, n_clusters, noise_pct, sil))
        marker = " <-- best" if (not np.isnan(sil) and sil > best_score) else ""
        if not np.isnan(sil) and sil > best_score:
            best_score  = sil
            best_params = {"eps": eps, "min_samples": ms}
        print(f"{eps:>6.1f} {ms:>6}  {n_clusters:>8}  {noise_pct:>6}%  "
              f"{'nan' if np.isnan(sil) else f'{sil:.3f}':>10}{marker}")

print(f"\nBest params : eps={best_params['eps']}  min_samples={best_params['min_samples']}")
print(f"Best silhouette : {best_score:.3f}")

# ── 4. Heatmap of silhouette scores ──────────────────────────────────────────
sil_matrix = np.full((len(min_samples_values), len(eps_values)), np.nan)
for eps, ms, n_c, noise, sil in grid_results:
    i = min_samples_values.index(ms)
    j = eps_values.index(eps)
    sil_matrix[i, j] = sil if not np.isnan(sil) else 0

fig, ax = plt.subplots(figsize=(10, 5))
im = ax.imshow(sil_matrix, aspect='auto', cmap='YlOrRd', vmin=0, vmax=0.5)
ax.set_xticks(range(len(eps_values)));      ax.set_xticklabels(eps_values)
ax.set_yticks(range(len(min_samples_values))); ax.set_yticklabels(min_samples_values)
ax.set_xlabel("eps"); ax.set_ylabel("min_samples")
ax.set_title("Silhouette score heatmap — brighter = better\n(grey = <2 clusters or >30% noise)")
plt.colorbar(im, ax=ax)

# mark best cell
bi = min_samples_values.index(best_params["min_samples"])
bj = eps_values.index(best_params["eps"])
ax.add_patch(plt.Rectangle((bj - 0.5, bi - 0.5), 1, 1,
             fill=False, edgecolor='blue', lw=3))
ax.text(bj, bi, f"{best_score:.3f}", ha='center', va='center',
        fontsize=9, fontweight='bold', color='blue')

plt.tight_layout()
plt.savefig("08b_heatmap.png", dpi=150)
plt.show()

# ── 5. Best DBSCAN vs KMeans side by side ────────────────────────────────────
pca  = PCA(n_components=2, random_state=42)
X_2d = pca.fit_transform(X_scaled)
var  = pca.explained_variance_ratio_

db_best  = DBSCAN(eps=best_params["eps"], min_samples=best_params["min_samples"])
lbls_db  = db_best.fit_predict(X_scaled)
km       = KMeans(n_clusters=5, n_init=10, random_state=42)
lbls_km  = km.fit_predict(X_scaled)
sil_km   = silhouette_score(X_scaled, lbls_km)

n_noise = (lbls_db == -1).sum()
valid   = lbls_db[lbls_db != -1]
sil_db  = silhouette_score(X_scaled[lbls_db != -1], valid)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, lbls, title in zip(axes,
        [lbls_km, lbls_db],
        [f"KMeans k=5  (sil={sil_km:.3f})",
         f"DBSCAN eps={best_params['eps']} ms={best_params['min_samples']}"
         f"  (sil={sil_db:.3f}, noise={n_noise})"]):
    unique = sorted(set(lbls))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(unique), 1)))
    for lbl, col in zip(unique, colors):
        mask   = lbls == lbl
        marker = 'x' if lbl == -1 else 'o'
        label  = 'Noise' if lbl == -1 else f'C{lbl}'
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   c=[col], marker=marker, s=25, alpha=0.8, label=label)
    ax.set_title(title)
    ax.set_xlabel(f"PC1 ({var[0]:.0%})")
    ax.set_ylabel(f"PC2 ({var[1]:.0%})")
    ax.legend(fontsize=8, markerscale=1.5)

plt.tight_layout()
plt.savefig("08c_best_vs_kmeans.png", dpi=150)
plt.show()

# ── TODO tasks ────────────────────────────────────────────────────────────────
# 1. Read the k-distance graph — where does the elbow sit for k=5?
# 2. Read the heatmap — which region (high eps / low eps) produces most clusters?
# 3. What happens to noise% as eps increases? Why?
# 4. Can you tune DBSCAN to find exactly 5 clusters with <10% noise?
#    Hint: look for the cell in the grid closest to n_clusters=5
