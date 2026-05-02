# 07_real_data.py
#
# Dataset: Mall Customers (embedded — no download needed)
# 200 customers with Age, Annual Income (k$), Spending Score (1-100)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import io

# ── 1. Dataset (embedded) ─────────────────────────────────────────────────────
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
print("Shape :", df.shape)
print("\nFirst rows:")
print(df.head())

# ── 2. Select features & scale ────────────────────────────────────────────────
features = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
X = df[features].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── 3. Elbow + Silhouette to choose k ────────────────────────────────────────
inertias, sil_scores = [], []
ks = range(2, 11)
for k in ks:
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    km.fit(X_scaled)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, km.labels_))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(ks, inertias, 'o-')
axes[0].set_xlabel("k"); axes[0].set_ylabel("Inertia")
axes[0].set_title("Elbow curve")

axes[1].plot(ks, sil_scores, 'o-', color='tab:orange')
axes[1].set_xlabel("k"); axes[1].set_ylabel("Silhouette score")
axes[1].set_title("Silhouette — pick the peak")

plt.tight_layout()
plt.savefig("07a_choosing_k.png", dpi=150)
plt.show()

best_k = list(ks)[np.argmax(sil_scores)]
print(f"\nBest k by silhouette : {best_k}")

# ── 4. Fit all 4 algorithms ───────────────────────────────────────────────────
models = [
    ("KMeans",       KMeans(n_clusters=best_k, n_init=10, random_state=42)),
    ("GMM",          GaussianMixture(n_components=best_k, random_state=42)),
    ("Hierarchical", AgglomerativeClustering(n_clusters=best_k, linkage="ward")),
    ("DBSCAN",       DBSCAN(eps=0.8, min_samples=5)),
]

results = {}
for name, model in models:
    if hasattr(model, "fit_predict"):
        lbls = model.fit_predict(X_scaled)
    else:
        model.fit(X_scaled); lbls = model.predict(X_scaled)
    mask = lbls != -1
    sil = silhouette_score(X_scaled[mask], lbls[mask]) if mask.sum() > 1 else float("nan")
    n_c = len(set(lbls)) - (1 if -1 in lbls else 0)
    results[name] = {"labels": lbls, "silhouette": round(sil, 3), "n_clusters": n_c}
    print(f"{name:<14}  clusters={n_c}  silhouette={sil:.3f}")

# ── 5. Visualise with PCA (3D → 2D) ──────────────────────────────────────────
pca = PCA(n_components=2, random_state=42)
X_2d = pca.fit_transform(X_scaled)
var  = pca.explained_variance_ratio_
print(f"\nPCA variance explained: PC1={var[0]:.1%}  PC2={var[1]:.1%}  total={sum(var):.1%}")

fig, axes = plt.subplots(1, 4, figsize=(18, 4))
for ax, (name, res) in zip(axes, results.items()):
    lbls   = res["labels"]
    unique = sorted(set(lbls))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(unique), 1)))
    for lbl, col in zip(unique, colors):
        mask   = lbls == lbl
        marker = 'x' if lbl == -1 else 'o'
        label  = 'Noise' if lbl == -1 else f'C{lbl}'
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   c=[col], marker=marker, s=20, alpha=0.7, label=label)
    ax.set_title(f"{name}\nsil={res['silhouette']}  k={res['n_clusters']}")
    ax.set_xlabel(f"PC1 ({var[0]:.0%})")
    ax.set_ylabel(f"PC2 ({var[1]:.0%})")
    ax.legend(fontsize=7, markerscale=1.5)

plt.tight_layout()
plt.savefig("07b_clusters_pca.png", dpi=150)
plt.show()

# ── 6. Interpret the best clustering ─────────────────────────────────────────
df["Cluster"] = results["KMeans"]["labels"]
print("\nCluster profiles (KMeans) — mean values per cluster:")
print(df.groupby("Cluster")[features].mean().round(1).to_string())

# ── TODO tasks ────────────────────────────────────────────────────────────────
# 1. Read the cluster profiles table — can you name each customer segment?
#    Hint: high income + high spending = "VIP"
#          low income + high spending  = "impulsive buyer"
#          high income + low spending  = "careful saver"
#          low income + low spending   = "budget conscious"
# 2. Try features = ["Annual Income (k$)", "Spending Score (1-100)"] only (2D)
#    Plot directly without PCA — do you see 5 cleaner visual clusters?
# 3. Encode Gender: gender_enc = pd.get_dummies(df["Gender"], drop_first=True)
#    Add it as a feature and re-run — does it change the clusters?
# 4. Try DBSCAN with eps=0.5 and eps=1.2 — how does the noise count change?

# 2D plot — Income vs Spending (no PCA needed)
km5 = KMeans(n_clusters=5, n_init=10, random_state=42)
lbls2d = km5.fit_predict(X_scaled)

plt.figure(figsize=(7, 5))
income = df["Annual Income (k$)"].values
spending = df["Spending Score (1-100)"].values
colors = plt.cm.tab10(np.linspace(0, 1, 5))
for i in range(5):
    mask = lbls2d == i
    plt.scatter(income[mask], spending[mask],
                c=[colors[i]], s=40, alpha=0.8, label=f"C{i}")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score")
plt.title("K-means k=5 — Income vs Spending (no PCA)")
plt.legend()
plt.tight_layout()
plt.savefig("07c_income_spending.png", dpi=150)
plt.show()