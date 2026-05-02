# Clustering from A to Z

Learning project following the YouTube series by [AI for you — Morgan Gautherot](https://www.youtube.com/watch?v=qxVYofkoYyg).

Covers k-means, DBSCAN, Gaussian Mixture Models, and Hierarchical clustering using scikit-learn.

## Setup

```bash
python -m venv venv --without-pip
venv\Scripts\activate          # Windows
source venv/bin/activate       # macOS / Linux

curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python get-pip.py
del get-pip.py

pip install -r requirements.txt
```

## Lessons

| File | Topic |
|------|-------|
| `01_datasets.py` | Generate & visualise datasets |
| `02_kmeans.py` | K-means + elbow method |
| `03_dbscan.py` | DBSCAN + k-distance graph |
| `04_gmm.py` | Gaussian Mixture Model + BIC |
| `05_hierarchical.py` | Agglomerative clustering + dendrogram |
| `06_evaluation.py` | Silhouette, ARI, Davies-Bouldin comparison |

## Run a lesson

```bash
python 01_datasets.py
```

Each script saves a `.png` plot in the project folder.
