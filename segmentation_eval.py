from pathlib import Path
import pandas as pd, numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

BASE = Path(".")
df = pd.read_csv(BASE/"customers_rfm.csv")
X = df[["RecencyDays","Frequency","Monetary"]].values
X = StandardScaler().fit_transform(X)

rows = []
for k in range(2, 9):
    sils, dbs, chs = [], [], []
    for seed in [0,1,2,3,4]:
        km = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(X)
        lab = km.labels_
        sils.append(silhouette_score(X, lab))
        dbs.append(davies_bouldin_score(X, lab))
        chs.append(calinski_harabasz_score(X, lab))
    rows.append({
        "k": k,
        "silhouette_mean": float(np.mean(sils)),
        "davies_bouldin_mean": float(np.mean(dbs)),
        "calinski_harabasz_mean": float(np.mean(chs))
    })

out = BASE/"cluster_metrics.csv"
pd.DataFrame(rows).to_csv(out, index=False)
print(f"Saved {out}")