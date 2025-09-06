# segmentation_train.py — Cluster customers on RFM features
#
# Usage:
#   python segmentation_train.py --clusters 4 --plots
#
# Input:  customers_rfm.csv
# Output: customers_segments.csv (+ inertia_plot.png, cluster_scatter.png)

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

BASE = Path(".").resolve()
IN_FILE = BASE / "customers_rfm.csv"
OUT_FILE = BASE / "customers_segments.csv"
INERTIA_PNG = BASE / "inertia_plot.png"
SCATTER_PNG = BASE / "cluster_scatter.png"

def parse_args():
    p = argparse.ArgumentParser(description="Cluster customers using KMeans on RFM features")
    p.add_argument("--clusters", type=int, default=4, help="number of clusters (default=4)")
    p.add_argument("--plots", action="store_true", help="save inertia and scatter plots")
    return p.parse_args()

def main():
    args = parse_args()
    if not IN_FILE.exists():
        raise FileNotFoundError(f"Missing {IN_FILE}, run rfm_build.py first.")

    df = pd.read_csv(IN_FILE)

    # Features for clustering
    features = ["RecencyDays", "Frequency", "Monetary"]
    X = df[features].values

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit KMeans
    km = KMeans(n_clusters=args.clusters, random_state=42, n_init=10)
    df["Cluster"] = km.fit_predict(X_scaled)

    # Save output
    df.to_csv(OUT_FILE, index=False, encoding="utf-8")
    print(f"[ok] wrote {OUT_FILE} with {args.clusters} clusters")

    if args.plots and plt is not None:
        # Inertia plot (Elbow method)
        inertias = []
        ks = range(1, 8)
        for k in ks:
            km_tmp = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_scaled)
            inertias.append(km_tmp.inertia_)
        plt.figure(figsize=(6,4))
        plt.plot(ks, inertias, marker="o")
        plt.xlabel("k"); plt.ylabel("Inertia")
        plt.title("Elbow Method")
        plt.tight_layout(); plt.savefig(INERTIA_PNG); plt.close()

        # Scatter: Frequency vs Monetary, colored by cluster
        plt.figure(figsize=(6,5))
        for c in range(args.clusters):
            d = df[df["Cluster"] == c]
            plt.scatter(d["Frequency"], d["Monetary"], label=f"Cluster {c}", alpha=0.7)
        plt.xlabel("Frequency"); plt.ylabel("Monetary (€)")
        plt.title("Clusters — Frequency vs Monetary")
        plt.legend()
        plt.tight_layout(); plt.savefig(SCATTER_PNG); plt.close()

        print(f"[plots] saved {INERTIA_PNG.name}, {SCATTER_PNG.name}")

if __name__ == "__main__":
    main()