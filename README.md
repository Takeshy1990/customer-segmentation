# 👥 Customer Segmentation with RFM & KMeans Clustering

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Pandas](https://img.shields.io/badge/Data-Pandas-green)
![Scikit-learn](https://img.shields.io/badge/ML-KMeans-orange)
![Matplotlib](https://img.shields.io/badge/Plots-Matplotlib-purple)
![ReportLab](https://img.shields.io/badge/Report-PDF-red)

This project demonstrates an **end-to-end unsupervised learning pipeline** for **Customer Segmentation**.  
We use **RFM analysis (Recency, Frequency, Monetary)** combined with **KMeans clustering** to group customers into meaningful segments for targeted marketing strategies.  

The deliverables include:
- Cleaned datasets (`customers_rfm.csv`, `customers_segments.csv`)  
- Visualizations (`inertia_plot.png`, `cluster_scatter.png`, `cluster_metrics.csv`)  
- Automated **PDF business report** (`segmentation_report.pdf`)  

---

## 📊 Data

### Raw Transactions — `transactions.csv`
Contains sample e-commerce transactions.

| Column       | Description                           | Example          |
|--------------|---------------------------------------|------------------|
| `customer_id`| Unique customer ID                    | C001             |
| `order_id`   | Unique order ID                       | O1001            |
| `order_date` | Date of transaction (YYYY-MM-DD)      | 2025-01-05       |
| `amount_eur` | Transaction amount (€)                | 42.90            |
| `category`   | Product category                      | Grocery          |
| `channel`    | Channel (Web/Store)                   | Web              |
| `country`    | Country code                          | GR               |

---

## ⚙️ Pipeline Overview

1. **RFM Build (`rfm_build.py`)**
   - Cleans raw transactions.
   - Computes RFM features:
     - `RecencyDays` = days since last purchase
     - `Frequency` = number of unique purchases
     - `Monetary` = total spend (€)
   - Adds extra metrics: AvgOrderValue, 90-day spend, First/Last purchase.
   - Outputs → `customers_rfm.csv`

2. **Clustering (`segmentation_train.py`)**
   - Scales RFM features and applies **KMeans**.
   - Produces:
     - `customers_segments.csv` (with cluster labels)
     - `inertia_plot.png` (Elbow method)
     - `cluster_scatter.png` (visualization)

3. **Evaluation (`segmentation_eval.py`)**
   - Compares clustering quality using:
     - Silhouette Score  
     - Davies–Bouldin Index  
     - Calinski–Harabasz Score  
   - Outputs → `cluster_metrics.csv`

4. **Reporting (`segmentation_report.py`)**
   - Summarizes cluster profiles.
   - Includes:
     - Table with averages per cluster
     - Scatter plot
     - Business insights
   - Outputs → `segmentation_report.pdf`

---

## ⚙️ Setup

You can reproduce this project using either **pip** or **conda**.

### Option A — pip
```bash
pip install -r requirements.txt
Option B — conda
bash
Αντιγραφή κώδικα
conda env create -f environment.yml
conda activate segmentation-env
For development (linting, tests):

bash
Αντιγραφή κώδικα
pip install -r requirements-dev.txt
🚀 How to Run
bash
Αντιγραφή κώδικα
# 1. Build RFM features
python rfm_build.py --plots

# 2. Train clusters (default: 4 clusters)
python segmentation_train.py --clusters 4 --plots

# 3. Evaluate clustering quality (2 ≤ k ≤ 8)
python segmentation_eval.py

# 4. Generate PDF report
python segmentation_report.py
Outputs:

customers_rfm.csv — RFM features

customers_segments.csv — RFM + cluster labels

inertia_plot.png, cluster_scatter.png — plots

cluster_metrics.csv — clustering metrics

segmentation_report.pdf — final business report

📈 Results (Demo)
Cluster Summary (example from demo data)
Cluster	Count	RecencyDays	Frequency	Monetary (€)
0	4	103.25	2.25	268.15
1	1	17.00	5.00	236.20
2	4	24.25	2.50	71.40
3	1	1.00	3.00	828.00

Visualizations
Elbow Method (Inertia):

Clusters (Frequency vs Monetary):

Clustering Metrics (example cluster_metrics.csv)
k	silhouette	davies_bouldin	calinski_harabasz
2	0.52	0.74	312.4
3	0.58	0.69	428.9
4	0.61	0.65	510.3
5	0.55	0.72	480.1

(→ In this demo, k=4 shows a good balance of silhouette and stability.)

📑 Executive Summary
Champions (Cluster 3) → very recent, high-value customers (VIPs).

Loyal (Cluster 1) → frequent buyers with steady spend.

At Risk (Cluster 0) → long recency, moderate spend — need retention campaigns.

Low Value (Cluster 2) → infrequent, low spend — potential upsell targets.

📌 This segmentation enables data-driven marketing strategies:

Retention campaigns for at-risk customers

Rewards programs for VIPs

Upsell campaigns for regular/low spenders

📦 Reproducibility
requirements.txt — pip dependencies

requirements-dev.txt — dev tools (pytest, flake8, black, mypy)

environment.yml — Conda environment

.github/workflows/ci.yml — GitHub Actions CI (smoke tests)

            | GR               
