# 👥 Customer Segmentation with RFM & Clustering

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Pandas](https://img.shields.io/badge/Data-Pandas-green)
![Scikit-learn](https://img.shields.io/badge/ML-KMeans-orange)
![ReportLab](https://img.shields.io/badge/Report-PDF-red)

This project demonstrates an **unsupervised learning pipeline** for **Customer Segmentation**.  
We use **RFM analysis (Recency, Frequency, Monetary)** and **KMeans clustering** to group customers into meaningful segments for targeted marketing strategies.

---

## 📊 Data

- **transactions.csv** → raw purchase transactions (customer_id, order_date, amount_eur, category, channel, country).
- **customers_rfm.csv** → engineered RFM features per customer.
- **customers_segments.csv** → final dataset with assigned clusters.

### Example transaction

| customer_id | order_id | order_date | amount_eur | category     | channel | country |
|-------------|----------|------------|------------|--------------|---------|---------|
| C001        | O1001    | 2025-01-05 | 42.90      | Grocery      | Web     | GR      |

---

## ⚙️ Pipeline Overview

1. **RFM Build (`rfm_build.py`)**
   - Cleans raw transactions.
   - Calculates RecencyDays, Frequency, Monetary, AvgOrderValue, 90-day metrics.
   - Generates `customers_rfm.csv`.

2. **Clustering (`segmentation_train.py`)**
   - Applies KMeans clustering on scaled RFM features.
   - Saves labeled dataset as `customers_segments.csv`.
   - Produces `inertia_plot.png` (elbow method) and `cluster_scatter.png`.

3. **Reporting (`segmentation_report.py`)**
   - Summarizes cluster profiles in a professional PDF.
   - Includes cluster averages table, scatter plot, and business insights.
   - Generates `segmentation_report.pdf`.

---

## 🚀 How to Run

```bash
# 1. Install dependencies
pip install pandas numpy scikit-learn matplotlib reportlab

# 2. Build RFM features
python rfm_build.py --plots

# 3. Train clusters (default 4 clusters)
python segmentation_train.py --clusters 4 --plots

# 4. Generate PDF report
python segmentation_report.py
Outputs:

customers_rfm.csv — RFM features.

customers_segments.csv — RFM + cluster labels.

inertia_plot.png, cluster_scatter.png — visualizations.

segmentation_report.pdf — final business report.

📈 Results (Demo)
Cluster summary (example):

Cluster	Count	RecencyDays	Frequency	Monetary (€)
0	4	103.25	2.25	268.15
1	1	17.00	5.00	236.20
2	4	24.25	2.50	71.40
3	1	1.00	3.00	828.00

Scatter Plot (Frequency vs Monetary):


📑 Executive Summary
Champions (Cluster 3) → recent, high-value, VIP customers.

Loyal (Cluster 1) → frequent buyers, steady spend.

Dormant / At Risk (Cluster 0) → long recency, medium spend.

Low Value (Cluster 2) → infrequent, low spend customers.

📌 Marketing actions:

Retention campaigns for At Risk.

Rewards programs for Champions.

Upsell strategies for Regulars/Low Value.