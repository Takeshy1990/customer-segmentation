# segmentation_report.py — Generate PDF report for customer segmentation
#
# Usage:
#   python segmentation_report.py
#
# Input:  customers_segments.csv
# Output: segmentation_report.pdf

from pathlib import Path
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image

BASE = Path(".").resolve()
IN_FILE = BASE / "customers_segments.csv"
OUT_FILE = BASE / "segmentation_report.pdf"

def build_report(df: pd.DataFrame):
    styles = getSampleStyleSheet()
    story = []

    # Title
    story.append(Paragraph("<b>Customer Segmentation Report</b>", styles["Title"]))
    story.append(Spacer(1, 12))

    # Intro
    intro = """This report summarizes the results of the customer segmentation analysis
    based on RFM (Recency, Frequency, Monetary) features. Customers have been grouped
    into distinct clusters, each representing a different behavior pattern."""
    story.append(Paragraph(intro, styles["Normal"]))
    story.append(Spacer(1, 12))

    # Cluster summary table
    summary = df.groupby("Cluster").agg({
        "RecencyDays": "mean",
        "Frequency": "mean",
        "Monetary": "mean",
        "customer_id": "count"
    }).rename(columns={"customer_id": "Count"}).reset_index()

    data = [summary.columns.tolist()] + summary.round(2).values.tolist()
    table = Table(data, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.grey),
        ("TEXTCOLOR", (0,0), (-1,0), colors.whitesmoke),
        ("ALIGN", (0,0), (-1,-1), "CENTER"),
        ("GRID", (0,0), (-1,-1), 0.5, colors.black)
    ]))
    story.append(table)
    story.append(Spacer(1, 12))

    # Insights
    insights = """<b>Insights:</b><br/>
    - Low Recency + High Frequency + High Monetary = Champions (loyal, valuable customers).<br/>
    - High Recency + Low Frequency = At risk or dormant customers.<br/>
    - Medium scores = Regular customers with growth potential.<br/>"""
    story.append(Paragraph(insights, styles["Normal"]))
    story.append(Spacer(1, 12))

    # If cluster_scatter.png exists, add it
    scatter_file = BASE / "cluster_scatter.png"
    if scatter_file.exists():
        story.append(Paragraph("Cluster Scatter Plot:", styles["Heading2"]))
        story.append(Image(str(scatter_file), width=400, height=300))
        story.append(Spacer(1, 12))

    # Conclusion
    concl = """This segmentation enables targeted marketing strategies such as
    retention campaigns for at-risk customers, VIP rewards for champions,
    and upsell campaigns for regular customers."""
    story.append(Paragraph(concl, styles["Normal"]))

    return story

def main():
    if not IN_FILE.exists():
        raise FileNotFoundError(f"Missing {IN_FILE}, run segmentation_train.py first.")

    df = pd.read_csv(IN_FILE)
    doc = SimpleDocTemplate(str(OUT_FILE), pagesize=A4)
    story = build_report(df)
    doc.build(story)
    print(f"[ok] wrote {OUT_FILE}")

if __name__ == "__main__":
    main()