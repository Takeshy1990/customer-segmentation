# rfm_build.py — Build RFM/features per customer from transactions
# Usage:
#   python rfm_build.py
#   python rfm_build.py --snapshot 2025-09-01 --plots
#   python rfm_build.py --file transactions.csv --sep , --encoding utf-8
#
# Input:  transactions.csv  (columns: customer_id,order_id,order_date,amount_eur,category,channel,country)
# Output: customers_rfm.csv (+ optional rfm_scatter.png, rfm_hist.png)

from __future__ import annotations
import argparse
from pathlib import Path
import sys
import math
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Optional plotting (only if --plots)
try:
    import matplotlib.pyplot as plt  # no seaborn by requirement
except Exception:
    plt = None

BASE = Path(".").resolve()
IN_FILE_DEFAULT = BASE / "transactions.csv"
OUT_FILE = BASE / "customers_rfm.csv"
SCATTER_PNG = BASE / "rfm_scatter.png"
HIST_PNG = BASE / "rfm_hist.png"

def parse_args():
    p = argparse.ArgumentParser(description="Build RFM dataset from transactions")
    p.add_argument("--file", type=str, default=str(IN_FILE_DEFAULT), help="input CSV path")
    p.add_argument("--sep", type=str, default=None, help="CSV separator (auto if not given)")
    p.add_argument("--encoding", type=str, default=None, help="file encoding (auto if not given)")
    p.add_argument("--snapshot", type=str, default=None, help="YYYY-MM-DD snapshot date (default: max(order_date)+1)")
    p.add_argument("--plots", action="store_true", help="export quick plots (rfm_scatter.png, rfm_hist.png)")
    return p.parse_args()

# ------------------------ IO helpers ------------------------ #
def try_read_csv(path: Path, sep=None, encoding=None) -> pd.DataFrame:
    seps = [sep] if sep else [",", ";", "\t", "|"]
    encs = [encoding] if encoding else ["utf-8", "utf-8-sig", "cp1253", "cp1252"]
    errors = []
    for s in seps:
        for e in encs:
            try:
                df = pd.read_csv(path, sep=s, encoding=e, engine="python")
                # Basic validation
                if "customer_id" in df.columns and "order_date" in df.columns and "amount_eur" in df.columns:
                    return df
            except Exception as ex:
                errors.append((s, e, str(ex)))
    raise RuntimeError(f"Could not read {path}. Tried: {errors[:3]}...")

def coerce_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Standardize column names to lowercase
    df.columns = [c.strip().lower() for c in df.columns]
    # Trim string columns
    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].astype(str).str.strip()
    # Required columns
    required = ["customer_id", "order_id", "order_date", "amount_eur"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Parse dates
    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce", utc=False)
    # Amount: make numeric; replace commas, currency symbols if any
    df["amount_eur"] = (
        df["amount_eur"]
        .astype(str)
        .str.replace(",", ".", regex=False)
        .str.replace("€", "", regex=False)
        .str.replace(" ", "", regex=False)
    )
    df["amount_eur"] = pd.to_numeric(df["amount_eur"], errors="coerce")

    # Drop invalid rows
    before = len(df)
    df = df[~df["customer_id"].isna() & ~df["order_id"].isna()]
    df = df[~df["order_date"].isna() & ~df["amount_eur"].isna()]
    # Remove negative or absurd amounts (keep zeros? we will drop zeros)
    df = df[df["amount_eur"] > 0]
    after = len(df)
    if after < before:
        print(f"[clean] dropped {before - after} invalid rows")

    return df

def pick_snapshot(df: pd.DataFrame, user_snapshot: str | None) -> pd.Timestamp:
    if user_snapshot:
        try:
            snap = pd.to_datetime(user_snapshot)
            return snap
        except Exception:
            raise ValueError(f"Invalid --snapshot date: {user_snapshot}")
    # Default snapshot: day after max order_date
    max_date = df["order_date"].max()
    if pd.isna(max_date):
        raise ValueError("Cannot infer snapshot date (no valid order_date)")
    return (max_date + pd.Timedelta(days=1)).normalize()

# ------------------------ Feature engineering ------------------------ #
def rfm_features(df: pd.DataFrame, snapshot: pd.Timestamp) -> pd.DataFrame:
    # Base aggregations
    grp = df.groupby("customer_id", as_index=False).agg(
        Monetary=("amount_eur", "sum"),
        Frequency=("order_id", "nunique"),
        FirstPurchase=("order_date", "min"),
        LastPurchase=("order_date", "max"),
        AvgOrderValue=("amount_eur", "mean")
    )
    grp["RecencyDays"] = (snapshot - grp["LastPurchase"]).dt.days.astype(int)
    grp["DaysSinceFirst"] = (snapshot - grp["FirstPurchase"]).dt.days.astype(int)

    # 90-day window metrics
    win_start = snapshot - pd.Timedelta(days=90)
    recent = df[df["order_date"] >= win_start]
    recent_agg = recent.groupby("customer_id", as_index=False).agg(
        OrdersLast90d=("order_id", "nunique"),
        MonetaryLast90d=("amount_eur", "sum")
    )
    grp = grp.merge(recent_agg, on="customer_id", how="left")
    grp[["OrdersLast90d", "MonetaryLast90d"]] = grp[["OrdersLast90d", "MonetaryLast90d"]].fillna(0)

    # Percentiles for R/F/M (winsorize for stability)
    def winsorize(x, q=0.99):
        hi = x.quantile(q)
        return np.clip(x, None, hi)

    grp["Monetary_clip"] = winsorize(grp["Monetary"])
    grp["Frequency_clip"] = winsorize(grp["Frequency"])
    grp["Recency_inv"] = 1 / (grp["RecencyDays"] + 1)  # smaller recency -> bigger score

    # Score each 1..5 using quintiles
    def score_quintile(series, reverse=False):
        # reverse=False: higher = better (F,M, Recency_inv)
        # reverse=True: higher raw = worse (RecencyDays), but we use Recency_inv instead
        q = series.rank(method="average", pct=True)
        # quintiles: (0, .2], (.2, .4], (.4, .6], (.6, .8], (.8, 1]
        return (np.ceil(q * 5)).astype(int).clip(1, 5)

    R_score = score_quintile(grp["Recency_inv"])
    F_score = score_quintile(grp["Frequency_clip"])
    M_score = score_quintile(grp["Monetary_clip"])

    grp["R_score"] = R_score
    grp["F_score"] = F_score
    grp["M_score"] = M_score
    grp["RFM_sum"] = grp["R_score"] + grp["F_score"] + grp["M_score"]

    # Segment naming (simple rule-based)
    def segment_row(r, f, m):
        if r >= 4 and f >= 4 and m >= 4:
            return "Champions"
        if r >= 4 and f >= 3:
            return "Loyal"
        if r >= 3 and f >= 2 and m >= 3:
            return "Potential Loyalist"
        if r <= 2 and f <= 2 and m <= 2:
            return "At Risk / Dormant"
        if r >= 3 and m >= 4:
            return "Big Spenders"
        if r >= 4 and f <= 2:
            return "New Customers"
        return "Regular"

    grp["Segment"] = [segment_row(r, f, m) for r, f, m in zip(R_score, F_score, M_score)]

    # Order columns nicely
    cols = [
        "customer_id", "RecencyDays", "Frequency", "Monetary",
        "R_score", "F_score", "M_score", "RFM_sum", "Segment",
        "AvgOrderValue", "OrdersLast90d", "MonetaryLast90d",
        "FirstPurchase", "LastPurchase", "DaysSinceFirst"
    ]
    grp = grp[cols].sort_values(["RFM_sum", "Monetary", "Frequency"], ascending=[False, False, False]).reset_index(drop=True)
    return grp

# ------------------------ Quick plots ------------------------ #
def quick_plots(df_out: pd.DataFrame):
    if plt is None:
        print("[plots] matplotlib not available, skipping plots")
        return
    try:
        # Scatter: Frequency vs Monetary colored by Segment
        segs = df_out["Segment"].unique().tolist()
        colors = {s: None for s in segs}  # default colors (matplotlib chooses)
        plt.figure(figsize=(6,5))
        for s in segs:
            d = df_out[df_out["Segment"] == s]
            plt.scatter(d["Frequency"], d["Monetary"], label=s, alpha=0.7)
        plt.xlabel("Frequency"); plt.ylabel("Monetary (€)")
        plt.title("RFM Scatter — Frequency vs Monetary")
        plt.legend(fontsize=8, loc="best")
        plt.tight_layout(); plt.savefig(SCATTER_PNG); plt.close()

        # Histogram of Recency
        plt.figure(figsize=(6,4))
        plt.hist(df_out["RecencyDays"], bins=10)
        plt.xlabel("Recency (days)"); plt.ylabel("Customers")
        plt.title("Distribution of Recency")
        plt.tight_layout(); plt.savefig(HIST_PNG); plt.close()
        print(f"[plots] saved {SCATTER_PNG.name}, {HIST_PNG.name}")
    except Exception as ex:
        print(f"[plots] warning: {ex}")

def main():
    args = parse_args()
    path = Path(args.file)
    if not path.exists():
        print(f"Input file not found: {path}", file=sys.stderr)
        sys.exit(1)

    df = try_read_csv(path, sep=args.sep, encoding=args.encoding)
    df = coerce_columns(df)

    snapshot = pick_snapshot(df, args.snapshot)
    print(f"[info] snapshot date = {snapshot.date()} (default = max(order_date)+1)")

    out = rfm_features(df, snapshot)
    out.to_csv(OUT_FILE, index=False, encoding="utf-8")
    print(f"[ok] wrote {OUT_FILE} with {len(out)} customers")

    if args.plots:
        quick_plots(out)

if __name__ == "__main__":
    main()