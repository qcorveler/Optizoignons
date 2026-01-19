# duopoly_data_audit.py
# Usage:
#   python duopoly_data_audit.py --zip Data.zip
# (ou mets le chemin complet vers ton zip)

import os
import re
import json
import zipfile
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Optional plotting (safe if missing)
try:
    import matplotlib.pyplot as plt
    HAS_PLT = True
except Exception:
    HAS_PLT = False


DUOPOLY_PATTERN = re.compile(r"duopoly_competition_details.*\.csv$", re.IGNORECASE)

EXPECTED_COLS = [
    "competition_id",
    "selling_season",
    "selling_period",
    "competitor_id",
    "price_competitor",
    "price",
    "demand",
    "competitor_has_capacity",
    "calculation_duration",
    "errors",
]


def unzip_to_tmp(zip_path: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(out_dir)
    return out_dir


def find_csvs(root: Path):
    csvs = []
    for p in root.rglob("*.csv"):
        if DUOPOLY_PATTERN.search(p.name):
            csvs.append(p)
    return sorted(csvs)


def robust_outlier_count(s: pd.Series, k=1.5):
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return 0
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return 0
    lo, hi = q1 - k * iqr, q3 + k * iqr
    return int(((s < lo) | (s > hi)).sum())


def basic_ranges(df: pd.DataFrame, col: str):
    s = pd.to_numeric(df[col], errors="coerce")
    return {
        "min": float(np.nanmin(s.values)) if np.isfinite(np.nanmin(s.values)) else None,
        "p01": float(np.nanpercentile(s.values, 1)) if np.isfinite(np.nanpercentile(s.values, 1)) else None,
        "p05": float(np.nanpercentile(s.values, 5)) if np.isfinite(np.nanpercentile(s.values, 5)) else None,
        "median": float(np.nanmedian(s.values)) if np.isfinite(np.nanmedian(s.values)) else None,
        "p95": float(np.nanpercentile(s.values, 95)) if np.isfinite(np.nanpercentile(s.values, 95)) else None,
        "p99": float(np.nanpercentile(s.values, 99)) if np.isfinite(np.nanpercentile(s.values, 99)) else None,
        "max": float(np.nanmax(s.values)) if np.isfinite(np.nanmax(s.values)) else None,
    }


def make_plots(df: pd.DataFrame, out_dir: Path):
    if not HAS_PLT:
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    # Sample for speed
    d = df.sample(min(len(df), 200_000), random_state=42)

    # 1) demand vs price (scatter alpha low)
    plt.figure()
    plt.scatter(d["price"], d["demand"], s=2)
    plt.xlabel("price")
    plt.ylabel("demand")
    plt.title("Demand vs Price (sample)")
    plt.tight_layout()
    plt.savefig(out_dir / "demand_vs_price.png", dpi=160)
    plt.close()

    # 2) demand vs price_competitor
    plt.figure()
    plt.scatter(d["price_competitor"], d["demand"], s=2)
    plt.xlabel("price_competitor")
    plt.ylabel("demand")
    plt.title("Demand vs Competitor Price (sample)")
    plt.tight_layout()
    plt.savefig(out_dir / "demand_vs_price_competitor.png", dpi=160)
    plt.close()

    # 3) price difference vs demand
    diff = d["price"] - d["price_competitor"]
    plt.figure()
    plt.scatter(diff, d["demand"], s=2)
    plt.xlabel("price - price_competitor")
    plt.ylabel("demand")
    plt.title("Demand vs Price Gap (sample)")
    plt.tight_layout()
    plt.savefig(out_dir / "demand_vs_price_gap.png", dpi=160)
    plt.close()


def main(zip_path: str):
    zip_path = Path(zip_path).expanduser().resolve()
    if not zip_path.exists():
        raise FileNotFoundError(f"Zip not found: {zip_path}")

    root = zip_path.parent
    work_dir = root / "_tmp_duopoly_unzip"
    outputs = root / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)

    unzip_to_tmp(zip_path, work_dir)
    csvs = find_csvs(work_dir)
    if not csvs:
        raise RuntimeError("No duopoly_competition_details*.csv found in zip after extraction.")

    # Load & concat
    dfs = []
    bad_cols_files = []
    for f in csvs:
        df = pd.read_csv(f)
        df["__source_file"] = f.name
        # check columns
        missing = [c for c in EXPECTED_COLS if c not in df.columns]
        extra = [c for c in df.columns if c not in EXPECTED_COLS + ["__source_file"]]
        if missing or extra:
            bad_cols_files.append({"file": f.name, "missing": missing, "extra": extra})
        # reorder if possible
        keep_cols = [c for c in EXPECTED_COLS if c in df.columns] + ["__source_file"]
        df = df[keep_cols]
        dfs.append(df)

    merged = pd.concat(dfs, ignore_index=True)

    # Type cleaning
    num_cols = ["selling_season", "selling_period", "price_competitor", "price", "demand", "calculation_duration"]
    for c in num_cols:
        if c in merged.columns:
            merged[c] = pd.to_numeric(merged[c], errors="coerce")

    if "competitor_has_capacity" in merged.columns:
        # normalize to bool if possible
        ch = merged["competitor_has_capacity"]
        if ch.dtype != bool:
            merged["competitor_has_capacity"] = ch.astype(str).str.lower().map(
                {"true": True, "false": False, "1": True, "0": False}
            )

    # Core audit
    n_rows = int(len(merged))
    n_files = int(len(csvs))
    col_list = list(merged.columns)

    missing_by_col = merged.isna().mean().sort_values(ascending=False).to_dict()

    # duplicates (exact rows)
    dup_rows = int(merged.duplicated().sum())

    # errors
    errors_nonempty = None
    if "errors" in merged.columns:
        # treat empty string / "nan" as no error
        e = merged["errors"].astype(str)
        errors_nonempty = int((e.str.strip().ne("") & e.str.lower().ne("nan")).sum())

    # Value ranges
    ranges = {}
    for c in ["price", "price_competitor", "demand", "calculation_duration", "selling_season", "selling_period"]:
        if c in merged.columns:
            ranges[c] = basic_ranges(merged, c)

    # Outliers counts (IQR rule)
    outliers = {}
    for c in ["price", "price_competitor", "demand", "calculation_duration"]:
        if c in merged.columns:
            outliers[c] = robust_outlier_count(merged[c])

    # Quick “profit proxy” (IF score ~ price*demand)
    if "price" in merged.columns and "demand" in merged.columns:
        merged["__revenue_proxy"] = merged["price"] * merged["demand"]
        ranges["revenue_proxy"] = basic_ranges(merged, "__revenue_proxy")
        outliers["revenue_proxy"] = robust_outlier_count(merged["__revenue_proxy"])

    # Group summaries: per competition_id
    group_summary = {}
    if "competition_id" in merged.columns:
        g = merged.groupby("competition_id", dropna=False)
        group_summary = {
            "n_competitions": int(g.ngroups),
            "rows_per_competition": {
                "min": int(g.size().min()),
                "median": float(g.size().median()),
                "max": int(g.size().max()),
            }
        }

    # Season/period sanity
    season_period = {}
    if "selling_season" in merged.columns and "selling_period" in merged.columns:
        season_period = {
            "unique_selling_season": int(merged["selling_season"].nunique(dropna=True)),
            "unique_selling_period": int(merged["selling_period"].nunique(dropna=True)),
        }

    # Save merged (parquet is best)
    merged_path = outputs / "merged.parquet"
    merged.to_parquet(merged_path, index=False)

    # Save small sample to share
    sample_path = outputs / "head_200.csv"
    merged.head(200).to_csv(sample_path, index=False)

    # Basic competitor capacity effect
    capacity_effect = {}
    if "competitor_has_capacity" in merged.columns and "demand" in merged.columns:
        tmp = merged.dropna(subset=["competitor_has_capacity", "demand"])
        if not tmp.empty:
            cap_stats = tmp.groupby("competitor_has_capacity")["demand"].agg(["count", "mean", "median"]).reset_index()
            capacity_effect = cap_stats.to_dict(orient="records")

    # Save plots
    make_plots(merged.dropna(subset=["price", "price_competitor", "demand"]), outputs)

    summary = {
        "files_found": [p.name for p in csvs],
        "n_files": n_files,
        "n_rows": n_rows,
        "columns": col_list,
        "bad_columns_files": bad_cols_files,
        "missing_rate_by_column": missing_by_col,
        "duplicate_rows_exact": dup_rows,
        "errors_nonempty_rows": errors_nonempty,
        "ranges": ranges,
        "outlier_counts_iqr": outliers,
        "competition_group_summary": group_summary,
        "season_period": season_period,
        "capacity_effect_on_demand": capacity_effect,
        "saved": {
            "merged_parquet": str(merged_path),
            "head_200_csv": str(sample_path),
            "plots_dir": str(outputs),
        },
        "next_questions_for_strategy": [
            "Score formula: profit = price*demand ? cost ? penalties ? capacity on your side ?",
            "At decision time, do you observe competitor_has_capacity and price_competitor for the same period?",
            "Do you choose one price per (season, period) or per row/market?",
        ],
    }

    summary_path = outputs / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n=== DONE ===")
    print(f"Saved merged dataset: {merged_path}")
    print(f"Saved sample: {sample_path}")
    print(f"Saved summary: {summary_path}")
    if HAS_PLT:
        print(f"Saved plots in: {outputs}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True, help="Path to Data.zip (duopoly files)")
    args = ap.parse_args()
    main(args.zip)
