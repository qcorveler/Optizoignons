import os
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Helpers
# -----------------------------
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def savefig(path: str) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

def clip_pc(series: pd.Series, lo=0.1, hi=120.0) -> pd.Series:
    return series.clip(lower=lo, upper=hi)

def add_revenue(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["revenue"] = df["price"] * df["demand"]
    df["price_competitor_clipped"] = clip_pc(df["price_competitor"])
    df["gap"] = df["price"] - df["price_competitor_clipped"]
    return df

def load_all_runs(data_dir: str) -> dict:
    paths = sorted(glob.glob(os.path.join(data_dir, "duopoly_competition_details*.csv")))
    runs = {}
    for p in paths:
        try:
            df = pd.read_csv(p)
            needed = {"competition_id", "price", "demand", "competitor_has_capacity", "price_competitor"}
            if not needed.issubset(df.columns):
                print(f"SKIP (missing cols): {os.path.basename(p)}")
                continue
            runs[os.path.basename(p)] = add_revenue(df)
        except Exception as e:
            print(f"SKIP (read error): {os.path.basename(p)} -> {e}")
    return runs

def summarize_runs(runs: dict) -> pd.DataFrame:
    rows = []
    for name, df in runs.items():
        cap_false = (df["competitor_has_capacity"] == False)
        cap_true  = (df["competitor_has_capacity"] == True)

        rows.append({
            "file": name,
            "rows": int(len(df)),
            "competitions": int(df["competition_id"].nunique()),
            "revenue_sum": float(df["revenue"].sum()),
            "revenue_mean": float(df["revenue"].mean()),
            "revenue_mean_per_comp": float(df.groupby("competition_id")["revenue"].mean().mean()),
            "demand_mean": float(df["demand"].mean()),
            "price_mean": float(df["price"].mean()),
            "price_median": float(df["price"].median()),
            "pc_mean": float(df["price_competitor_clipped"].mean()),
            "gap_mean": float(df["gap"].mean()),
            "capFalse_share": float(cap_false.mean()),
            "capFalse_rev_mean": float(df.loc[cap_false, "revenue"].mean()) if cap_false.any() else np.nan,
            "capTrue_rev_mean": float(df.loc[cap_true, "revenue"].mean()) if cap_true.any() else np.nan,
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("revenue_mean", ascending=False)
    return out

def binned_mean(x: pd.Series, y: pd.Series, bins: int = 30):
    # returns bin centers + mean(y) per bin
    x = x.to_numpy()
    y = y.to_numpy()
    edges = np.linspace(np.nanmin(x), np.nanmax(x), bins + 1)
    idx = np.digitize(x, edges) - 1
    centers, means = [], []
    for b in range(bins):
        mask = idx == b
        if mask.sum() < 50:
            continue
        centers.append((edges[b] + edges[b+1]) / 2)
        means.append(np.nanmean(y[mask]))
    return np.array(centers), np.array(means)

# -----------------------------
# Plot suite
# -----------------------------
def plot_ranking(summary: pd.DataFrame, outdir: str):
    # 1) Revenue mean ranking
    df = summary.copy()
    df = df.sort_values("revenue_mean", ascending=True)

    plt.figure(figsize=(10, 6))
    plt.barh(df["file"], df["revenue_mean"])
    plt.xlabel("Average revenue per period (price × demand)")
    plt.title("Run ranking by average revenue (comparable)")
    savefig(os.path.join(outdir, "01_ranking_revenue_mean.png"))

    # 2) Price mean vs revenue mean scatter
    plt.figure(figsize=(7, 6))
    plt.scatter(summary["price_mean"], summary["revenue_mean"])
    for _, r in summary.iterrows():
        plt.annotate(r["file"].split("(")[-1].split(")")[0], (r["price_mean"], r["revenue_mean"]), fontsize=8)
    plt.xlabel("Mean price")
    plt.ylabel("Mean revenue")
    plt.title("Price level vs performance (run-level)")
    savefig(os.path.join(outdir, "02_price_mean_vs_revenue_mean.png"))

    # 3) capFalse_share vs revenue_mean
    plt.figure(figsize=(7, 6))
    plt.scatter(summary["capFalse_share"], summary["revenue_mean"])
    for _, r in summary.iterrows():
        plt.annotate(r["file"].split("(")[-1].split(")")[0], (r["capFalse_share"], r["revenue_mean"]), fontsize=8)
    plt.xlabel("Share of periods with competitor_has_capacity = False")
    plt.ylabel("Mean revenue")
    plt.title("Impact of competitor capacity regime on performance")
    savefig(os.path.join(outdir, "03_capFalse_share_vs_revenue_mean.png"))

def plot_global_eda(all_df: pd.DataFrame, outdir: str):
    # demand distribution
    plt.figure(figsize=(8, 5))
    plt.hist(all_df["demand"], bins=30)
    plt.xlabel("Demand")
    plt.ylabel("Count")
    plt.title("Demand distribution (all runs pooled)")
    savefig(os.path.join(outdir, "04_global_demand_hist.png"))

    # price distribution
    plt.figure(figsize=(8, 5))
    plt.hist(all_df["price"], bins=40)
    plt.xlabel("Price")
    plt.ylabel("Count")
    plt.title("Our price distribution (all runs pooled)")
    savefig(os.path.join(outdir, "05_global_price_hist.png"))

    # competitor price clipped distribution
    plt.figure(figsize=(8, 5))
    plt.hist(all_df["price_competitor_clipped"], bins=40)
    plt.xlabel("Competitor price (clipped)")
    plt.ylabel("Count")
    plt.title("Competitor price distribution (clipped, all runs pooled)")
    savefig(os.path.join(outdir, "06_global_comp_price_hist.png"))

    # demand vs price (binned mean) split by capacity
    for cap in [True, False]:
        sub = all_df[all_df["competitor_has_capacity"] == cap]
        if len(sub) == 0:
            continue
        x, y = binned_mean(sub["price"], sub["demand"], bins=30)
        plt.figure(figsize=(7, 5))
        plt.plot(x, y, marker="o", linewidth=1)
        plt.xlabel("Our price")
        plt.ylabel("Mean demand (binned)")
        plt.title(f"Demand vs Price (binned) | competitor_has_capacity={cap}")
        savefig(os.path.join(outdir, f"07_demand_vs_price_binned_cap_{cap}.png"))

    # revenue vs price (binned mean) split by capacity
    for cap in [True, False]:
        sub = all_df[all_df["competitor_has_capacity"] == cap]
        if len(sub) == 0:
            continue
        x, y = binned_mean(sub["price"], sub["revenue"], bins=30)
        plt.figure(figsize=(7, 5))
        plt.plot(x, y, marker="o", linewidth=1)
        plt.xlabel("Our price")
        plt.ylabel("Mean revenue (binned)")
        plt.title(f"Revenue vs Price (binned) | competitor_has_capacity={cap}")
        savefig(os.path.join(outdir, f"08_revenue_vs_price_binned_cap_{cap}.png"))

def plot_best_vs_worst(best_df: pd.DataFrame, worst_df: pd.DataFrame, best_name: str, worst_name: str, outdir: str):
    # Price histogram overlay
    plt.figure(figsize=(8, 5))
    plt.hist(best_df["price"], bins=40, alpha=0.6, label=f"Best: {best_name}")
    plt.hist(worst_df["price"], bins=40, alpha=0.6, label=f"Worst: {worst_name}")
    plt.xlabel("Price")
    plt.ylabel("Count")
    plt.title("Price distribution: best vs worst run")
    plt.legend()
    savefig(os.path.join(outdir, "09_best_vs_worst_price_hist.png"))

    # Revenue histogram overlay
    plt.figure(figsize=(8, 5))
    plt.hist(best_df["revenue"], bins=40, alpha=0.6, label=f"Best: {best_name}")
    plt.hist(worst_df["revenue"], bins=40, alpha=0.6, label=f"Worst: {worst_name}")
    plt.xlabel("Revenue (price × demand)")
    plt.ylabel("Count")
    plt.title("Revenue distribution: best vs worst run")
    plt.legend()
    savefig(os.path.join(outdir, "10_best_vs_worst_revenue_hist.png"))

    # Gap histogram overlay (price - competitor)
    plt.figure(figsize=(8, 5))
    plt.hist(best_df["gap"], bins=50, alpha=0.6, label=f"Best: {best_name}")
    plt.hist(worst_df["gap"], bins=50, alpha=0.6, label=f"Worst: {worst_name}")
    plt.xlabel("Gap = our price - competitor price (clipped)")
    plt.ylabel("Count")
    plt.title("Pricing gap distribution: best vs worst run")
    plt.legend()
    savefig(os.path.join(outdir, "11_best_vs_worst_gap_hist.png"))

    # Binned revenue vs price for best and worst (cap=True only, most common)
    for cap in [True, False]:
        bsub = best_df[best_df["competitor_has_capacity"] == cap]
        wsub = worst_df[worst_df["competitor_has_capacity"] == cap]
        if len(bsub) < 1000 or len(wsub) < 1000:
            continue
        xb, yb = binned_mean(bsub["price"], bsub["revenue"], bins=25)
        xw, yw = binned_mean(wsub["price"], wsub["revenue"], bins=25)

        plt.figure(figsize=(7, 5))
        plt.plot(xb, yb, marker="o", linewidth=1, label=f"Best: {best_name}")
        plt.plot(xw, yw, marker="o", linewidth=1, label=f"Worst: {worst_name}")
        plt.xlabel("Our price")
        plt.ylabel("Mean revenue (binned)")
        plt.title(f"Revenue vs Price (binned) | cap={cap}")
        plt.legend()
        savefig(os.path.join(outdir, f"12_best_vs_worst_revenue_vs_price_cap_{cap}.png"))

def plot_per_competition_time_series(df: pd.DataFrame, run_name: str, outdir: str):
    # Pick the first competition_id and plot price & demand over selling_period if exists
    if "selling_period" not in df.columns:
        return

    cid = df["competition_id"].iloc[0]
    sub = df[df["competition_id"] == cid].copy()
    sub = sub.sort_values("selling_period")

    # Price time series
    plt.figure(figsize=(10, 4))
    plt.plot(sub["selling_period"], sub["price"], linewidth=1)
    plt.xlabel("Selling period")
    plt.ylabel("Our price")
    plt.title(f"Our price over time | {run_name} | competition_id={cid}")
    savefig(os.path.join(outdir, "13_timeseries_price_example.png"))

    # Demand time series
    plt.figure(figsize=(10, 4))
    plt.plot(sub["selling_period"], sub["demand"], linewidth=1)
    plt.xlabel("Selling period")
    plt.ylabel("Demand")
    plt.title(f"Demand over time | {run_name} | competition_id={cid}")
    savefig(os.path.join(outdir, "14_timeseries_demand_example.png"))

def plot_runtime_story_panel(summary: pd.DataFrame, outdir: str):
    # A compact “story” plot: revenue_mean and price_mean side-by-side for top 8 runs
    top = summary.head(8).copy()
    top = top.iloc[::-1]

    fig = plt.figure(figsize=(10, 6))
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.barh(top["file"], top["revenue_mean"])
    ax1.set_xlabel("Mean revenue")
    ax1.set_title("Top runs by mean revenue (for presentation)")
    savefig(os.path.join(outdir, "15_top_runs_bar.png"))

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=r".\Data", help="Folder with duopoly_competition_details*.csv")
    ap.add_argument("--out_dir", default=r".\outputs\figs_presentation", help="Output folder for figures")
    ap.add_argument("--best", default=None, help="Best run filename (e.g., 'duopoly_competition_details (5).csv')")
    ap.add_argument("--worst", default=None, help="Worst run filename (e.g., 'duopoly_competition_details (11).csv')")
    args = ap.parse_args()

    ensure_dir(args.out_dir)

    runs = load_all_runs(args.data_dir)
    if not runs:
        raise FileNotFoundError(f"No runs loaded from {os.path.abspath(args.data_dir)}")

    summary = summarize_runs(runs)
    summary.to_csv(os.path.join(args.out_dir, "summary_runs.csv"), index=False)

    # Pool all data (for global EDA)
    all_df = pd.concat(list(runs.values()), ignore_index=True)

    # Plots
    plot_ranking(summary, args.out_dir)
    plot_global_eda(all_df, args.out_dir)
    plot_runtime_story_panel(summary, args.out_dir)

    # Best/worst defaults from summary if not provided
    best_name = args.best if args.best else summary.iloc[0]["file"]
    worst_name = args.worst if args.worst else summary.iloc[-1]["file"]

    if best_name in runs and worst_name in runs:
        plot_best_vs_worst(runs[best_name], runs[worst_name], best_name, worst_name, args.out_dir)
        # Optional time series on best run
        plot_per_competition_time_series(runs[best_name], best_name, args.out_dir)

    print("\n=== DONE ===")
    print(f"Figures saved in: {os.path.abspath(args.out_dir)}")
    print(f"Run summary saved: {os.path.abspath(os.path.join(args.out_dir, 'summary_runs.csv'))}")
    print(f"Best run used: {best_name}")
    print(f"Worst run used: {worst_name}")

if __name__ == "__main__":
    main()
