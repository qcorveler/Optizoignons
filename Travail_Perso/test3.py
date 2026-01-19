import glob
import os
import pandas as pd

DATA_DIR = r".\Data"
pattern = os.path.join(DATA_DIR, "*.csv")
paths = sorted(glob.glob(pattern))

print(f"Chemin courant: {os.getcwd()}")
print(f"Pattern: {pattern}")
print(f"Found {len(paths)} files")

if not paths:
    raise FileNotFoundError(
        f"Aucun CSV trouvé dans {os.path.abspath(DATA_DIR)}.\n"
        "Pour vérifier: fais `ls .\\Data` dans PowerShell."
    )

rows = []
for p in paths:
    df = pd.read_csv(p)

    needed = {"competition_id", "price", "demand", "competitor_has_capacity"}
    if not needed.issubset(df.columns):
        print(f"SKIP (missing columns): {os.path.basename(p)}")
        continue

    df["revenue"] = df["price"] * df["demand"]

    cap_false = (df["competitor_has_capacity"] == False)
    cap_true  = (df["competitor_has_capacity"] == True)

    rows.append({
        "file": os.path.basename(p),
        "rows": int(len(df)),
        "competitions": int(df["competition_id"].nunique()),
        "revenue_sum": float(df["revenue"].sum()),
        "revenue_mean": float(df["revenue"].mean()),
        "revenue_mean_per_comp": float(df.groupby("competition_id")["revenue"].mean().mean()),
        "demand_mean": float(df["demand"].mean()),
        "price_mean": float(df["price"].mean()),
        "capFalse_revenue_mean": float(df.loc[cap_false, "revenue"].mean()) if cap_false.any() else float("nan"),
        "capTrue_revenue_mean": float(df.loc[cap_true, "revenue"].mean()) if cap_true.any() else float("nan"),
        "capFalse_share": float(cap_false.mean()),
    })

out = pd.DataFrame(rows)
if out.empty:
    raise RuntimeError("Aucun fichier exploitable (tous SKIP).")

# TRI COMPARABLE : revenue_mean
out = out.sort_values("revenue_mean", ascending=False)

print("\n=== RANKING PAR revenue_mean (comparable) ===")
print(out.to_string(index=False))

os.makedirs("outputs", exist_ok=True)
out.to_csv("outputs/run_comparison_by_mean.csv", index=False)
print("\nSaved: outputs/run_comparison_by_mean.csv")
