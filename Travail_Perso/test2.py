import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_parquet("outputs/merged.parquet")

# clip competitor price to see the real structure
df["pc"] = df["price_competitor"].clip(0.1, 120.0)
df["gap"] = df["price"] - df["pc"]

def binned_curve(x, y, bins):
    b = pd.cut(x, bins=bins)
    g = pd.DataFrame({"bin": b, "y": y}).groupby("bin")["y"].mean()
    mid = np.array([i.mid for i in g.index])
    return mid, g.values

for cap in [False, True]:
    d = df[df["competitor_has_capacity"] == cap].copy()

    # Demand vs price (binned mean)
    plt.figure()
    xmid, ymean = binned_curve(d["price"], d["demand"], bins=np.arange(5, 101, 2))
    plt.plot(xmid, ymean, marker="o")
    plt.title(f"Mean demand vs price (binned) | capacity={cap}")
    plt.xlabel("price"); plt.ylabel("mean demand")
    plt.tight_layout()
    plt.savefig(f"outputs/binned_demand_vs_price_cap_{cap}.png", dpi=160)
    plt.close()

    # Mean revenue proxy vs price (binned mean)
    plt.figure()
    rev = d["price"] * d["demand"]
    xmid, ymean = binned_curve(d["price"], rev, bins=np.arange(5, 101, 2))
    plt.plot(xmid, ymean, marker="o")
    plt.title(f"Mean revenue proxy vs price (binned) | capacity={cap}")
    plt.xlabel("price"); plt.ylabel("mean price*demand")
    plt.tight_layout()
    plt.savefig(f"outputs/binned_revenue_vs_price_cap_{cap}.png", dpi=160)
    plt.close()

    # Demand vs competitor price (binned mean)
    plt.figure()
    xmid, ymean = binned_curve(d["pc"], d["demand"], bins=np.arange(0, 121, 5))
    plt.plot(xmid, ymean, marker="o")
    plt.title(f"Mean demand vs competitor price (binned) | capacity={cap}")
    plt.xlabel("price_competitor (clipped)"); plt.ylabel("mean demand")
    plt.tight_layout()
    plt.savefig(f"outputs/binned_demand_vs_pc_cap_{cap}.png", dpi=160)
    plt.close()

print("Saved binned plots to outputs/")

