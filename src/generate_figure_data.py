import sys
sys.stdout.reconfigure(encoding="utf-8")

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

IN_DIR = Path("output/paper2/figures")
OUT_DIR = Path("output/paper2/figures")

# Set academic plotting style
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams["font.family"] = "sans-serif"

def save_plot(fig, name):
    path = OUT_DIR / name
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")

# -------------------------------------------------------------------------
# FIG 1: Clustering Sensitivity
# -------------------------------------------------------------------------
print("Plotting Fig 1...")
df1 = pd.read_csv(IN_DIR / "fig1_clustering_sensitivity.csv")
df1["err_low"] = df1["estimate"] - df1["conf.low"]
df1["err_high"] = df1["conf.high"] - df1["estimate"]

fig, ax = plt.subplots(figsize=(8, 4))
sns.scatterplot(
    data=df1, x="estimate", y="measure", hue="cluster", 
    palette={"psgc10": "#1f77b4", "province_code": "#d62728"},
    s=100, ax=ax, zorder=3
)
for i, row in df1.iterrows():
    color = "#1f77b4" if row["cluster"] == "psgc10" else "#d62728"
    ax.errorbar(
        x=row["estimate"], y=row["measure"], 
        xerr=[[row["err_low"]], [row["err_high"]]], 
        fmt="none", color=color, capsize=4, capthick=2, zorder=2
    )
ax.axvline(0, color="red", linestyle="--", zorder=1)
ax.set_xlabel("Beta (Percentage points of poverty incidence)")
ax.set_ylabel("")
ax.set_title("Clustering Sensitivity: LGU vs Province")
ax.legend(title="Clustering Level", loc="lower right")
save_plot(fig, "fig1_clustering_sensitivity.png")

# -------------------------------------------------------------------------
# FIG 2: Trajectories
# -------------------------------------------------------------------------
print("Plotting Fig 2...")
df2 = pd.read_csv(IN_DIR / "fig2_trajectories.csv")
fig, ax = plt.subplots(figsize=(8, 5))
sns.lineplot(
    data=df2, x="year", y="mean_poverty", hue="group", 
    marker="o", markersize=8, linewidth=2, ax=ax,
    hue_order=["Always", "Exit", "Entry", "Never"]
)
ax.set_xticks([2018, 2021, 2023])
ax.set_xlabel("Fiscal Year")
ax.set_ylabel("Mean Poverty Incidence (%)")
ax.set_title("Poverty Trajectories by Dynasty Entry Group")
ax.legend(title="Group")
save_plot(fig, "fig2_trajectories.png")

# -------------------------------------------------------------------------
# FIG 3: Heterogeneity
# -------------------------------------------------------------------------
print("Plotting Fig 3...")
df3 = pd.read_csv(IN_DIR / "fig3_heterogeneity.csv")
df3["err_low"] = df3["estimate"] - df3["conf.low"]
df3["err_high"] = df3["conf.high"] - df3["estimate"]

fig, ax = plt.subplots(figsize=(8, 5))
sns.scatterplot(
    data=df3, x="estimate", y="category", hue="type", 
    palette={"Year": "#2ca02c", "Island": "#9467bd"},
    s=100, ax=ax, zorder=3
)
for i, row in df3.iterrows():
    color = "#2ca02c" if row["type"] == "Year" else "#9467bd"
    ax.errorbar(
        x=row["estimate"], y=row["category"], 
        xerr=[[row["err_low"]], [row["err_high"]]], 
        fmt="none", color=color, capsize=4, capthick=2, zorder=2
    )
ax.axvline(0, color="red", linestyle="--", zorder=1)
ax.set_xlabel("Beta (Percentage points of poverty incidence)")
ax.set_ylabel("")
ax.set_title("Heterogeneity in the Dynasty-Poverty Association")
ax.legend(title="Dimension", loc="lower right")
save_plot(fig, "fig3_heterogeneity.png")

# -------------------------------------------------------------------------
# FIG 4: Fat vs Thin
# -------------------------------------------------------------------------
print("Plotting Fig 4...")
df4 = pd.read_csv(IN_DIR / "fig4_fat_vs_thin.csv")
df4["err_low"] = df4["estimate"] - df4["conf.low"]
df4["err_high"] = df4["conf.high"] - df4["estimate"]

fig, ax = plt.subplots(figsize=(8, 3))
sns.scatterplot(
    data=df4, x="estimate", y="contrast", color="#333333",
    s=100, ax=ax, zorder=3
)
for i, row in df4.iterrows():
    ax.errorbar(
        x=row["estimate"], y=row["contrast"], 
        xerr=[[row["err_low"]], [row["err_high"]]], 
        fmt="none", color="#333333", capsize=4, capthick=2, zorder=2
    )
ax.axvline(0, color="red", linestyle="--", zorder=1)
ax.set_xlabel("Beta (Percentage points of poverty incidence)")
ax.set_ylabel("")
ax.set_title("Fat vs Thin Dynasties: A Bounded Null")
save_plot(fig, "fig4_fat_vs_thin.png")

# -------------------------------------------------------------------------
# FIG A: Convergence Test (Binned Scatter)
# -------------------------------------------------------------------------
print("Plotting Fig A...")
dfA_bins = pd.read_csv(IN_DIR / "figA_convergence_bins.csv")
fig, ax = plt.subplots(figsize=(7, 5))
sns.scatterplot(
    data=dfA_bins, x="mean_pov_2018", y="mean_delta", hue="dynasty_label",
    palette={"Non-Dynastic": "#1f77b4", "Dynastic": "#d62728"},
    s=80, ax=ax, zorder=3
)
sns.regplot(
    data=dfA_bins, x="mean_pov_2018", y="mean_delta", 
    scatter=False, color="#333333", ax=ax, line_kws={"linewidth": 2, "linestyle": "--"}
)
ax.axhline(0, color="gray", linestyle=":", zorder=1)
ax.set_xlabel("Initial Poverty Incidence in 2018 (%)")
ax.set_ylabel("Change in Poverty 2018→2023 (pp)")
ax.set_title("Convergence: Initial Poverty Drives Decline (Not Dynasty)")
ax.legend(title="2018 Status")
save_plot(fig, "figA_convergence_scatter.png")

# FIG A Coefs
dfA_coef = pd.read_csv(IN_DIR / "figA_convergence_coeffs.csv")
dfA_coef["err_low"] = dfA_coef["estimate"] - dfA_coef["conf.low"]
dfA_coef["err_high"] = dfA_coef["conf.high"] - dfA_coef["estimate"]
fig, ax = plt.subplots(figsize=(7, 2.5))
sns.scatterplot(data=dfA_coef, x="estimate", y="term", color="#333333", s=100, ax=ax, zorder=3)
for i, row in dfA_coef.iterrows():
    ax.errorbar(x=row["estimate"], y=row["term"], xerr=[[row["err_low"]], [row["err_high"]]], 
                fmt="none", color="#333333", capsize=4, capthick=2, zorder=2)
ax.axvline(0, color="red", linestyle="--", zorder=1)
ax.set_xlabel("Coefficient Estimate")
ax.set_ylabel("")
ax.set_title("Convergence Model Coefficients")
save_plot(fig, "figA_convergence_coeffs.png")

# -------------------------------------------------------------------------
# FIG B: Entry vs Never
# -------------------------------------------------------------------------
print("Plotting Fig B...")
dfB = pd.read_csv(IN_DIR / "figB_entry_vs_never.csv")
fig, ax = plt.subplots(figsize=(6, 4))
means = dfB[dfB["type"] == "mean"]
diff = dfB[dfB["type"] == "diff"].iloc[0]

sns.barplot(
    data=means, x="group", y="mean", color="#4c72b0", 
    alpha=0.7, ax=ax, zorder=2
)
# Add the difference and CI as a text annotation
ax.text(1, means["mean"].max() + 0.5, 
        f"Difference: {diff['mean']:+.2f} pp\n95% CI: [{diff['conf.low']:.2f}, {diff['conf.high']:.2f}]",
        ha="center", fontsize=10, bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))

ax.set_ylim(means["mean"].min() - 2, means["mean"].max() + 2)
ax.set_xlabel("")
ax.set_ylabel("Mean Poverty Incidence in 2018 (%)")
ax.set_title("Pre-Transition Null: Entry vs Never")
ax.grid(axis='x') # Remove vertical gridlines for bar chart
save_plot(fig, "figB_entry_vs_never.png")

print("\nAll plots successfully generated and saved to PNG!")