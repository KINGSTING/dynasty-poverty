import sys, os, importlib.util
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

here = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("m09", os.path.join(here, "descriptive_checks.py"))
m09 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m09)
COL, fit = m09.COL, m09.fit

DATA = sys.argv[1] if len(sys.argv) > 1 else "data/processed/panel_dynasty_funds.parquet"
OUTD = sys.argv[2] if len(sys.argv) > 2 else "results"
FIG = "figures"; os.makedirs(FIG, exist_ok=True)
Y, S, A, X = COL["y"], COL["share"], COL["anyd"], COL["controls"]
U, T, P = COL["unit"], COL["year"], COL["prov"]

plt.rcParams.update({"font.size": 9, "font.family": "sans-serif", "axes.spines.top": False,
                     "axes.spines.right": False, "axes.linewidth": 0.8, "figure.dpi": 100})
BLK, GRY = "#111111", "#8a8a8a"

def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig); print("saved", name)

def one(res, term):
    r = res[res.term == term].iloc[0]
    return r.beta, r.lo, r.hi

df = m09.prepare(DATA)
df = df[df[S].notna()].copy()                       # clean sample: share observed (never miscoded as 0)
cs = df.dropna(subset=[Y, S, A] + X).copy()         # common regression sample
print(f"common sample N={len(cs):,}, LGUs={cs[U].nunique():,}")

# ------------------------------------------------------------------ Fig 1: specifications
cs["region"] = cs[U].str[:2]
base = cs.groupby(U)[Y].transform(lambda s: np.nan)
b18 = cs[cs[T] == cs[T].min()].set_index(U)[Y]
cs["base_pov"] = cs[U].map(b18)
terc = pd.qcut(cs["base_pov"], 3, labels=False, duplicates="drop")
tc = []
for k in (1, 2):
    for yy in sorted(cs[T].unique())[1:]:
        n = f"t{k}_y{yy}"; cs[n] = ((terc == k) & (cs[T] == yy)).astype(float); tc.append(n)
cnt = cs.groupby(U).size(); bal = cs[cs[U].isin(cnt[cnt == cs[T].nunique()].index)]

rows = []
for M, lab in ((S, "Dynastic share (0 to 1)"), (A, "Any dynastic official")):
    for nm, dat, fe, cl, extra in (
        ("LGU-clustered", cs, ("unit", "year"), "unit", []),
        ("Province-clustered (main)", cs, ("unit", "year"), "prov", []),
        ("Region-clustered", cs, ("unit", "year"), "region", []),
        ("Baseline tercile x year", cs, ("unit", "year"), "prov", tc),
        ("Province x year effects", cs, ("unit", "provyear"), "prov", []),
        ("Balanced panel", bal, ("unit", "year"), "prov", []),
    ):
        r = fit(dat, Y, [M] + X + extra, fe=fe, cluster=cl)
        b, lo, hi = one(r, M); rows.append(dict(measure=lab, spec=nm, b=b, lo=lo, hi=hi, N=int(r.N.iloc[0])))
f1 = pd.DataFrame(rows); f1.to_csv(os.path.join(OUTD, "fig1_data.csv"), index=False)

fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), sharey=True)
specs = list(dict.fromkeys(f1.spec))
for ax, (lab, g) in zip(axes, f1.groupby("measure", sort=False)):
    g = g.set_index("spec").loc[specs]
    yy = np.arange(len(g))[::-1]
    for y_, (_, r) in zip(yy, g.iterrows()):
        main = "main" in _
        ax.plot([r.lo, r.hi], [y_, y_], color=BLK, lw=1.4 if main else 1.0)
        ax.plot(r.b, y_, "o", color=BLK if main else GRY, ms=5, mfc=BLK if main else "white")
    ax.axvline(0, color=GRY, lw=0.8, ls="--")
    ax.set_yticks(yy); ax.set_yticklabels(specs); ax.set_xlabel("Poverty incidence, percentage points")
    ax.set_title(lab, fontsize=9, loc="left")
fig.tight_layout(); save(fig, "fig1_association")

# ------------------------------------------------------------------ Fig 2: years and islands
rows = []
for M, lab in ((S, "Dynastic share"), (A, "Any dynastic official")):
    ycols = []
    d = cs.copy()
    for yy in sorted(d[T].unique()):
        d[f"m_{yy}"] = d[M] * (d[T] == yy); ycols.append(f"m_{yy}")
    r = fit(d, Y, ycols + X, cluster="prov")
    for yy, c in zip(sorted(d[T].unique()), ycols):
        b, lo, hi = one(r, c); rows.append(dict(measure=lab, panel="By year", group=str(yy), b=b, lo=lo, hi=hi))
    for isl in ("Luzon", "Visayas", "Mindanao"):
        s_ = cs[cs.island == isl]
        r = fit(s_, Y, [M] + X, cluster="prov")
        b, lo, hi = one(r, M); rows.append(dict(measure=lab, panel="By island group", group=isl, b=b, lo=lo, hi=hi))
f2 = pd.DataFrame(rows); f2.to_csv(os.path.join(OUTD, "fig2_data.csv"), index=False)

fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
for ax, (panel, g) in zip(axes, f2.groupby("panel", sort=False)):
    groups = list(dict.fromkeys(g.group)); xs = np.arange(len(groups))
    for off, (lab, gg) in zip((-0.12, 0.12), g.groupby("measure", sort=False)):
        gg = gg.set_index("group").loc[groups]
        filled = lab == "Dynastic share"
        ax.errorbar(xs + off, gg.b, yerr=[gg.b - gg.lo, gg.hi - gg.b], fmt="o" if filled else "s",
                    color=BLK, mfc=BLK if filled else "white", ms=5, lw=1, capsize=2, label=lab)
    ax.axhline(0, color=GRY, lw=0.8, ls="--")
    ax.set_xticks(xs); ax.set_xticklabels(groups); ax.set_title(panel, fontsize=9, loc="left")
axes[0].set_ylabel("Poverty incidence, percentage points")
axes[0].legend(frameon=False, fontsize=8, loc="upper right")
fig.tight_layout(); save(fig, "fig2_heterogeneity")

# ------------------------------------------------------------------ Fig 3: transition groups
w = df.pivot_table(index=U, columns=T, values=[Y, S], aggfunc="first")
y0, y1 = 2018, 2023
ok = w[(S, y0)].notna() & w[(S, y1)].notna() & w[(Y, y0)].notna() & w[(Y, y1)].notna()
w = w[ok]
grp = np.where((w[(S, y0)] > 0) & (w[(S, y1)] > 0), "Always dynastic",
      np.where((w[(S, y0)] == 0) & (w[(S, y1)] > 0), "Became dynastic",
      np.where((w[(S, y0)] > 0) & (w[(S, y1)] == 0), "Stopped being dynastic", "Never dynastic")))
w = w.assign(grp=grp)
rows = []
for gname, g in w.groupby("grp"):
    for yy in (y0, y1):
        v = g[(Y, yy)]; rows.append(dict(group=gname, year=yy, mean=v.mean(), se=v.std() / np.sqrt(len(v)), n=len(v)))
f3 = pd.DataFrame(rows); f3.to_csv(os.path.join(OUTD, "fig3_data.csv"), index=False)
print(f3.pivot(index="group", columns="year", values="mean").round(2).assign(n=f3.groupby("group").n.first()))

order = ["Always dynastic", "Became dynastic", "Stopped being dynastic", "Never dynastic"]
styles = {"Always dynastic": ("o", "-", BLK), "Became dynastic": ("s", "-", GRY),
          "Stopped being dynastic": ("s", "--", GRY), "Never dynastic": ("o", "--", BLK)}
fig, ax = plt.subplots(figsize=(4.6, 3.4))
for gname in order:
    g = f3[f3.group == gname].sort_values("year"); mk, ls, c = styles[gname]
    ax.errorbar(g.year, g["mean"], yerr=1.96 * g.se, marker=mk, ls=ls, color=c, mfc=c, ms=5, lw=1.2, capsize=2,
                label=f"{gname} (n={int(g.n.iloc[0])})")
ax.set_xticks([y0, y1]); ax.set_xlim(y0 - 0.8, y1 + 0.8)
ax.set_ylabel("Mean poverty incidence (%)"); ax.legend(frameon=False, fontsize=7.5, loc="lower left")
fig.tight_layout(); save(fig, "fig3_transition")

# ------------------------------------------------------------------ Fig 4: budget equivalence
eq_path = os.path.join(OUTD, "bureaucracy_equivalence.csv")
if os.path.exists(eq_path):
    eq = pd.read_csv(eq_path); eq = eq[eq.measure == "any dynastic"]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.6), sharey=True)
    for ax, (dsg, g) in zip(axes, eq.groupby("design", sort=False)):
        g = g.reset_index(drop=True); yy = np.arange(len(g))[::-1]
        ax.axvspan(-0.2, 0.2, color="#e6e6e6", lw=0)
        ax.axvline(0, color=GRY, lw=0.8, ls="--")
        for y_, (_, r) in zip(yy, g.iterrows()):
            flag = r.equiv_p > 0.05
            ax.plot([r.lo, r.hi], [y_, y_], color=BLK, lw=1.1)
            ax.plot(r.beta_SD, y_, "o", ms=5, color=BLK, mfc="white" if flag else BLK)
        ax.set_yticks(yy); ax.set_yticklabels(g.outcome)
        ax.set_xlim(-0.35, 0.35); ax.set_xlabel("Difference, SD of outcome")
        ax.set_title("Within LGU over time" if dsg.startswith("within LGU") else "Within province-year",
                     fontsize=9, loc="left")
    fig.text(0.5, -0.02, "Shaded band: +/-0.20 SD equivalence margin. Hollow marker: equivalence not established.",
             ha="center", fontsize=7.5)
    fig.tight_layout(); save(fig, "fig4_budget_equiv")
else:
    print("fig4 skipped: run 14_bureaucracy_equivalence.py first")
print("\nDone. Files in", FIG)