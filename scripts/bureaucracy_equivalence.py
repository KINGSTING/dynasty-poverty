import sys, os, importlib.util
import numpy as np, pandas as pd
from scipy import stats

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

here = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("m09", os.path.join(here, "descriptive_checks.py"))
m09 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m09)
COL = m09.COL

DATA = sys.argv[1] if len(sys.argv) > 1 else "data/processed/panel_dynasty_funds.parquet"
OUTD = sys.argv[2] if len(sys.argv) > 2 else "results"
MARGIN = 0.20          # equivalence margin in pooled SD units (state and justify in the paper)

df = m09.prepare(DATA)
S, A, Y = COL["share"], COL["anyd"], COL["y"]
df = df[df[S].notna()].copy()                       # clean sample: share observed

# ---- budget outcomes (same construction as your 13_clean_institutional_tests.py)
den = df["total_current_operating_exp"].replace(0, np.nan)
for k, c in (("share_gps", "gps"), ("share_edu", "education"), ("share_health", "health"),
             ("share_welfare", "social_welfare")):
    df[k] = (df[c] / den).clip(0, 1)
z = lambda s: (s - s.mean()) / s.std()
comp = {"share_welfare": 1, "share_health": 1, "share_edu": 1, "share_gps": -1,
        "fund_balance_share": 1, "revenue_hhi": -1}
df["iq_index"] = pd.concat([comp[k] * z(df[k]) for k in comp], axis=1).mean(axis=1)

outcomes = {"Health share": "share_health", "Education share": "share_edu",
            "Social welfare share": "share_welfare", "General public services share": "share_gps",
            "Fund balance share": "fund_balance_share", "Revenue concentration (HHI)": "revenue_hhi",
            "NTA dependency": "nta_dependency", "IQ index": "iq_index"}
for lab, v in outcomes.items():
    df[f"z_{v}"] = z(df[v])

CTRL = ["log_population"]       # only a size control: fiscal ratios are mechanically tied to NTA/local sources

def tost(beta, se, G, margin=MARGIN):
    t1 = (beta + margin) / se; t2 = (beta - margin) / se
    p_lo = stats.t.sf(t1, G - 1); p_hi = stats.t.cdf(t2, G - 1)
    crit90 = stats.t.ppf(0.95, G - 1)
    bound = max(abs(beta - crit90 * se), abs(beta + crit90 * se))
    return max(p_lo, p_hi), bound

rows = []
def run(design, fe):
    for M, mname in ((A, "any dynastic"), (S, "dynastic share")):
        for lab, v in outcomes.items():
            d = df.dropna(subset=[f"z_{v}", M] + CTRL)
            r = m09.fit(d, f"z_{v}", [M] + CTRL, fe=fe, cluster="prov")
            r = r[r.term == M].iloc[0]
            p_eq, bound = tost(r.beta, r.se, int(r.clusters))
            rows.append(dict(design=design, measure=mname, outcome=lab, beta_SD=r.beta, se=r.se, p=r.p,
                             lo=r.lo, hi=r.hi, N=int(r.N), clusters=int(r.clusters),
                             equiv_p=p_eq, equiv_margin=MARGIN, smallest_margin_ruled_in=bound))
run("within LGU (LGU + year FE)", ("unit", "year"))
run("within province-year (dynastic vs non-dynastic LGUs in the same province)", ("provyear",))

res = pd.DataFrame(rows)
# Holm adjustment across the 8 outcomes, separately for each design x measure, on the usual 'differs' test
res["holm_p"] = np.nan
for (dsg, ms), g in res.groupby(["design", "measure"]):
    order = g.p.sort_values().index; m = len(g)
    adj = np.maximum.accumulate([(m - i) * g.p[idx] for i, idx in enumerate(order)])
    res.loc[order, "holm_p"] = np.minimum(adj, 1.0)

pd.set_option("display.width", 200)
for (dsg, ms), g in res.groupby(["design", "measure"], sort=False):
    print(f"\n=== {dsg} | {ms} | margin +/-{MARGIN} SD ===")
    print(g[["outcome", "beta_SD", "se", "p", "holm_p", "lo", "hi", "equiv_p", "smallest_margin_ruled_in", "N"]]
          .round(3).to_string(index=False))

# ---- (c) does budget composition 'explain' the dynasty-poverty association?
print("\n=== Dynasty-poverty association with and without budget controls (descriptive, not mediation) ===")
budget = ["share_health", "share_edu", "share_welfare", "share_gps", "fund_balance_share", "revenue_hhi"]
X0 = COL["controls"]
d = df.dropna(subset=[Y, A, S] + X0 + budget)
out_c = []
for M, mname in ((A, "any dynastic"), (S, "dynastic share")):
    a = m09.fit(d, Y, [M] + X0, cluster="prov"); a = a[a.term == M].iloc[0]
    b = m09.fit(d, Y, [M] + X0 + budget, cluster="prov"); b = b[b.term == M].iloc[0]
    print(f"{mname:<15s} baseline {a.beta:+.3f} (se {a.se:.3f})  ->  with budget controls {b.beta:+.3f} (se {b.se:.3f})"
          f"   change {b.beta - a.beta:+.3f}   N={int(a.N)}")
    out_c.append(dict(measure=mname, baseline=a.beta, baseline_se=a.se, with_budget=b.beta, with_budget_se=b.se, N=int(a.N)))

os.makedirs(OUTD, exist_ok=True)
res.to_csv(os.path.join(OUTD, "bureaucracy_equivalence.csv"), index=False)
pd.DataFrame(out_c).to_csv(os.path.join(OUTD, "bureaucracy_budget_controls.csv"), index=False)
print("\nSaved to", OUTD)