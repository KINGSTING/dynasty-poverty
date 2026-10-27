"""
16_institutional_clean.py - replaces 13_clean_institutional_tests.py (same tests, same estimator as 09/14).

Why: script 13 uses statsmodels dummy-variable OLS, whose clustered SEs count the ~1,360 absorbed LGU
effects and come out ~37% larger than the absorbed-FE (Stata reghdfe / xtreg, fe) convention the paper states.
This version uses the estimator from 09_descriptive_checks.py, so every number in Section 4.5 matches
the rest of the paper. Province-clustered throughout.

Tests : 1. dynasty x centred health share      2. dynasty effect by IQ tercile + low-minus-high contrast
        3. dynasty x each of six budget components (Bonferroni)      4. dynasty -> IQ index
Run   : python src\\16_institutional_clean.py
"""
import sys, os, importlib.util
import numpy as np, pandas as pd

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
Y, S, A, X = COL["y"], COL["share"], COL["anyd"], COL["controls"]

df = m09.prepare(DATA)
df = df[df[S].notna()].copy()                              # clean sample: share observed

den = df["total_current_operating_exp"].replace(0, np.nan)
for k, c in (("share_gps", "gps"), ("share_edu", "education"), ("share_health", "health"),
             ("share_welfare", "social_welfare")):
    df[k] = (df[c] / den).clip(0, 1)
z = lambda s: (s - s.mean()) / s.std()
comp = {"share_welfare": 1, "share_health": 1, "share_edu": 1, "share_gps": -1,
        "fund_balance_share": 1, "revenue_hhi": -1}
df["iq_index"] = pd.concat([sgn * z(df[k]) for k, sgn in comp.items()], axis=1).mean(axis=1)

need = [Y, S, A, "iq_index", "share_health"] + X + list(comp)
cs = df.dropna(subset=need).copy()
print(f"\nSample: {len(cs):,} LGU-years, {cs[COL['unit']].nunique():,} LGUs, "
      f"{cs[COL['prov']].nunique()} provinces (script 09 baseline sample: 2,932 / 1,363; "
      f"differences come from the extra budget variables required here)")
rows = []
def keep(test, term, r, **kw):
    r = r[r.term == term].iloc[0]
    rows.append(dict(test=test, term=term, beta=r.beta, se=r.se, p=r.p, lo=r.lo, hi=r.hi, N=int(r.N), **kw))
    return r

# ---- 1. health interaction (centred, so the main effect is the dynasty effect at mean health share)
d = cs.copy()
d["health_c"] = d["share_health"] - d["share_health"].mean()
d["dyn_x_health_c"] = d[A] * d["health_c"]
r = m09.fit(d, Y, [A, "health_c", "dyn_x_health_c"] + X, cluster="prov")
a = keep("1 health interaction", A, r); b = keep("1 health interaction", "dyn_x_health_c", r)
print("\n== 1. Dynasty x health share (centred)")
print(f"   dynasty at mean health: {a.beta:+.3f} (se {a.se:.3f}, p {a.p:.3f})")
print(f"   interaction:            {b.beta:+.3f} (se {b.se:.3f}, p {b.p:.3f})")

# ---- 2. IQ terciles: pooled model with tercile-specific dynasty slopes (one covariance matrix -> valid contrast)
d = cs.copy()
d["terc"] = pd.qcut(d["iq_index"], 3, labels=False)
names = ["Low", "Mid", "High"]
for k, nm in enumerate(names):
    d[f"dyn_{nm}"] = d[A] * (d["terc"] == k)
d["t_mid"] = (d["terc"] == 1).astype(float); d["t_high"] = (d["terc"] == 2).astype(float)
r = m09.fit(d, Y, [f"dyn_{n}" for n in names] + ["t_mid", "t_high"] + X, cluster="prov")
print("\n== 2. Dynasty effect by IQ tercile (pooled, tercile-specific slopes, LGU+year FE)")
for nm in names:
    q = keep("2 IQ tercile", f"dyn_{nm}", r, group=nm)
    print(f"   {nm:<5s}: {q.beta:+.3f} (se {q.se:.3f}, p {q.p:.3f})")
c = m09.contrast(r, "dyn_Low", "dyn_High")
print(f"   Low - High: {c['diff']:+.3f} (se {c['se']:.3f}, p {c['p']:.3f}, CI [{c['lo']:.2f}, {c['hi']:.2f}])")
rows.append(dict(test="2 IQ tercile", term="Low - High", beta=c["diff"], se=c["se"], p=c["p"], lo=c["lo"], hi=c["hi"],
                 N=len(d)))

# ---- 3. six component interactions, Bonferroni
print("\n== 3. Dynasty x budget component (components centred; Bonferroni over 6)")
alpha = 0.05 / len(comp); print(f"   Bonferroni threshold: {alpha:.4f}")
for v in comp:
    d = cs.copy()
    d[f"{v}_c"] = d[v] - d[v].mean(); d[f"dxc_{v}"] = d[A] * d[f"{v}_c"]
    r = m09.fit(d, Y, [A, f"{v}_c", f"dxc_{v}"] + X, cluster="prov")
    q = keep("3 component interaction", f"dxc_{v}", r, component=v)
    print(f"   {v:<20s} {q.beta:+9.3f} (se {q.se:.3f}, p {q.p:.3f}) {'***' if q.p < alpha else ''}")

# ---- 4. does dynastic exposure move the IQ index within LGUs?
r = m09.fit(cs, "iq_index", [A, "log_population"], cluster="prov")
q = keep("4 dynasty -> IQ index", A, r)
print(f"\n== 4. Dynasty -> IQ index (LGU+year FE): {q.beta:+.3f} (se {q.se:.3f}, p {q.p:.3f})")

os.makedirs(OUTD, exist_ok=True)
pd.DataFrame(rows).to_csv(os.path.join(OUTD, "institutional_clean.csv"), index=False)
print("\nSaved", os.path.join(OUTD, "institutional_clean.csv"))