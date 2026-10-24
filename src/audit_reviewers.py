import sys
sys.stdout.reconfigure(encoding="utf-8")

from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats as st

OUT_DATA = Path("data/processed")
OUT = Path("output")
OUT.mkdir(parents=True, exist_ok=True)

BASE = "nta_dependency + log_total_local_sources + log_population"
SAE_YEARS = [2018, 2021, 2023]


def _scalar(x):
    """Robust scalar extraction: handles 0-d, 1-element, and 2-D arrays."""
    return float(np.asarray(x).ravel()[0])


def _contrast_ci(w):
    """Return (lo, hi) from a statsmodels ContrastResults, whether
    conf_int is a method (current) or an attribute (older)."""
    ci = w.conf_int() if callable(getattr(w, "conf_int", None)) else w.conf_int
    a = np.asarray(ci).ravel()
    return float(a[0]), float(a[1])


# -------------------------------------------------------------------------
# Load + correct keys
# -------------------------------------------------------------------------
df = pd.read_parquet(OUT_DATA / "panel_dynasty_funds.parquet")
df["psgc10"] = df["psgc10"].astype(str).str.zfill(10)

# PSGC layout: RR(2) PP(2) M(1) BBBB(4)
df["region"]        = df["psgc10"].str[:2]
df["province_code"] = df["psgc10"].str[:4]     # <-- FIX A (carried from v2)

# Sanity check: report many-to-one province names, but do NOT warn on the
# expected HUC-vs-mother-province collision. The 4-char key is RR+PP, so a
# name mismatch is expected whenever an HUC has its own PSGC province code
# (e.g. 0600 Iloilo City vs 0601 Iloilo). Flag only if a code maps to
# >2 distinct names, which would indicate a real key bug.
chk = (df.dropna(subset=["province"])
         .query("province != ''")
         .groupby("province_code")["province"].nunique())
bad = chk[chk > 2]               # >2 names = suspicious; 2 names = HUC collision
if len(bad):
    print(f"[WARN] province_code -> province has >2 names for {len(bad)} codes: "
          f"{bad.index.tolist()[:10]}")
else:
    print(f"[OK] province_code is well-formed on {len(chk)} codes "
          f"({int((chk == 2).sum())} HUC-vs-mother-province name collisions, expected)")

n_prov = df["province_code"].nunique()
print(f"[INFO] unique province_code (4-char) = {n_prov}; "
      f"unique region (2-char) = {df['region'].nunique()}")

df["income_class_num"] = (
    df["income_class"].astype(str)
      .str.replace("*", "", regex=False)
      .str.extract(r"(\d+)", expand=False).astype(float))

REQ = ["psgc10", "fiscal_year", "province_code", "region",
       "nta_dependency", "log_total_local_sources", "log_population"]

sae = df[df["fiscal_year"].isin(SAE_YEARS)].copy()
print(f"\nSAE panel: {len(sae):,} obs, {sae['psgc10'].nunique():,} LGUs, "
      f"{sae['province_code'].nunique():,} province codes")
print(f"  NCR obs in SAE: {int((sae['region']=='13').sum()):,}")


def fit(data, outcome, dv, fe="lgu_year", extra="",
        cluster="psgc10", drop=None):
    need = [outcome, dv, cluster] + REQ
    if drop:
        need += drop
    sub = data.dropna(subset=[c for c in need if c in data.columns]).copy()
    if len(sub) < 100:
        return None
    fe_term = {"lgu_year":  " + C(psgc10) + C(fiscal_year)",
               "lgu":       " + C(psgc10)",
               "year":      " + C(fiscal_year)",
               "prov_year": " + C(province_code) + C(fiscal_year)",
               "none":      ""}[fe]
    formula = f"{outcome} ~ {dv} + {BASE} {extra}{fe_term}"
    try:
        m = smf.ols(formula, data=sub).fit(
            cov_type="cluster",
            cov_kwds={"groups": sub[cluster].values})
    except Exception as e:
        print(f"  FAILED: {str(e)[:80]}")
        return None
    ci = m.conf_int().loc[dv]
    return {
        "label":      f"{dv} | {fe} | cl={cluster}",
        "dv":         dv, "fe": fe, "cluster": cluster,
        "beta":       float(m.params[dv]),
        "se":         float(m.bse[dv]),
        "t":          float(m.tvalues[dv]),
        "p":          float(m.pvalues[dv]),
        "ci_lo":      float(ci[0]),
        "ci_hi":      float(ci[1]),
        "N":          int(m.nobs),
        "n_clusters": int(sub[cluster].nunique()),
        "R2":         float(m.rsquared),
        "n_lgu":      int(sub["psgc10"].nunique()),
    }


def line(r):
    if r is None:
        return "  (skipped)"
    return (f"  {r['label']:<58s}  N={r['N']:>5,}  "
            f"beta={r['beta']:>+8.4f}  se={r['se']:>7.4f}  "
            f"t={r['t']:>6.2f}  p={r['p']:>6.3f}  "
            f"clusters={r['n_clusters']:>3}")


records = []
def keep(r):
    if r:
        records.append(r)
    return r


# =========================================================================
# 1. INFERENCE
# =========================================================================
print("\n" + "=" * 90)
print("1. INFERENCE: province clustering / province x year FE / mayor-only")
print("=" * 90)

DVS = ["dynastic_share_senior", "dynasty_senior_any"]
for dv in DVS:
    print(f"\n--- {dv} ---")
    print(line(keep(fit(sae, "poverty_pi", dv, "lgu_year", cluster="psgc10"))))
    print(line(keep(fit(sae, "poverty_pi", dv, "lgu_year", cluster="region"))))
    print(line(keep(fit(sae, "poverty_pi", dv, "lgu_year", cluster="province_code"))))

for dv in DVS:
    print(f"\n--- {dv} | province x year FE (cluster province) ---")
    sub = sae.dropna(subset=[dv, "poverty_pi", "province_code"] + REQ).copy()
    formula = (f"poverty_pi ~ {dv} + {BASE} + C(province_code) + C(fiscal_year)")
    m = smf.ols(formula, data=sub).fit(
        cov_type="cluster",
        cov_kwds={"groups": sub["province_code"].values})
    r = {
        "label":      f"{dv} | prov_year FE | cl=province",
        "dv":         dv, "fe": "prov_year", "cluster": "province_code",
        "beta":       float(m.params[dv]),
        "se":         float(m.bse[dv]),
        "t":          float(m.tvalues[dv]),
        "p":          float(m.pvalues[dv]),
        "ci_lo":      float(m.conf_int().loc[dv, 0]),
        "ci_hi":      float(m.conf_int().loc[dv, 1]),
        "N":          int(m.nobs),
        "n_clusters": int(sub["province_code"].nunique()),
        "R2":         float(m.rsquared),
        "n_lgu":      int(sub["psgc10"].nunique()),
    }
    print(line(r)); keep(r)

print("\n--- building mayor-only dynasty measure ---")
cand = pd.read_parquet(OUT_DATA / "openhalalan_candidates.parquet")
cand["psgc10"] = cand["psgc10"].astype(str).str.zfill(10)
may = (cand[cand["pos_norm"] == "MAYOR"]
       [["psgc10", "Year", "dynasty_flag"]]
       .rename(columns={"dynasty_flag": "mayor_dyn"}))
may["Year"] = may["Year"].astype(int)
rows = []
for _, r in may.iterrows():
    e = int(r["Year"])
    for fy in range(e, e + 3):
        if 2004 <= fy <= 2024:
            rows.append({"psgc10": r["psgc10"], "fiscal_year": fy,
                         "mayor_dyn": int(r["mayor_dyn"])})
mayor_panel = (pd.DataFrame(rows).drop_duplicates(["psgc10", "fiscal_year"]))
sae_m = sae.merge(mayor_panel, on=["psgc10", "fiscal_year"], how="left")
sae_m["mayor_dyn"] = sae_m["mayor_dyn"].fillna(0)
print(f"  mayor_dyn mean: {sae_m['mayor_dyn'].mean():.3f}")
for fe in ["lgu_year", "lgu"]:
    for cl in ["psgc10", "province_code"]:
        print(line(keep(fit(sae_m, "poverty_pi", "mayor_dyn", fe, cluster=cl))))


# =========================================================================
# 2. FAT x SENIOR INTERACTION
# =========================================================================
print("\n" + "=" * 90)
print("2. FAT x SENIOR INTERACTION (both main effects, single dropna)")
print("=" * 90)
cand2 = cand.copy()
may2 = (cand2[cand2["pos_norm"] == "MAYOR"]
        [["psgc10", "Year", "Community", "dynasty_flag"]]
        .rename(columns={"Community": "mayor_comm",
                         "dynasty_flag": "mayor_dyn"}))
vmay2 = (cand2[cand2["pos_norm"] == "VICE_MAYOR"]
         [["psgc10", "Year", "Community", "dynasty_flag"]]
         .rename(columns={"Community": "vmayor_comm",
                          "dynasty_flag": "vmayor_dyn"}))
fat = may2.merge(vmay2, on=["psgc10", "Year"], how="inner")
fat["fat_same_clan"] = (
    (fat["mayor_dyn"] == 1) & (fat["vmayor_dyn"] == 1) &
    (fat["mayor_comm"] == fat["vmayor_comm"])).astype(int)
fat_rows = []
for _, r in fat.iterrows():
    e = int(r["Year"])
    for fy in range(e, e + 3):
        if 2004 <= fy <= 2024:
            fat_rows.append({"psgc10": r["psgc10"], "fiscal_year": fy,
                             "fat_same_clan": r["fat_same_clan"]})
fat_panel = (pd.DataFrame(fat_rows).drop_duplicates(["psgc10", "fiscal_year"]))
sae_f = sae.merge(fat_panel, on=["psgc10", "fiscal_year"], how="left")
sae_f["fat_same_clan"] = sae_f["fat_same_clan"].fillna(0)
sae_f["fat_x_senior"]  = sae_f["fat_same_clan"] * sae_f["dynastic_share_senior"]

sub = sae_f.dropna(subset=["poverty_pi", "fat_same_clan",
                            "dynastic_share_senior", "fat_x_senior",
                            "psgc10", "fiscal_year"] + REQ).copy()
formula = (f"poverty_pi ~ fat_same_clan + dynastic_share_senior "
           f"+ fat_x_senior + {BASE} + C(psgc10) + C(fiscal_year)")
m = smf.ols(formula, data=sub).fit(
    cov_type="cluster",
    cov_kwds={"groups": sub["psgc10"].values})

g = sub.groupby("psgc10")
fat_within_sd = g["fat_same_clan"].std().mean()
fat_mean_lgu  = g["fat_same_clan"].mean()
n_switch_lgu  = int(((fat_mean_lgu > 0) & (fat_mean_lgu < 1)).sum())
print(f"  N={int(m.nobs):,}  R2={m.rsquared:.4f}")
print(f"  var(dynastic_share_senior) in sample: {sub['dynastic_share_senior'].var():.4f}")
print(f"  mean(fat_same_clan) in sample:        {sub['fat_same_clan'].mean():.4f}")
print(f"  within-LGU SD of fat_same_clan (avg): {fat_within_sd:.4f}")
print(f"  LGUs where fat_same_clan switches:    {n_switch_lgu} / {sub['psgc10'].nunique()}")
for term in ["fat_same_clan", "dynastic_share_senior", "fat_x_senior"]:
    print(f"  {term:<28s} beta={m.params[term]:>+8.4f}  "
          f"se={m.bse[term]:>7.4f}  t={m.tvalues[term]:>6.2f}  "
          f"p={m.pvalues[term]:>6.3f}")
r = {"label": "fat_x_senior (corrected)",
     "dv": "fat_x_senior", "fe": "lgu_year", "cluster": "psgc10",
     "beta":  float(m.params["fat_x_senior"]),
     "se":    float(m.bse["fat_x_senior"]),
     "t":     float(m.tvalues["fat_x_senior"]),
     "p":     float(m.pvalues["fat_x_senior"]),
     "ci_lo": float(m.conf_int().loc["fat_x_senior", 0]),
     "ci_hi": float(m.conf_int().loc["fat_x_senior", 1]),
     "N":     int(m.nobs),
     "n_clusters": int(sub["psgc10"].nunique()),
     "R2":    float(m.rsquared),
     "n_lgu": int(sub["psgc10"].nunique())}
keep(r)


# =========================================================================
# 3. ENTRY / EXIT + 2018 ENTRY-vs-NEVER
# =========================================================================
print("\n" + "=" * 90)
print("3. ENTRY / EXIT + 2018 ENTRY-vs-NEVER TEST")
print("=" * 90)
pivot = (sae.sort_values(["psgc10", "fiscal_year"])
            .pivot_table(index="psgc10", columns="fiscal_year",
                         values=["poverty_pi", "dynastic_share_senior"],
                         aggfunc="mean"))
has18     = pivot[("dynastic_share_senior", 2018)] > 0.01
has_later = ((pivot[("dynastic_share_senior", 2021)] > 0.01) |
             (pivot[("dynastic_share_senior", 2023)] > 0.01))
groups = {
    "Entry":  has18.eq(False) & has_later.eq(True),
    "Never":  has18.eq(False) & has_later.eq(False),
    "Always": has18.eq(True)  & has_later.eq(True),
    "Exit":   has18.eq(True)  & has_later.eq(False),
}
t2 = []
for label, mask in groups.items():
    t2.append({
        "group": label, "n_lgu": int(mask.sum()),
        "poverty_2018_mean": float(pivot[("poverty_pi", 2018)][mask].mean()),
        "poverty_2023_mean": float(pivot[("poverty_pi", 2023)][mask].mean()),
    })
t2 = pd.DataFrame(t2)
t2.to_csv(OUT / "audit_transition_groups.csv", index=False)
print(t2.to_string(index=False))

e18 = pivot[("poverty_pi", 2018)][groups["Entry"]].dropna()
n18 = pivot[("poverty_pi", 2018)][groups["Never"]].dropna()
diff = e18.mean() - n18.mean()
se   = np.sqrt(e18.var(ddof=1) / len(e18) + n18.var(ddof=1) / len(n18))
tstat = diff / se
pval  = 2 * st.t.sf(abs(tstat), df=len(e18) + len(n18) - 2)
ci    = st.t.interval(0.95, df=len(e18) + len(n18) - 2, loc=diff, scale=se)
print(f"\n  2018 Entry - Never: {diff:+.3f} pp  "
      f"se={se:.3f}  t={tstat:.2f}  p={pval:.3f}  "
      f"95% CI [{ci[0]:+.3f}, {ci[1]:+.3f}]")
records.append({"label": "2018 Entry - Never gap",
                "dv": "entry_gap", "fe": "none", "cluster": "none",
                "beta": float(diff), "se": float(se),
                "t": float(tstat), "p": float(pval),
                "ci_lo": float(ci[0]), "ci_hi": float(ci[1]),
                "N": int(len(e18) + len(n18)),
                "n_clusters": 0, "R2": np.nan, "n_lgu": 0})

sae_t = sae.merge(
    pd.DataFrame({k: v for k, v in groups.items()}).astype(int),
    left_on="psgc10", right_index=True, how="left")
sae_t[["Entry", "Never", "Always", "Exit"]] = \
    sae_t[["Entry", "Never", "Always", "Exit"]].fillna(0)
for cat in ["Entry", "Exit"]:
    other = "Never" if cat == "Entry" else "Always"
    sub = sae_t[sae_t[cat].eq(1) | sae_t[other].eq(1)].copy()
    sub["is_cat"] = sub[cat].astype(int)
    prov_means = sub.groupby("province_code")["is_cat"].mean()
    n_varies = int(((prov_means > 0) & (prov_means < 1)).sum())
    print(f"\n  [{cat} vs {other}] within-province is_cat varies in "
          f"{n_varies} / {len(prov_means)} provinces")
    print(line(keep(fit(sub, "poverty_pi", "is_cat", "lgu_year",
                         cluster="province_code"))))


# =========================================================================
# 4. DIRECT THREE-CATEGORY (none / other / same-clan)
# =========================================================================
print("\n" + "=" * 90)
print("4. DIRECT FAT vs THIN (three categories, one regression)")
print("=" * 90)
sae_c = sae_f.copy()
sae_c["cat"] = np.select(
    [sae_c["fat_same_clan"].eq(1),
     sae_c["dynastic_share_senior"].gt(0) & sae_c["fat_same_clan"].eq(0)],
    ["same_clan", "other_dyn"], default="none")
sae_c["other_dyn"] = sae_c["cat"].eq("other_dyn").astype(int)
sae_c["same_clan"] = sae_c["cat"].eq("same_clan").astype(int)
sub = sae_c.dropna(subset=["poverty_pi", "other_dyn", "same_clan",
                            "psgc10", "fiscal_year"] + REQ).copy()
m = smf.ols(f"poverty_pi ~ other_dyn + same_clan + {BASE} "
            f"+ C(psgc10) + C(fiscal_year)",
            data=sub).fit(cov_type="cluster",
                          cov_kwds={"groups": sub["psgc10"].values})
print(f"  N={int(m.nobs):,}  R2={m.rsquared:.4f}")
for t in ["other_dyn", "same_clan"]:
    print(f"  {t:<12s} beta={m.params[t]:>+8.4f}  se={m.bse[t]:>7.4f}  "
          f"t={m.tvalues[t]:>6.2f}  p={m.pvalues[t]:>6.3f}")

# FIX B (v3): w.conf_int is a METHOD here, not an attribute.
w = m.t_test("same_clan - other_dyn = 0")
beta = _scalar(w.effect)
sd   = _scalar(w.sd)
tval = _scalar(w.tvalue)
pval = _scalar(w.pvalue)
ci_lo, ci_hi = _contrast_ci(w)
print(f"  same_clan - other_dyn: beta={beta:>+8.4f}  se={sd:>7.4f}  "
      f"t={tval:>6.2f}  p={pval:>6.3f}  "
      f"95% CI [{ci_lo:+.4f}, {ci_hi:+.4f}]")
records.append({"label": "same_clan - other_dyn",
                "dv": "same_clan_minus_other_dyn", "fe": "lgu_year",
                "cluster": "psgc10",
                "beta": beta, "se": sd, "t": tval, "p": pval,
                "ci_lo": ci_lo, "ci_hi": ci_hi,
                "N": int(m.nobs),
                "n_clusters": int(sub["psgc10"].nunique()),
                "R2": float(m.rsquared),
                "n_lgu": int(sub["psgc10"].nunique())})


# =========================================================================
# 5. YEAR INTERACTIONS + SCALINGS
# =========================================================================
print("\n" + "=" * 90)
print("5. YEAR INTERACTIONS + SCALINGS")
print("=" * 90)
for dv in DVS:
    print(f"\n--- {dv} ---")
    for scale, col in [("raw", "poverty_pi"), ("pooled_z", "poverty_pz"),
                       ("log", "poverty_log"), ("rank", "poverty_rank")]:
        sub = sae.copy()
        if col not in sub.columns:
            if col == "poverty_pz":
                sub["poverty_pz"] = (sub["poverty_pi"]
                    - sub["poverty_pi"].mean()) / sub["poverty_pi"].std()
            elif col == "poverty_log":
                sub["poverty_log"] = np.log1p(sub["poverty_pi"])
            elif col == "poverty_rank":
                sub["poverty_rank"] = sub["poverty_pi"].rank(pct=True)
        sub = sub.dropna(subset=[col, dv, "psgc10", "fiscal_year"] + REQ).copy()
        formula = (f"{col} ~ {dv} * C(fiscal_year) + {BASE} + C(psgc10)")
        m = smf.ols(formula, data=sub).fit(
            cov_type="cluster",
            cov_kwds={"groups": sub["psgc10"].values})
        inter = [t for t in m.params.index if ":" in t and dv in t]
        for t in inter:
            yr = t.split("[T.")[-1].rstrip("]")
            print(f"  {scale:<9s} x {yr}: beta={m.params[t]:>+9.4f}  "
                  f"se={m.bse[t]:>7.4f}  t={m.tvalues[t]:>6.2f}  "
                  f"p={m.pvalues[t]:>6.3f}  "
                  f"95% CI [{m.conf_int().loc[t,0]:+.4f}, "
                  f"{m.conf_int().loc[t,1]:+.4f}]")
            records.append({"label": f"{dv} | {scale} x {yr}",
                            "dv": dv, "fe": "lgu_year", "cluster": "psgc10",
                            "beta": float(m.params[t]),
                            "se":   float(m.bse[t]),
                            "t":    float(m.tvalues[t]),
                            "p":    float(m.pvalues[t]),
                            "ci_lo": float(m.conf_int().loc[t, 0]),
                            "ci_hi": float(m.conf_int().loc[t, 1]),
                            "N": int(m.nobs),
                            "n_clusters": int(sub["psgc10"].nunique()),
                            "R2": float(m.rsquared),
                            "n_lgu": int(sub["psgc10"].nunique())})


# =========================================================================
# 6. NO-OP ROBUSTNESS ROWS + RANK DEFICIENCY
# =========================================================================
print("\n" + "=" * 90)
print("6. NO-OP ROBUSTNESS ROWS: pre/post N")
print("=" * 90)
counts = sae.groupby("psgc10")["fiscal_year"].nunique()
balanced_ids = counts[counts == 3].index
sae_bal = sae[sae["psgc10"].isin(balanced_ids)]
print(f"  pre  balanced: N={len(sae):,}  LGUs={sae['psgc10'].nunique():,}")
print(f"  post balanced: N={len(sae_bal):,}  LGUs={sae_bal['psgc10'].nunique():,}")
no_ncr = sae[sae["region"] != "13"]
print(f"  pre  no NCR:  N={len(sae):,}  NCR rows={int((sae['region']=='13').sum()):,}")
print(f"  post no NCR:  N={len(no_ncr):,}  LGUs={no_ncr['psgc10'].nunique():,}")
if "is_independent_city" in sae.columns:
    ind = sae["is_independent_city"].fillna(False).astype(bool)
    print(f"  pre  no HUC:  N={len(sae):,}  HUC rows={int(ind.sum()):,}")
    print(f"  post no HUC:  N={len(sae[~ind]):,}  LGUs={sae[~ind]['psgc10'].nunique():,}")
else:
    print("  is_independent_city not present — R7 is a no-op in 04.")

print("\n  rank check: poverty_pi ~ dynastic_share_senior + income_class_num "
      "+ C(psgc10) + C(fiscal_year)")
sub = sae.dropna(subset=["poverty_pi", "dynastic_share_senior",
                          "income_class_num", "psgc10", "fiscal_year"]
                  + REQ).copy()
m = smf.ols("poverty_pi ~ dynastic_share_senior + income_class_num "
            "+ " + BASE + " + C(psgc10) + C(fiscal_year)",
            data=sub).fit()
print(f"  rank: {m.model.rank} / {m.model.exog.shape[1]}  "
      f"(equal => full rank)")
print(f"  income_class_num coefficient: "
      f"{m.params.get('income_class_num', np.nan):.6f}")


# =========================================================================
# Write
# =========================================================================
out = pd.DataFrame(records)
csv_path = OUT / "audit_reviewers.csv"
out.to_csv(csv_path, index=False)
txt_path = OUT / "audit_reviewers.txt"
with open(txt_path, "w", encoding="utf-8") as f:
    f.write(out.to_string(index=False))
print(f"\nWrote {csv_path}")
print(f"Wrote {txt_path}")