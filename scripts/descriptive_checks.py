import sys, os
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
import numpy as np, pandas as pd
from scipy import stats

COL = dict(unit="psgc10", year="fiscal_year", prov="province_code", y="poverty_pi",
           share="dynastic_share_senior", anyd="dynasty_senior_any", fat="fat_same_clan",
           controls=["nta_dependency", "log_total_local_sources", "log_population"])

# ---------------------------------------------------------------- core estimator
def demean(M, groups, tol=1e-10, maxit=500):
    """Absorb one or more FE (list of 1-d label arrays) by alternating projections."""
    M = M.astype(float).copy()
    codes = [pd.factorize(g)[0] for g in groups]
    for _ in range(maxit):
        old = M.copy()
        for c in codes:
            n = c.max() + 1
            cnt = np.bincount(c, minlength=n)[:, None]
            for j in range(M.shape[1]):
                M[:, j] -= (np.bincount(c, weights=M[:, j], minlength=n)[:, None] / cnt)[c, 0]
        if np.abs(M - old).max() < tol:
            break
    return M

def fit(df, y, xcols, fe=("unit", "year"), cluster="unit", extra_dummies_year=None):
    """OLS with absorbed FE. fe in {'unit','year','prov','provyear'}. Returns tidy DataFrame."""
    d = df.dropna(subset=[y] + list(xcols) + [COL["unit"], COL["year"], COL["prov"]]).copy()
    X = d[list(xcols)].astype(float).values
    Y = d[[y]].astype(float).values
    keymap = {"unit": d[COL["unit"]].values, "year": d[COL["year"]].values,
              "prov": d[COL["prov"]].values,
              "provyear": (d[COL["prov"]].astype(str) + "_" + d[COL["year"]].astype(str)).values}
    Z = demean(np.hstack([Y, X]), [keymap[f] for f in fe])
    Yt, Xt = Z[:, :1], Z[:, 1:]
    keep = Xt.std(0) > 1e-9                      # drop regressors fully absorbed (e.g. time-invariant)
    names = [c for c, k in zip(xcols, keep) if k]
    Xt = Xt[:, keep]
    XtX_inv = np.linalg.pinv(Xt.T @ Xt)
    b = XtX_inv @ Xt.T @ Yt
    e = Yt - Xt @ b
    cl = pd.factorize(d[{"unit": COL["unit"], "prov": COL["prov"],
                         "region": "region"}[cluster]])[0]
    G = cl.max() + 1
    meat = np.zeros((Xt.shape[1],) * 2)
    for g in range(G):
        s = Xt[cl == g].T @ e[cl == g]
        meat += s @ s.T
    V = XtX_inv @ meat @ XtX_inv * G / (G - 1)
    se = np.sqrt(np.diag(V)); bb = b.ravel()
    p = 2 * stats.t.sf(np.abs(bb / se), G - 1)
    crit = stats.t.ppf(0.975, G - 1)
    out = pd.DataFrame(dict(term=names, beta=bb, se=se, p=p, lo=bb - crit * se, hi=bb + crit * se,
                            N=len(d), clusters=G))
    out.attrs["V"], out.attrs["names"], out.attrs["G"] = V, names, G
    return out

def contrast(res, a, b):
    """beta_a - beta_b with CI from the stored covariance matrix."""
    n = res.attrs["names"]; V = res.attrs["V"]; G = res.attrs["G"]
    w = np.zeros(len(n)); w[n.index(a)] = 1; w[n.index(b)] -= 1
    est = w @ res.set_index("term").loc[n, "beta"].values
    se = np.sqrt(w @ V @ w); crit = stats.t.ppf(0.975, G - 1)
    return dict(diff=est, se=se, p=2 * stats.t.sf(abs(est / se), G - 1), lo=est - crit * se, hi=est + crit * se)

def show(title, res, terms=None):
    print(f"\n== {title}")
    r = res if terms is None else res[res.term.isin(terms)]
    print(r.round(3).to_string(index=False))
    r = pd.DataFrame(r.to_dict('list'))        # drop attrs so concat works
    return r.assign(model=title)

# ---------------------------------------------------------------- data prep
def build_fat(cand_path):
    """Fat (same-clan) indicator by LGU x fiscal year from candidate data. Missing stays missing."""
    cand = pd.read_parquet(cand_path)
    cand["psgc10"] = cand["psgc10"].astype(str).str.zfill(10)
    print("\n[candidate data] columns:", list(cand.columns))
    win = next((c for c in ("winner", "elected", "won", "is_winner", "is_elected") if c in cand.columns), None)
    if win:
        keep = cand[win].astype(str).str.lower().isin(["1", "1.0", "true", "yes", "y", "w", "won", "elected"])
        print(f"[candidate data] filtering to winners via '{win}': kept {keep.sum():,} of {len(cand):,} rows")
        cand = cand[keep]
    else:
        print("[candidate data] WARNING: no winner column found; ALL candidates (losers too) are used. "
              "Check this - a loser who shares a clan with another candidate would flag 'fat'.")
    m = (cand[cand["pos_norm"] == "MAYOR"][["psgc10", "Year", "Community", "dynasty_flag"]]
         .rename(columns={"Community": "m_comm", "dynasty_flag": "m_dyn"}))
    v = (cand[cand["pos_norm"] == "VICE_MAYOR"][["psgc10", "Year", "Community", "dynasty_flag"]]
         .rename(columns={"Community": "v_comm", "dynasty_flag": "v_dyn"}))
    f = m.merge(v, on=["psgc10", "Year"], how="inner")
    f["fat_same_clan"] = ((f.m_dyn == 1) & (f.v_dyn == 1) & (f.m_comm == f.v_comm)).astype(float)
    f = f.groupby(["psgc10", "Year"], as_index=False)["fat_same_clan"].max()   # one value per LGU-election
    parts = [f.assign(fiscal_year=f["Year"].astype(int) + k) for k in range(3)]
    panel = pd.concat(parts).groupby(["psgc10", "fiscal_year"], as_index=False)["fat_same_clan"].max()
    return panel

def prepare(path):
    df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    df[COL["unit"]] = df[COL["unit"]].astype(str).str.zfill(10)
    if COL["prov"] not in df.columns:           # same rule your audit scripts use
        df[COL["prov"]] = df[COL["unit"]].str[:4]
    df = df[df[COL["year"]].isin([2018, 2021, 2023])].copy()
    if COL["fat"] not in df.columns:            # your parquet does not store it; build as audit_reviewers does
        cpath = os.path.join(os.path.dirname(path), "openhalalan_candidates.parquet")
        fp = build_fat(cpath)
        df = df.merge(fp, left_on=[COL["unit"], COL["year"]], right_on=["psgc10", "fiscal_year"], how="left",
                      suffixes=("", "_f"))
        print(f"[fat] LGU-years with fat information: {df[COL['fat']].notna().sum():,} of {len(df):,}")
    df["region"] = df[COL["unit"]].str[:2]
    # Your stored dynasty_senior_any is 0 whenever the share is missing (checked in the output).
    # Build a clean binary: missing where the share is missing, else share > 0.
    df["any_asCoded"] = df[COL["anyd"]]
    df[COL["anyd"]] = np.where(df[COL["share"]].isna(), np.nan, (df[COL["share"]] > 0).astype(float))
    ok = df[COL["share"]].notna()
    print(f"[binary] share observed: {ok.sum():,} LGU-years; stored any == (share>0) on these: "
          f"{(df.loc[ok,'any_asCoded'] == df.loc[ok, COL['anyd']]).mean():.3f}; "
          f"stored any is 0 where share missing: {(df.loc[~ok,'any_asCoded'].dropna()==0).mean():.3f} "
          f"({df.loc[~ok,'any_asCoded'].notna().sum():,} rows)")
    # three-category dynasty variable; NaN stays NaN (NOT coded as non-dynastic)
    a, f = df[COL["anyd"]], df[COL["fat"]]
    df["cat_none"] = np.where(a.isna(), np.nan, (a == 0).astype(float))
    df["same_clan"] = np.where(a.isna() | f.isna(), np.nan, ((a == 1) & (f == 1)).astype(float))
    df["other_dyn"] = np.where(a.isna() | f.isna(), np.nan, ((a == 1) & (f != 1)).astype(float))
    # island groups: same mapping as your corrected audit_reviewers.py (13 NCR->Luzon, 18 NIR->Visayas)
    luzon = {"01", "02", "03", "04", "05", "13", "14", "17"}; visayas = {"06", "07", "08", "18"}
    mind = {"09", "10", "11", "12", "16", "19"}
    df["island"] = df["region"].map(lambda r: "Luzon" if r in luzon else "Visayas" if r in visayas
                                    else "Mindanao" if r in mind else "Other")
    return df

# ---------------------------------------------------------------- checks
def main(path, outdir):
    os.makedirs(outdir, exist_ok=True)
    df = prepare(path); X = COL["controls"]; Y = COL["y"]; S, A = COL["share"], COL["anyd"]
    out = []

    # 0. sample definition: who has the continuous share?
    t = df.groupby(COL["year"]).agg(rows=(Y, "size"), pov=(Y, "count"), share=(S, "count"), anyd=(A, "count"))
    print("\n== Non-missing counts by year (share is missing where?)\n", t)
    cmp_ = df.assign(has_share=df[S].notna()).groupby("has_share")[[Y, A]].mean()
    print("\n== Poverty and Any-dynastic by whether share is observed\n", cmp_.round(3))

    print("\n== Descriptives, SAE years (non-missing N, mean, sd)")
    print(df[[Y, S, A, COL["fat"]] + X].describe().T[["count", "mean", "std", "min", "max"]].round(3))
    print("   same_clan obs:", int(df.same_clan.sum()), " other_dyn obs:", int(df.other_dyn.sum()),
          " none obs:", int(df.cat_none.sum()), " (three-category sample N:", int(df["other_dyn"].notna().sum()), ")")

    # 1. baseline FE, COMMON sample, three clustering levels
    common = df.dropna(subset=[Y, S, A] + X)
    for cl in ["unit", "prov", "region"]:
        out.append(show(f"A. share, cluster={cl}", fit(common, Y, [S] + X, cluster=cl), [S]))
        out.append(show(f"A. any,   cluster={cl}", fit(common, Y, [A] + X, cluster=cl), [A]))
    out.append(show("A'. any AS CODED in your parquet (missing share -> 0), prov cluster",
                    fit(df, Y, ["any_asCoded"] + X, cluster="prov"), ["any_asCoded"]))
    out.append(show("A''. any, true province-by-year FE (within-province variation)",
                    fit(common, Y, [A] + X, fe=("unit", "provyear"), cluster="prov"), [A]))

    # 2. three categories: none (base) / other_dyn / same_clan, with the contrast
    r = fit(df, Y, ["other_dyn", "same_clan"] + X, cluster="prov")
    out.append(show("B. three-category (prov cluster)", r, ["other_dyn", "same_clan"]))
    c = contrast(r, "same_clan", "other_dyn"); print("   same_clan - other_dyn:", {k: round(v, 3) for k, v in c.items()})
    print("   N same_clan obs:", int(df.same_clan.sum()), "  N other_dyn obs:", int(df.other_dyn.sum()))

    # 3. year-specific associations + tests that they differ; raw and within-year z; both measures
    for M, tag in ((A, "any"), (S, "share")):
        d = df.dropna(subset=[Y, M] + X).copy()
        d["y_z"] = d.groupby(COL["year"])[Y].transform(lambda s_: (s_ - s_.mean()) / s_.std())
        ycols = []
        for yy in (2018, 2021, 2023):
            d[f"{tag}_{yy}"] = d[M] * (d[COL["year"]] == yy); ycols.append(f"{tag}_{yy}")
        for dep in (Y, "y_z"):
            r = fit(d, dep, ycols + X, cluster="prov")
            out.append(show(f"C. {tag} x year, dep={dep}", r, ycols))
            for a_, b_ in ((ycols[2], ycols[0]), (ycols[1], ycols[0]), (ycols[2], ycols[1])):
                c = contrast(r, a_, b_); print(f"   {a_}-{b_}: diff={c['diff']:.3f} se={c['se']:.3f} p={c['p']:.3f}")

    # 4. baseline-adjusted FE: poverty tercile in first round x year FE
    base = df[df[COL["year"]] == df[COL["year"]].min()].set_index(COL["unit"])[Y]
    d = df.copy(); d["base_pov"] = d[COL["unit"]].map(base)
    d["terc"] = pd.qcut(d["base_pov"], 3, labels=False, duplicates="drop")
    tc = []
    for k in (1, 2):
        for yy in (2021, 2023):
            n = f"t{k}_y{yy}"; d[n] = ((d["terc"] == k) & (d[COL["year"]] == yy)).astype(float); tc.append(n)
    out.append(show("D. any, baseline tercile x year (prov cluster)", fit(d, Y, [A] + X + tc, cluster="prov"), [A]))
    out.append(show("D'. share, baseline tercile x year", fit(d, Y, [S] + X + tc, cluster="prov"), [S]))

    # 5. long difference with NaN-safe transition groups (binary measure, both rounds observed)
    lo, hi = df[COL["year"]].min(), df[COL["year"]].max()
    w = df.pivot_table(index=COL["unit"], columns=COL["year"], values=[Y, A], aggfunc="first")
    g = pd.DataFrame({"pov0": w[(Y, lo)], "pov1": w[(Y, hi)], "d0": w[(A, lo)], "d1": w[(A, hi)]}).dropna()
    g["dpov"] = g.pov1 - g.pov0
    g["grp"] = np.select([(g.d0 == 1) & (g.d1 == 1), (g.d0 == 1) & (g.d1 == 0),
                          (g.d0 == 0) & (g.d1 == 1)], ["Always", "Exit", "Entry"], "Never")
    prov = df.drop_duplicates(COL["unit"]).set_index(COL["unit"])[COL["prov"]]
    g["province_code"] = g.index.map(prov); g["psgc10"] = g.index
    print("\n== E. transition groups (NaN-safe), N and means\n",
          g.groupby("grp")[["pov0", "pov1", "dpov"]].agg(["count", "mean"]).round(2))
    g2 = g.assign(fiscal_year=0, region=g.index.astype(str).str.zfill(10).str[:2])
    for gp in ("Entry", "Always", "Exit"):
        s = g2[g2.grp.isin([gp, "Never"])].copy(); s["is_g"] = (s.grp == gp).astype(float)
        r = fit(s, "pov0", ["is_g"], fe=("year",), cluster="prov")
        out.append(show(f"E. {gp} minus Never, baseline poverty (prov cluster)", r))
    g2["dyn0"] = g2.d0
    g2["lag"] = g2.pov0
    out.append(show("F. long diff: dpov ~ dyn0 (no baseline)", fit(g2, "dpov", ["dyn0"], fe=("year",), cluster="prov")))
    out.append(show("F'. long diff: dpov ~ dyn0 + pov0", fit(g2, "dpov", ["dyn0", "lag"], fe=("year",), cluster="prov")))

    # 6. island groups, separate regressions per island (differences between islands are NOT tested here)
    for M, tag in ((A, "any"), (S, "share")):
        for isl, s_ in df.dropna(subset=[Y, M] + X).groupby("island"):
            if isl != "Other" and s_[COL["prov"]].nunique() >= 3:
                out.append(show(f"G. {tag} within {isl}", fit(s_, Y, [M] + X, cluster="prov"), [M]))

    # 6b. are the island differences themselves distinguishable? (Luzon = reference)
    for M, tag in ((A, "any"), (S, "share")):
        d = df.dropna(subset=[Y, M] + X); d = d[d.island != "Other"].copy()
        d[f"{tag}_x_Mind"] = d[M] * (d.island == "Mindanao"); d[f"{tag}_x_Vis"] = d[M] * (d.island == "Visayas")
        out.append(show(f"G'. {tag}: Luzon effect and differences from Luzon (province cluster)",
                        fit(d, Y, [M, f"{tag}_x_Mind", f"{tag}_x_Vis"] + X, cluster="prov"),
                        [M, f"{tag}_x_Mind", f"{tag}_x_Vis"]))

    # 7. HUC / balanced panel filters if flags exist
    if "is_huc" in df:
        out.append(show("H. excluding HUCs", fit(df[df.is_huc != 1], Y, [A] + X, cluster="prov"), [A]))
    cnt = df.dropna(subset=[Y, A] + X).groupby(COL["unit"]).size()
    bal = df[df[COL["unit"]].isin(cnt[cnt == df[COL["year"]].nunique()].index)]
    out.append(show("H'. balanced panel", fit(bal, Y, [A] + X, cluster="prov"), [A]))

    # 8. moderation by fiscal variables (replaces audit_institutional_resilience.py; clean binary, province cluster)
    for M, tag in ((A, "any"), (S, "share")):
        d = df.dropna(subset=[Y, M] + X).copy()
        for v, nm in (("nta_dependency", "NTA"), ("log_total_local_sources", "LocalSrc")):
            d[f"{v}_c"] = d[v] - d[v].mean()                      # centred on the regression sample
            d[f"{tag}_x_{nm}"] = d[M] * d[f"{v}_c"]
            others = [c for c in X if c != v]
            r = fit(d, Y, [M, f"{tag}_x_{nm}", f"{v}_c"] + others, cluster="prov")
            out.append(show(f"I. {tag} x {nm} (centred)", r, [M, f"{tag}_x_{nm}", f"{v}_c"]))

    # 9. optional: other dynasty measures, on the SAME rows where the share is observed
    for extra in ("mayor_dyn", "dynasty_mayor_any", "dynasty_gov_any", "gov_dyn"):
        if extra in df.columns:
            d = df[df[S].notna()]
            out.append(show(f"J. {extra} (share-observed rows), prov cluster", fit(d, Y, [extra] + X, cluster="prov"), [extra]))

    pd.concat(out).to_csv(os.path.join(outdir, "descriptive_checks.csv"), index=False)
    print("\nSaved", os.path.join(outdir, "descriptive_checks.csv"))

if __name__ == "__main__":
    data = sys.argv[1] if len(sys.argv) > 1 else "data/processed/panel_dynasty_funds.parquet"
    outd = sys.argv[2] if len(sys.argv) > 2 else "results"
    main(data, outd)