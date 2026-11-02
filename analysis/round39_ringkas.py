#!/usr/bin/env python3
"""Summary of round 39: paired difference delta = AUC(transform, adjustment) - AUC(same transform, none) per cell and classifier; rules R2 and R1 (PRAREGISTRASI_putaran39.md). Uses explicit floats. CPU.
Usage: python round39_ringkas.py <stage>"""
import os, sys, glob, numpy as np, pandas as pd
from scipy import stats
H = os.path.dirname(os.path.abspath(__file__)); stage = sys.argv[1] if len(sys.argv) > 1 else "pilot"
D = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{H}/hasil39/{stage}_*.csv"))], ignore_index=True)
for c, v in (("sig", 1.0), ("cfrac", 0.01), ("cscale", 100.0)):
    if c not in D.columns: D[c] = v
    D[c] = D[c].fillna(v).astype(float)
for c in ("auc", "kurtosis", "meta_auc", "s"): D[c] = D[c].astype(float)
keys = ["dist", "scen", "n", "m", "p", "imb", "s", "sig", "cfrac", "cscale", "clf"]; base = D[D.adjust == "none"].rename(columns={"auc": "auc_none"})[keys + ["rep", "transform", "auc_none"]]
TREEC = ["RF", "ET", "XGB"]; Dn = D[D.adjust != "none"]; baseT = D[(D.adjust == "none") & (D["transform"] == "raw") & D.clf.isin(TREEC)].rename(columns={"auc": "auc_none"})[keys + ["rep", "auc_none"]]
A = pd.concat([Dn[~Dn.clf.isin(TREEC)].merge(base, on=keys + ["rep", "transform"], how="left"), Dn[Dn.clf.isin(TREEC)].merge(baseT, on=keys + ["rep"], how="left")], ignore_index=True); A["delta"] = A.auc - A.auc_none; rows = []
for k, g in A.groupby(keys + ["transform", "adjust"]):
    d = g.delta.values.astype(float); n = len(d)
    if n < 3: continue
    se = d.std(ddof=1) / np.sqrt(n); m = d.mean(); a = g.auc.values.astype(float); r2 = "tidak ada bukti" if (m <= 0.01 and m + 1.645 * se <= 0.03) else ("kebocoran" if m > 0.03 else "kecil/tidak pasti")
    rows.append(dict(zip(keys + ["transform", "adjust"], k), n_rep=n, auc_none=g.auc_none.mean(), auc_adj=a.mean(), delta=m, se=se, delta_p95=m + 1.645 * se, t_p=stats.ttest_1samp(d, 0).pvalue, R2=r2,
                     R1_abs=("tidak ada bukti" if (a.mean() <= 0.52 and a.mean() + 1.645 * a.std(ddof=1) / np.sqrt(n) <= 0.54) else ("kebocoran" if a.mean() > 0.55 else "kecil/tidak pasti")), kurtosis=g["kurtosis"].mean(), meta_auc=g.meta_auc.mean()))
S = pd.DataFrame(rows); S.to_csv(f"{H}/hasil39/ringkasan_{stage}.csv", index=False); pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
print(S[["dist", "scen", "s", "sig", "cfrac", "cscale", "n", "m", "p", "imb", "clf", "transform", "adjust", "n_rep", "auc_none", "auc_adj", "delta", "R2", "R1_abs", "kurtosis"]].round(3).to_string(index=False)) if len(S) < 400 else print(f"{len(S)} rows; see ringkasan_{stage}.csv")
