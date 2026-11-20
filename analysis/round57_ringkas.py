#!/usr/bin/env python3
"""Putaran 57: ringkasan sesuai PRAREGISTRASI_putaran57_60.md (di-hash sebelum hasil dilihat). Tidak melatih model."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
from scipy import stats
from round49_kalibrasi import decide, THR
FC = [f"fold_auc_{i}" for i in range(5)]; O = os.path.join(HERE, "hasil57"); LAB = {0: "no evidence", 1: "uncertain", 2: "leakage"}
dec = lambda d: LAB[int(decide(np.array([d.mean()]), np.array([d.std(ddof=1) / np.sqrt(len(d))]), THR["R2"]["asli"])[0])]
def stored(fam, cohort):
    if cohort == "tuep":
        a = pd.read_csv(os.path.join(HERE, "hasil42", f"{fam}.csv")); a = a[a.cond == "H1"]; return pd.concat([a, pd.read_csv(os.path.join(HERE, "hasil50", f"A_{fam}.csv"))])
    return pd.read_csv(os.path.join(HERE, "hasil50", f"B_{fam}.csv"))
parts = []
for fam in ("RF", "XGB"):
    new = pd.read_csv(os.path.join(O, f"{fam}.csv"))
    for cohort in ("tuep", "emotiv"):
        s = stored(fam, cohort).copy(); s["cohort"] = cohort; s["fam"] = fam; parts.append(s[["fam", "cohort", "rep", "gamma", "transform", "adjust"] + FC])
    parts.append(new[["fam", "cohort", "rep", "gamma", "transform", "adjust"] + FC])
parts.append(pd.read_csv(os.path.join(O, "LRw.csv"))[["fam", "cohort", "rep", "gamma", "transform", "adjust"] + FC])
D = pd.concat(parts, ignore_index=True); D["fm"] = 100 * D[FC].mean(1); D["pipe"] = D["transform"] + "-" + D["adjust"]
D = D.drop_duplicates(["fam", "cohort", "rep", "gamma", "pipe"], keep="last")
# invariansi pohon: T-none vs raw-none
for fam in ("RF", "XGB"):
    W = D[D.fam == fam].pivot_table(index=["cohort", "rep", "gamma"], columns="pipe", values="fm")
    for t in ("symlog", "log", "rank"):
        d = (W[f"{t}-none"] - W["raw-none"]).dropna().abs(); print(f"{fam} {t}-none vs raw-none: unit {len(d)}, identik {int((d < 1e-9).sum())}, selisih maks {d.max():.3f} pp, rata-rata |selisih| {d.mean():.3f} pp")
rows = []
for (fam, cohort, g), G in D.groupby(["fam", "cohort", "gamma"]):
    W = G.pivot_table(index="rep", columns="pipe", values="fm")
    for t in ("symlog", "log", "rank"):
        c = [p for p in ("raw-none", "raw-pat", f"{t}-none", f"{t}-pat") if p in W.columns]
        if len(c) < 4: continue
        Q = W[c].dropna(); a = Q[f"{t}-pat"] - Q[f"{t}-none"]; tr = Q[f"{t}-none"] - Q["raw-none"]; ia = a - (Q["raw-pat"] - Q["raw-none"])
        ci = stats.t.ppf(0.975, len(a) - 1) * a.std(ddof=1) / np.sqrt(len(a))
        rows.append(dict(fam=fam, cohort=cohort, gamma=g, T=t, n=len(Q), raw=Q["raw-none"].mean(), rawA=Q["raw-pat"].mean(), Tn=Q[f"{t}-none"].mean(), TA=Q[f"{t}-pat"].mean(),
                         adj_in_T=a.mean(), adj_lo=a.mean() - ci, adj_hi=a.mean() + ci, R2_adj_in_T=dec(a / 100), T_effect=tr.mean(), interaction=ia.mean(), old_TA_minus_raw=(Q[f"{t}-pat"] - Q["raw-none"]).mean()))
T = pd.DataFrame(rows); T.to_csv(os.path.join(O, "empat_sel.csv"), index=False); pd.set_option("display.width", 250); print(T.round(2).to_string(index=False))
pr = []
for fam in ("RF", "XGB"):
    W = D[(D.fam == fam) & (D.cohort == "tuep") & (D.gamma == 1.0)].pivot_table(index="rep", columns="pipe", values="fm")[["rank-pat", "rank-none"]].dropna(); d = W["rank-pat"] - W["rank-none"]
    t, p = stats.ttest_1samp(d, 0); ci = stats.t.interval(0.95, len(d) - 1, d.mean(), d.std(ddof=1) / np.sqrt(len(d))); pr.append(dict(fam=fam, n=len(d), diff=d.mean(), lo=ci[0], hi=ci[1], t=t, p=p))
P = pd.DataFrame(pr); o = np.argsort(P.p.values); adj = np.empty(2); run = 0
for r, i in enumerate(o): run = max(run, min(1, (2 - r) * P.p.values[i])); adj[i] = run
P["p_holm"] = adj; P.to_csv(os.path.join(O, "uji_primer.csv"), index=False); print("\nPRIMER (TUEP gamma 1, rank+pat - rank+none, pp):"); print(P.round(4).to_string(index=False))
