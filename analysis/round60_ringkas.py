#!/usr/bin/env python3
"""Putaran 60: ringkasan sesuai PRAREGISTRASI_putaran57_60.md (di-hash sebelum hasil dilihat). Audit independen tidak tumpang tindih (urut rep) dan prediksi dari pool tersimpan."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
from round49_kalibrasi import decide, THR
O = os.path.join(HERE, "hasil60"); RNG = np.random.default_rng(60)
D = pd.read_csv(os.path.join(O, "replikasi.csv")).drop_duplicates(["code", "rep"]).sort_values(["code", "rep"]); assert D.groupby("code").size().eq(200).all()
def decs(x, d):
    r1 = decide(np.array([x.mean()]), np.array([x.std(ddof=1) / np.sqrt(len(x))]), THR["R1"]["asli"])[0]; r2 = decide(np.array([d.mean()]), np.array([d.std(ddof=1) / np.sqrt(len(d))]), THR["R2"]["asli"])[0]
    return r1, r2, (2 if 2 in (r1, r2) else (0 if r1 == r2 == 0 else 1))
def stored_pool(code):
    import round60_validasi_aturan as V
    if code in V.SIM:
        cell, t, _ = V.SIM[code]; B = pd.read_csv(os.path.join(HERE, "hasil39", "B_cpu.csv")); m = (B.clf == "RF") & (B.scen == "H1")
        for k in ("dist", "n", "m", "p", "s", "sig", "cfrac", "cscale"): m &= np.isclose(B[k], cell[k]) if k != "dist" else (B[k] == cell[k])
        m &= ~B.imb.astype(str).str.lower().eq("true"); G = B[m].pivot_table(index="rep", columns=["transform", "adjust"], values="auc")
        return G[(t, "pat")].values, (G[(t, "pat")] - G[("raw", "none")]).values
    s = V.REF[code][0]; R = pd.read_csv(os.path.join(HERE, "hasil55", "replikasi.csv")); R = R[(R.nref == 80) & (R.design == "jarang") & (R.s == s)]
    return R.CR_pooled.values, (R.CR_pooled - R.C1_pooled).values
rows = []
for code, G in D.groupby("code"):
    x, d = G.auc_adj.values, G.delta.values; truth = G.truth.iloc[0]
    for n in (20, 50, 100):
        k = len(x) // n; cnt = np.zeros((3, 3), int)
        for i in range(k):
            r = decs(x[i * n:(i + 1) * n], d[i * n:(i + 1) * n])
            for j in range(3): cnt[j, r[j]] += 1
        rows.append(dict(code=code, truth=truth, n=n, audits=k, mean_d_fresh=100 * d.mean(), sd_d_fresh=100 * d.std(ddof=1),
                         **{f"{rule}_{lab}": cnt[j, v] for j, rule in enumerate(("R1", "R2", "gab")) for v, lab in enumerate(("tb", "tp", "bocor"))}))
    xs, ds = stored_pool(code); idx = RNG.integers(0, len(xs), (4000, 20)); X, Dd = xs[idx], ds[idx]
    r2 = decide(Dd.mean(1), Dd.std(1, ddof=1) / np.sqrt(20), THR["R2"]["asli"]); r1 = decide(X.mean(1), X.std(1, ddof=1) / np.sqrt(20), THR["R1"]["asli"])
    rows[-3].update(pred_pool_n=len(xs), pred_mean_d=100 * ds.mean(), pred_R2_bocor=float((r2 == 2).mean()), pred_R2_tb=float((r2 == 0).mean()), pred_R1_bocor=float((r1 == 2).mean()))
T = pd.DataFrame(rows); T.to_csv(os.path.join(O, "audit_independen.csv"), index=False); pd.set_option("display.width", 260); print(T.round(3).to_string(index=False))
for truth, G in T.groupby("truth"):
    print(f"\n== {truth}"); print(G.groupby("n")[["audits", "R1_bocor", "R2_bocor", "gab_bocor", "R2_tb", "gab_tb"]].sum().to_string())
