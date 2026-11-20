#!/usr/bin/env python3
"""Putaran 55: ringkasan sesuai PRAREGISTRASI_putaran55.md (di-hash sebelum hasil dilihat). Membaca hasil55/replikasi.csv; menulis
hasil55/ringkas_sel.csv, hasil55/uji_primer.csv, hasil55/ringkas.log (lewat stdout). Tidak melatih model.
Pemakaian: python round55_ringkas.py"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
from scipy import stats
from round49_kalibrasi import decide, THR
OUT = os.path.join(HERE, "hasil55"); D = pd.read_csv(os.path.join(OUT, "replikasi.csv"))
D = D.drop_duplicates(["nref", "design", "s", "rep"]); LAB = {0: "no evidence", 1: "uncertain", 2: "leakage"}
assert len(D) == 3000 and D.groupby(["nref", "design", "s"]).size().eq(100).all(), "unit tidak lengkap"
for v in ("C1", "CR", "AP"):
    assert D[f"{v}_pooled"].between(0, 1).all()
D["leak_CR"] = 100 * (D.CR_pooled - D.C1_pooled); D["leak_AP"] = 100 * (D.AP_pooled - D.C1_pooled)
D["leak_CR_fold"] = 100 * (D.CR_fold - D.C1_fold); D["leak_AP_fold"] = 100 * (D.AP_fold - D.C1_fold)


def dec(x, rule):
    x = np.asarray(x, float); return LAB[int(decide(np.array([x.mean()]), np.array([x.std(ddof=1) / np.sqrt(len(x))]), THR[rule]["asli"])[0])]


rows = []
for (n, d, s), g in D.groupby(["nref", "design", "s"]):
    ci = lambda x: stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    rows.append(dict(nref=n, design=d, s=s, n_rep=len(g), min_ctrl_era_cell_median=g.min_ctrl_era_cell.median(),
                     C1_auc=100 * g.C1_pooled.mean(), CR_auc=100 * g.CR_pooled.mean(), AP_auc=100 * g.AP_pooled.mean(),
                     leak_CR=g.leak_CR.mean(), leak_CR_ci=ci(g.leak_CR), leak_CR_sd=g.leak_CR.std(ddof=1), leak_AP=g.leak_AP.mean(), leak_AP_ci=ci(g.leak_AP),
                     leak_CR_fold=g.leak_CR_fold.mean(), leak_AP_fold=g.leak_AP_fold.mean(),
                     R1_CR=dec(g.CR_pooled, "R1"), R2_CR=dec(g.leak_CR / 100, "R2"), R2_AP=dec(g.leak_AP / 100, "R2")))
S = pd.DataFrame(rows).sort_values(["design", "s", "nref"]); S.to_csv(os.path.join(OUT, "ringkas_sel.csv"), index=False)
pd.set_option("display.width", 250); print(S.round(2).to_string(index=False))
# keluarga primer (Holm, 3 uji Welch satu sisi)
cell = lambda n, d, s: D[(D.nref == n) & (D.design == d) & (D.s == s)].leak_CR.values
T = [("P1 nref 80 > 1000 (jarang, s=1)", cell(80, "jarang", 1.0), cell(1000, "jarang", 1.0)),
     ("P2 jarang > cermin (nref 80, s=1)", cell(80, "jarang", 1.0), cell(80, "cermin", 1.0)),
     ("P3 s=1 > s=0 (nref 80, jarang)", cell(80, "jarang", 1.0), cell(80, "jarang", 0.0))]
U = []
for name, a, b in T:
    t, p = stats.ttest_ind(a, b, equal_var=False, alternative="greater")
    df = (a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)) ** 2 / ((a.var(ddof=1) / len(a)) ** 2 / (len(a) - 1) + (b.var(ddof=1) / len(b)) ** 2 / (len(b) - 1))
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)); dlt = a.mean() - b.mean(); tc = stats.t.ppf(0.975, df)
    U.append(dict(test=name, diff=dlt, lo=dlt - tc * se, hi=dlt + tc * se, t=t, p=p))
U = pd.DataFrame(U); p = U.p.values; o = np.argsort(p); adj = np.empty(3); run = 0
for r, i in enumerate(o):
    run = max(run, min(1, (3 - r) * p[i])); adj[i] = run
U["p_holm"] = adj; U.to_csv(os.path.join(OUT, "uji_primer.csv"), index=False); print(); print(U.round(4).to_string(index=False))
# sekunder: kemiringan kebocoran CR terhadap log10 n_ref (per desain, s=1)
for d in ("jarang", "cermin"):
    g = D[(D.design == d) & (D.s == 1.0)]; sl = stats.linregress(np.log10(g.nref), g.leak_CR)
    print(f"kemiringan {d} s=1: {sl.slope:.2f} pp per dekade (SE {sl.stderr:.2f})")
# pemeriksaan validitas
z = S[S.s == 0.0]; print("\ns=0: keputusan R2 CR:", z.R2_CR.value_counts().to_dict(), "| rata-rata |AP-C1| semua sel:", round(S.leak_AP.abs().mean(), 2), "| R2 AP:", S.R2_AP.value_counts().to_dict())
