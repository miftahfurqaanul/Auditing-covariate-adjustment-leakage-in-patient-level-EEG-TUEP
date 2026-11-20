#!/usr/bin/env python3
"""Putaran 58: ringkasan sesuai PRAREGISTRASI_putaran57_60.md (di-hash sebelum hasil dilihat). Tidak melatih model."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
from scipy import stats
from round49_kalibrasi import decide, THR
FC = [f"fold_auc_{i}" for i in range(5)]; O = os.path.join(HERE, "hasil58"); LAB = {0: "no evidence", 1: "uncertain", 2: "leakage"}
dec = lambda d: LAB[int(decide(np.array([d.mean()]), np.array([d.std(ddof=1) / np.sqrt(len(d))]), THR["R2"]["asli"])[0])]
D = pd.read_csv(os.path.join(O, "replikasi.csv")); D["fm"] = 100 * D[FC].mean(1); D = D.drop_duplicates(["cohort", "rep", "gamma", "adjust", "patients", "weight", "aggregate"])
W = D.pivot_table(index=["cohort", "rep", "gamma", "patients", "weight", "aggregate"], columns="adjust", values="fm").dropna(); W["leak"] = W["pat"] - W["none"]; L = W.reset_index()
rows = []
for (c, g, P, Wt, A), G in L.groupby(["cohort", "gamma", "patients", "weight", "aggregate"]):
    d = G.leak.values; ci = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    rows.append(dict(cohort=c, gamma=g, patients=P, weight=Wt, aggregate=A, n=len(d), auc_none=G["none"].mean(), auc_pat=G["pat"].mean(), leak=d.mean(), lo=d.mean() - ci, hi=d.mean() + ci, R2=dec(d / 100)))
T = pd.DataFrame(rows); T.to_csv(os.path.join(O, "varian.csv"), index=False); pd.set_option("display.width", 250); print(T.round(2).to_string(index=False))
# efek utama faktorial (gamma 1): rata-rata atas faktor lain
for c in ("tuep", "emotiv"):
    G = L[(L.cohort == c) & (L.gamma == 1.0)]
    for f, (a, b) in (("patients", ("p80", "p100")), ("weight", ("win", "pat")), ("aggregate", ("prob", "logit"))):
        x = G.groupby(["rep", f]).leak.mean().unstack(); d = x[a] - x[b]; print(f"{c} gamma 1 efek utama {f} ({a} - {b}): {d.mean():+.2f} pp (SE {d.std(ddof=1) / np.sqrt(len(d)):.2f})")
# kontras primer
G = L[(L.cohort == "tuep") & (L.gamma == 1.0)].set_index(["rep", "patients", "weight", "aggregate"]).leak
base = G.xs(("p80", "win", "prob"), level=[1, 2, 3]); match = G.xs(("p100", "pat", "logit"), level=[1, 2, 3]); d = (base - match).dropna()
t, p2 = stats.ttest_1samp(d, 0); p1 = p2 / 2 if t > 0 else 1 - p2 / 2; ci = stats.t.interval(0.95, len(d) - 1, d.mean(), d.std(ddof=1) / np.sqrt(len(d)))
print(f"\nPRIMER (TUEP gamma 1): kebocoran resep pohon - kebocoran varian disamakan = {d.mean():+.2f} pp (95% CI {ci[0]:+.2f} to {ci[1]:+.2f}); t = {t:.2f}, p satu sisi = {p1:.3g}; n = {len(d)}")
pd.DataFrame([dict(diff=d.mean(), lo=ci[0], hi=ci[1], t=t, p_one_sided=p1, n=len(d))]).to_csv(os.path.join(O, "uji_primer.csv"), index=False)
