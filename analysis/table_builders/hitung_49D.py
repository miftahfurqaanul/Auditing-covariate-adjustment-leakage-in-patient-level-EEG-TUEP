#!/usr/bin/env python3
"""Recomputes 49D from hasil49/D_replikasi.csv (no training). Output: angka_49D.csv, uji_primer_49D.csv, prediksi_49D.csv (in this folder)."""
import os, numpy as np, pandas as pd
from scipy import stats
HERE = os.path.dirname(os.path.abspath(__file__)); H = os.path.join(os.path.dirname(os.path.dirname(HERE)), "analisis_tambahan", "hasil49")
D = pd.read_csv(os.path.join(H, "D_replikasi.csv"))
assert len(D) == 720 and not D.duplicated(["icc", "m", "dist", "rep"]).any() and D.groupby(["dist", "m", "icc"]).size().eq(20).all()
assert np.abs(D.RFwin_A - D.RFwin_B).max() == 0 and np.abs(D.LRpat_win - D.LRpat_pat).max() == 0     # A = B and LR per window = LR per patient (uniform m)
for k, (a, b) in dict(A=("RFwin_A", "RFwin_none"), C=("RFpat_C", "RFpat_none"), LR=("LRpat_win", "LRpat_none")).items(): D[k] = 100 * (D[a] - D[b])
D["AmC"] = D.A - D.C
G = D.groupby(["dist", "m", "icc"]).agg(**{f"{k}_mean": (k, "mean") for k in ("A", "C", "LR", "AmC")}, **{f"{k}_se": (k, "sem") for k in ("A", "C", "LR", "AmC")}).reset_index(); G.to_csv(os.path.join(HERE, "angka_49D.csv"), index=False)
L = D[(D.dist == "lognormal") & (D.m == 50)]; a1 = L[L.icc == 0.95].sort_values("rep").A.values; a0 = L[L.icc == 0.0].sort_values("rep").A.values
t1 = stats.ttest_ind(a1, a0, equal_var=False); v1, v0 = a1.var(ddof=1) / 20, a0.var(ddof=1) / 20; se1 = np.sqrt(v1 + v0); df1 = (v1 + v0) ** 2 / (v1 ** 2 / 19 + v0 ** 2 / 19); d1 = a1.mean() - a0.mean(); tc = stats.t.ppf(.975, df1)
p1 = 1 - stats.t.cdf(d1 / se1, df1)
q = L[L.icc == 0.0].sort_values("rep"); dd = q.A.values - q.C.values; se2 = dd.std(ddof=1) / np.sqrt(20); tc2 = stats.t.ppf(.975, 19); p2 = 1 - stats.t.cdf(dd.mean() / se2, 19)
holm = [min(1, max(2 * min(p1, p2), p1)), min(1, 2 * min(p1, p2))] if p1 > p2 else [min(1, 2 * p1), min(1, max(2 * p1, p2))]
P = pd.DataFrame([dict(uji="D1", selisih=d1, ci_lo=d1 - tc * se1, ci_hi=d1 + tc * se1, p_satu_sisi=p1, df=df1, holm=holm[0]), dict(uji="D2", selisih=dd.mean(), ci_lo=dd.mean() - tc2 * se2, ci_hi=dd.mean() + tc2 * se2, p_satu_sisi=p2, df=19, holm=holm[1])]); P.to_csv(os.path.join(HERE, "uji_primer_49D.csv"), index=False)
ref = pd.read_csv(os.path.join(H, "ringkas_D_primer.csv")); assert abs(ref.p.iloc[0] - p1) < 1e-9 and abs(ref.p.iloc[1] - p2) < 1e-9
# plan predictions
nn = G[G.dist != "normal"]; pr = []
pr.append(("D1 positive and passes Holm (55%)", "meleset", f"difference {d1:+.2f} poin; p = {p1:.2f}; Holm {holm[0]:.2f}"))
pr.append(("D2 positive and passes Holm (65%)", "terpenuhi" if (dd.mean() > 0 and holm[1] < .05) else "meleset", f"difference {dd.mean():+.2f} poin; p = {p2:.1e}; Holm {holm[1]:.1e}"))
g0 = G[(G.dist == "lognormal") & (G.m == 50)].set_index("icc").AmC_mean; pr.append(("A - C narrows at ICC 0.95 compared with ICC 0 (55%)", "terpenuhi" if g0[0.95] < g0[0.0] else "meleset", f"lognormal, m = 50: {g0[0.0]:.1f} -> {g0[0.95]:.1f} poin"))
nlr = int((G.LR_mean.abs() > 1.0).sum()); pr.append(("LR |delta| <= 1 poin di semua sel (80%)", "terpenuhi" if nlr == 0 else "meleset", f"{nlr} of 36 cells exceed; max absolute {G.LR_mean.abs().max():.1f} poin"))
N_ = G[G.dist == "normal"]; nn_ = {k: int((N_[f"{k}_mean"].abs() > 1.0).sum()) for k in ("A", "C", "LR")}; pr.append(("Normal distribution |delta| <= 1 point for all arms (75%)", "terpenuhi" if sum(nn_.values()) == 0 else "meleset", f"normal cells that exceed: A {nn_['A']}, C {nn_['C']}, LR {nn_['LR']} of 12; max absolute A {N_.A_mean.abs().max():.1f}, C {N_.C_mean.abs().max():.1f}, LR {N_.LR_mean.abs().max():.1f}"))
inc = {(d, ic): bool(G[(G.dist == d) & (G.icc == ic)].sort_values("m").A_mean.is_monotonic_increasing) for d in ("normal", "lognormal", "contam") for ic in (0.0,)}; pr.append(("Delta A increases with m at ICC 0 (40%)", "meleset" if not inc[("lognormal", 0.0)] and not inc[("contam", 0.0)] else "terpenuhi", "; ".join(f"{d}: " + ", ".join(f"{v:.1f}" for v in G[(G.dist == d) & (G.icc == 0.0)].sort_values('m').A_mean) for d in ("normal", "lognormal", "contam"))))
dec = sum(int(G[(G.dist == d) & (G.m == m) & (G.icc == 0.95)].A_mean.iloc[0] < G[(G.dist == d) & (G.m == m) & (G.icc == 0.0)].A_mean.iloc[0]) for d in ("lognormal", "contam") for m in (10, 50, 200)); pr.append(("(descriptive) delta A at ICC 0.95 lower than at ICC 0", "-", f"{dec} of 6 combinations of heavy-tailed distribution x m"))
pd.DataFrame(pr, columns=["prediction", "status", "number"]).to_csv(os.path.join(HERE, "prediksi_49D.csv"), index=False)
print(P.round(5).to_string(index=False)); print(pd.DataFrame(pr, columns=["prediction", "status", "number"]).to_string(index=False))
nl = G[G.dist != "normal"]; print("A (heavy-tailed): min", round(nl.A_mean.min(), 1), "max", round(nl.A_mean.max(), 1), "cells with 95% CI > 0:", int(((nl.A_mean - 1.96 * nl.A_se) > 0).sum()), "of", len(nl))
print("A normal: max absolute", round(N_.A_mean.abs().max(), 1), "cells with CI not containing 0:", int(((N_.A_mean.abs() - 1.96 * N_.A_se) > 0).sum()))
print("C berekor berat: min", round(nl.C_mean.min(), 1), "maks", round(nl.C_mean.max(), 1)); print("LR maks", round(G.LR_mean.max(), 1), round(G.LR_mean.min(), 1))
