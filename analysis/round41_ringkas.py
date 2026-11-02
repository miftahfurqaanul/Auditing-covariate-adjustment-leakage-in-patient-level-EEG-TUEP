#!/usr/bin/env python3
"""Round 41: summary of hasil41/bruto_*.npz according to PRAREGISTRASI_putaran41.md. CPU only. Usage: python round41_ringkas.py"""
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "hasil41")
import numpy as np, pandas as pd

KS = [5.0, 10.0, 15.0, np.inf]; KN = ["5", "10", "15", "tak hingga"]
DIFFS = [("F2_adj", "Floor"), ("F2_adj", "C1"), ("F2_adj", "I2_adj"), ("F2_adj", "Demo"), ("F2_adj", "LaBraM_FT"), ("F2_adj", "CBraMod_FT")]


def pval(d, null):
    return max(min(1.0, 2 * min((d <= null).mean(), (d >= null).mean())), 1 / len(d))


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); h = np.empty(len(p)); run = 0
    for r, i in enumerate(o):
        run = max(run, min(1, p[i] * (len(p) - r))); h[i] = run
    return h


rows, drows = [], []
for sname in ("year x device", "year"):
    z = np.load(f"{OUT}/bruto_{sname.replace(' ', '_')}.npz"); pt, npairs, rep, B, NP = z["point"], z["npairs"], z["rep_mean"], z["boot"], z["boot_npairs"]; M = list(z["models"])
    for i, kn in enumerate(KN):
        for j, m in enumerate(M):
            b = B[:, i, j]; rows.append(dict(strata=sname, k=kn, model=m, npairs=int(npairs[i]), auc=pt[i, j], lo=np.nanpercentile(b, 2.5), hi=np.nanpercentile(b, 97.5), p_vs_0p5=pval(b[~np.isnan(b)], 0.5), auc_rep_mean=rep[i, j], boot_npairs_med=float(np.median(NP[:, i]))))
        for a, c in DIFFS:
            d = B[:, i, M.index(a)] - B[:, i, M.index(c)]; ok = ~np.isnan(d); dd = pt[i, M.index(a)] - pt[i, M.index(c)]
            drows.append(dict(strata=sname, k=kn, test=f"{a} - {c}", npairs=int(npairs[i]), diff=dd, lo=np.percentile(d[ok], 2.5), hi=np.percentile(d[ok], 97.5), p=pval(d[ok], 0.0)))
T = pd.DataFrame(rows); D = pd.DataFrame(drows)
# primary family (pre-registration section 2): A_k = F2_adj AUC_k > 0.5; B_k = F2_adj - Floor > 0; k = 5, 10, 15; year x device strata; Holm over 6 tests
fa = T[(T.strata == "year x device") & (T.model == "F2_adj") & T.k.isin(["5", "10", "15"])].copy(); fa["uji"] = "A_" + fa.k; fa = fa.rename(columns={"p_vs_0p5": "p"})[["uji", "npairs", "auc", "lo", "hi", "p"]]
fb = D[(D.strata == "year x device") & (D.test == "F2_adj - Floor") & D.k.isin(["5", "10", "15"])].copy(); fb["uji"] = "B_" + fb.k; fb = fb.rename(columns={"diff": "auc"})[["uji", "npairs", "auc", "lo", "hi", "p"]]
P = pd.concat([fa, fb], ignore_index=True); P["p_holm"] = holm(P.p.values); P["lolos_holm"] = P.p_holm < 0.05
T.to_csv(f"{OUT}/ringkas_auc.csv", index=False); D.to_csv(f"{OUT}/ringkas_selisih.csv", index=False); P.to_csv(f"{OUT}/ringkas_primer.csv", index=False)
pd.set_option("display.width", 250)
print("== Pasangan efektif ==\n" + T[T.model == "F2_adj"][["strata", "k", "npairs"]].to_string(index=False))
print("\n== Primary family (Holm over 6 tests) ==\n" + P.round(4).to_string(index=False))
print("\n== AUC within year x device strata per model and k ==\n" + T[T.strata == "year x device"].pivot_table(index="model", columns="k", values="auc").loc[:, KN].round(4).to_string())
print("\n== 95% CI for F2_adj, Floor, Demo, C1 (year x device) ==\n" + T[(T.strata == "year x device") & T.model.isin(["F2_adj", "Floor", "Demo", "C1", "I2_adj"])][["model", "k", "auc", "lo", "hi", "p_vs_0p5"]].round(4).to_string(index=False))
print("\n== Difference F2_adj vs comparators (descriptive) ==\n" + D[D.strata == "year x device"][["k", "test", "npairs", "diff", "lo", "hi", "p"]].round(4).to_string(index=False))
print("\n== Strata 'tahun' (sekunder; deskriptif) ==\n" + T[(T.strata == "year") & T.model.isin(["F2_adj", "Floor", "Demo", "C1"])].pivot_table(index="model", columns="k", values="auc").loc[:, KN].round(4).to_string())
# sanity check
z0 = np.load(f"{OUT}/bruto_year_x_device.npz"); M = list(z0["models"]); pt = z0["point"]
c1 = abs(pt[3, M.index("F2_adj")] - 0.769534) < 1e-6 and abs(pt[3, M.index("Floor")] - 0.431728) < 1e-6; dm = pt[0, M.index("Demo")]; pm = np.load(f"{OUT}/bruto_permutasi.npz")["perm"]
c3 = bool(np.all(np.abs(np.nanmean(pm[:, :, M.index("F2_adj")], 0) - 0.5) < 0.02))
print(f"\n== Sanity checks ==\n(1) k = infinity equals round 30 (F2 0.769534, metadata 0.431728): {c1}\n(2) Demo at k=5 = {dm:.4f}; within [0.40, 0.60]: {0.40 <= dm <= 0.60}\n(3) Label permutation within cells (200x): mean F2 AUC_k per k = {np.round(np.nanmean(pm[:, :, M.index('F2_adj')], 0), 4).tolist()}; within 0.5 +- 0.02: {c3}")
