#!/usr/bin/env python3
"""Round 44: summary (CPU) according to PRAREGISTRASI_putaran44.md. Reads hasil44/{A,B,C}_prediksi.npz, E_nm_prediksi.npz, D_ipw_rf_plasmode.csv. Usage: python round44_ringkas.py"""
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "hasil44"); sys.path.insert(0, HERE); sys.path.insert(0, "/home/miftah_f/RTX-5090/tuep_fresh/code")
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
import round28_analisis as A
import round41_strata_kaliper as S41
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300)
Y, J, c17, AGE = A.Y, A.J, A.c17, A.AGE.astype(float); N = len(Y)


def nb(a, b):
    d = np.asarray(a) - np.asarray(b); se = np.sqrt(d.var(ddof=1) * (1 / 50 + 40 / 160)); t = d.mean() / se; h = stats.t.ppf(0.975, 49) * se; return d.mean(), d.mean() - h, d.mean() + h, 2 * stats.t.sf(abs(t), 49)


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); h = np.empty(len(p)); run = 0
    for r, i in enumerate(o):
        run = max(run, min(1, p[i] * (len(p) - r))); h[i] = run
    return h


def dec_R1(m, se):
    return "tidak ada bukti" if (m <= 0.52 and m + 1.645 * se <= 0.54) else ("kebocoran" if m > 0.55 else "tidak pasti")


L = lambda f: np.load(os.path.join(OUT, f)); M = {}
for f in ("A_prediksi.npz", "B_prediksi.npz", "C_prediksi.npz", "E_nm_prediksi.npz"):
    if os.path.exists(os.path.join(OUT, f)):
        z = L(f); M.update({k: np.asarray(z[k], float) for k in z.files})
h30 = np.load(os.path.join(HERE, "hasil30/eval_prediksi_v4.npz")); h30p3 = np.load(os.path.join(HERE, "hasil30/p3_v5_prediksi_data_asli.npz")); I2ADJ = np.load(os.path.join(HERE, "hasil29/B_prediksi_data_asli.npz"))["V4_AS"]
REF = {"F2-adj": h30["F2_adj"], "I2-adj": I2ADJ, "P3-adj": h30["P3_adj"], "NM-adj": h30["NM_adj"], "C1": h30["C1"], "Floor": h30["Floor"], "Demo": h30["Demo"]}
# H44-0
d1 = np.abs(c17.fold_aucs(J, M["I2__asis"]) - c17.fold_aucs(J, I2ADJ)).max(); d2 = np.abs(c17.fold_aucs(J, M["P3__asis"]) - c17.fold_aucs(J, np.asarray(h30p3["P3_V5_AS"], float))).max()
print(f"[H44-0] reproduction: I2 as-is vs hasil29 V4_AS: max per-fold AUC difference {d1:.2e}; P3 asis vs hasil30 P3_V5_AS: {d2:.2e} (limit 1e-9): {'PASSED' if max(d1, d2) <= 1e-9 else 'FAILED'}")
# F2-Clean
def avg(*ms):
    return np.mean(np.stack(ms), axis=0)


M["F2-Clean-noNM"] = avg(M["I2__rank-noAS-D2"], M["P3__rank-noAS-D2"]); M["F2-Clean-noNM-D1"] = avg(M["I2__rank-noAS-D1"], M["P3__rank-noAS-D1"]); M["F2-Clean-noNM-noRank"] = avg(M["I2__noAS"], M["P3__noAS"]); M["F2-Clean-noNM-30"] = avg(M["I2-30"], M["P3-30"])
if "NM_clean" in M:
    M["F2-Clean"] = avg(M["I2__rank-noAS-D2"], M["P3__rank-noAS-D2"], M["NM_clean"])
else:
    print("[warning] NM_clean does not exist yet: F2-Clean (with NM) cannot be assembled yet")
FA = {k: c17.fold_aucs(J, v) for k, v in {**REF, **M}.items() if v.shape == (10, N)}
# AUC table + Spisak test
rows = []
for k in [k for k in M if M[k].shape == (10, N)] + ["F2-adj", "I2-adj", "P3-adj", "NM-adj", "C1"]:
    P = M[k] if k in M else REF[k]; fa = c17.fold_aucs(J, P); se = np.sqrt(fa.var(ddof=1) * (1 / 50 + 40 / 160)); h = stats.t.ppf(0.975, 49) * se; s = P.mean(0); r = dict(model=k, auc_fold=100 * fa.mean(), lo=100 * (fa.mean() - h), hi=100 * (fa.mean() + h), auc_gab=100 * np.mean([roc_auc_score(Y, p) for p in P]))
    for cn in ("age", "era", "device", "montage"):
        c, kind = A.CONF_PARTIAL[cn]; r2, p = A.partial_test(s, c, kind, 44100, B=2000); r[f"r2_{cn}"], r[f"p_{cn}"] = r2, p
    for ref in ("F2-adj", "I2-adj", "C1", "Floor"):
        d, lo, hi, p = nb(fa, c17.fold_aucs(J, REF[ref])); r[f"d_{ref}"], r[f"p_{ref}"] = d, p
    rows.append(r)
T = pd.DataFrame(rows); T.to_csv(f"{OUT}/ringkas_modelos.csv", index=False)
print("\n== Per-fold AUC, differences (vs F2-adj / I2-adj / C1 / metadata) and partial Spisak test (r^2, p) ==\n" + T[["model", "auc_fold", "lo", "hi", "auc_gab", "r2_age", "p_age", "r2_era", "p_era", "r2_device", "p_device", "r2_montage", "p_montage"]].round(4).to_string(index=False))
print("\n" + T[["model", "d_F2-adj", "p_F2-adj", "d_I2-adj", "p_I2-adj", "d_C1", "p_C1", "d_Floor", "p_Floor"]].round(4).to_string(index=False))
age_ok = {r.model: bool(r.r2_age < 0.02 and r.p_age > 0.10) for r in T.itertuples()}; print("\n[H44-1] passed (age r^2 < 0.02 and p > 0.10): " + ", ".join(f"{k}={v}" for k, v in age_ok.items() if k.startswith(("F2", "I2__rank", "P3__rank", "NM", "IPW"))))
# analysis within strata (year x device x age caliper)
g = S41.STR["year x device"]; ks = S41.KS; names = [n for n in ("F2-Clean", "F2-Clean-noNM", "F2-Clean-noNM-30", "F2-adj", "I2__rank-noAS-D2", "P3__rank-noAS-D2", "NM_clean", "IPW-I2", "IPW-RF", "C1", "Floor", "Demo") if (n in M and M[n].shape == (10, N)) or n in REF]
Sm = np.column_stack([(M[n] if n in M else REF[n]).mean(0) for n in names]); ei, ni = S41.make_pairs(Y, g, np.arange(N)); A0, np0 = S41.auc_by_k(Sm, ei, ni)
rng = np.random.default_rng(44200); groups = [np.where(g == s)[0] for s in np.unique(g)]; B = np.full((2000, len(ks), len(names)), np.nan)
for b in range(2000):
    idx = np.concatenate([rng.choice(v, len(v)) for v in groups]); e, n_ = S41.make_pairs(Y, g, idx); B[b] = S41.auc_by_k(Sm, e, n_)[0]
np.savez(f"{OUT}/bruto_strata.npz", point=A0, npairs=np0, boot=B, names=np.array(names), ks=np.array(ks))
def pv(d, null):
    d = d[~np.isnan(d)]; return max(min(1.0, 2 * min((d <= null).mean(), (d >= null).mean())), 1 / len(d))
KN = ["5", "10", "15", "tak hingga"]; rows = []
for i, kn in enumerate(KN):
    for j, n in enumerate(names):
        rows.append(dict(k=kn, model=n, npairs=int(np0[i]), auc=A0[i, j], lo=np.nanpercentile(B[:, i, j], 2.5), hi=np.nanpercentile(B[:, i, j], 97.5), p_vs_0p5=pv(B[:, i, j], 0.5)))
St = pd.DataFrame(rows); St.to_csv(f"{OUT}/ringkas_strata.csv", index=False); print("\n== AUC within year x device strata (age caliper k) ==\n" + St.pivot_table(index="model", columns="k", values="auc").loc[:, KN].round(4).to_string())
# keluarga primer
main = "F2-Clean" if "F2-Clean" in M else None; pr = []
if main:
    fa = FA[main]
    for ref in ("F2-adj", "C1", "Floor"):
        d, lo, hi, p = nb(fa, FA[ref]); pr.append(dict(uji=f"T{len(pr) + 1}: {main} - {ref} (per-fold AUC)", nilai=d, lo=lo, hi=hi, p=p))
    j = names.index(main); i = KN.index("10"); pr.append(dict(uji="T4: AUC_10 of F2-Clean > 0.5 (year x device, age caliper 10)", nilai=A0[i, j], lo=np.nanpercentile(B[:, i, j], 2.5), hi=np.nanpercentile(B[:, i, j], 97.5), p=pv(B[:, i, j], 0.5)))
    P = pd.DataFrame(pr); P["p_holm"] = holm(P.p.values); P.to_csv(f"{OUT}/ringkas_primer.csv", index=False); print("\n== Primary family (Holm over 4 tests) ==\n" + P.round(4).to_string(index=False))
    print(f"\n[Klaim] T4 lolos Holm < 0.05: {bool(P.p_holm.iloc[3] < 0.05)}; H44-1 F2-Clean lolos: {age_ok['F2-Clean']} -> claim 'retains signal without AS': {bool(P.p_holm.iloc[3] < 0.05 and age_ok['F2-Clean'])}")
# K2 permutation control
rng = np.random.default_rng(44201); cells = [np.where(g == s)[0] for s in np.unique(g)]; ref_name = main or "F2-Clean-noNM"; sc = M[ref_name].mean(0); perm = []
for _ in range(200):
    yp = Y.copy()
    for v in cells:
        yp[v] = rng.permutation(Y[v])
    e, n_ = S41.make_pairs(yp, g, np.arange(N)); perm.append(S41.auc_by_k(sc[:, None], e, n_)[0][:, 0])
perm = np.array(perm); print(f"\n[K2] label permutation within cells (200x), {ref_name}: mean AUC_k per k {np.round(np.nanmean(perm, 0), 4).tolist()}; within 0.5 +- 0.02: {bool(np.all(np.abs(np.nanmean(perm, 0) - 0.5) < 0.02))}")
# plasmode IPW-RF
f = f"{OUT}/D_ipw_rf_plasmode.csv"
if os.path.exists(f):
    D = pd.read_csv(f); FC = [f"fold_auc_{i}" for i in range(5)]; D["fold_mean"] = D[FC].mean(axis=1); base = pd.read_csv(os.path.join(HERE, "hasil42/RF.csv")); base["fold_mean"] = base[FC].mean(axis=1); base["pipe"] = base["transform"] + "-" + base["adjust"]; rows = []
    for (cond, g_), G in D.groupby(["cond", "gamma"]):
        x = G["fold_mean"].values; m, se = x.mean(), x.std(ddof=1) / np.sqrt(len(x)); r = dict(cond=cond, gamma=g_, nrep=len(x), ipw_rf=m, se=se, R1=dec_R1(m, se))
        for pipe in ("raw-none", "raw-pat", "rank-pat"):
            b = base[(base["cond"] == cond) & (base["gamma"] == g_) & (base["pipe"] == pipe)].set_index("rep")["fold_mean"]; r[pipe] = b.loc[G["rep"].values].mean()
        rows.append(r)
    Pd = pd.DataFrame(rows); Pd.to_csv(f"{OUT}/ringkas_ipw_rf_plasmode.csv", index=False); print("\n== IPW-RF raw (plasmode; features permuted) vs the round 42 pipeline on the same replicates ==\n" + Pd.round(4).to_string(index=False))
