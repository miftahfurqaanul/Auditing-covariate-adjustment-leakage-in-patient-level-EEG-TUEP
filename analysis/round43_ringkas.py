#!/usr/bin/env python3
"""Round 43: summary (CPU) according to PRAREGISTRASI_putaran43.md. Reads hasil43/{A_prediksi.npz, C_null_replikasi.csv, D_bruto_*.npz, D_perm.npy}. Usage: python round43_ringkas.py"""
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "hasil43"); sys.path.insert(0, HERE); sys.path.insert(0, "/home/miftah_f/RTX-5090/tuep_fresh/code")
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
import round28_analisis as A
import round43_optimisme as O
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
Y, J, c17 = A.Y, A.J, A.c17


def nb(a, b):
    d = np.asarray(a) - np.asarray(b); se = np.sqrt(d.var(ddof=1) * (1 / 50 + 40 / 160)); t = d.mean() / se; h = stats.t.ppf(0.975, 49) * se; return d.mean(), d.mean() - h, d.mean() + h, 2 * stats.t.sf(abs(t), 49)


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); h = np.empty(len(p)); run = 0
    for r, i in enumerate(o):
        run = max(run, min(1, p[i] * (len(p) - r))); h[i] = run
    return h


def dec_R1(m, se):
    return "tidak ada bukti" if (m <= 0.52 and m + 1.645 * se <= 0.54) else ("kebocoran" if m > 0.55 else "tidak pasti")


Z = np.load(f"{OUT}/A_prediksi.npz"); h30 = np.load(os.path.join(HERE, "hasil30/eval_prediksi_v4.npz")); I2 = np.load(os.path.join(HERE, "hasil29/B_prediksi_data_asli.npz"))["V4_AS"]
REF = {"I2-adj": I2, "C1": h30["C1"], "Floor": h30["Floor"], "F2-adj": h30["F2_adj"]}; FA = {k: c17.fold_aucs(J, v) for k, v in REF.items()}
tags = [k for k in Z.files if not k.endswith("__C")]; rows = []
for t in tags:
    fa = c17.fold_aucs(J, Z[t]); se = np.sqrt(fa.var(ddof=1) * (1 / 50 + 40 / 160)); h = stats.t.ppf(0.975, 49) * se
    r = dict(model=t, auc_fold=100 * fa.mean(), lo=100 * (fa.mean() - h), hi=100 * (fa.mean() + h), auc_gab=100 * np.mean([roc_auc_score(Y, p) for p in Z[t]]), C_mediana=float(np.median(Z[t + "__C"])))
    for k in ("I2-adj", "C1", "Floor", "F2-adj"):
        d, lo, hi, p = nb(fa, FA[k]); r[f"d_{k}"], r[f"p_{k}"] = d, p
    rows.append(r)
T = pd.DataFrame(rows); T.to_csv(f"{OUT}/ringkas_A.csv", index=False)
print("== Parts A-B: per-fold AUC (50 folds), differences from I2-adj / C1 / metadata / F2-adj ==\n" + T.round(4).to_string(index=False))
prim = ["LaBraM-eq__AS", "CBraMod-eq__AS", "BENDR-eq__AS", "Comp50__AS"]; pr = []
for t in prim:
    d, lo, hi, p = nb(c17.fold_aucs(J, Z[t]), FA["I2-adj"]); pr.append(dict(uji=f"{t} - I2-adj", diff=d, lo=lo, hi=hi, p=p))
P = pd.DataFrame(pr); P["p_holm"] = holm(P["p"].values); P.to_csv(f"{OUT}/ringkas_primer.csv", index=False); print("\n== Primary family (Holm over 4 tests; two-sided, Nadeau-Bengio) ==\n" + P.round(4).to_string(index=False))
# C
f = f"{OUT}/C_null_replikasi.csv"
if os.path.exists(f):
    C = pd.read_csv(f); rows = []
    for (m, s), g in C.groupby(["model", "scen"]):
        for met in ("fold_mean", "auc"):
            x = g[met].values; mean, se = x.mean(), x.std(ddof=1) / np.sqrt(len(x)); rows.append(dict(model=m, scen=s, metrica=met, nrep=len(x), mean=mean, se=se, R1=dec_R1(mean, se)))
    R = pd.DataFrame(rows); R.to_csv(f"{OUT}/ringkas_C_R1.csv", index=False); print("\n== Part C: R1 on the noAS pipeline (features permuted, real label) ==\n" + R[R.metrica == "fold_mean"].round(4).to_string(index=False))
# D
main, allc, ref, _ = O.load_candidates(); sets = {"utama": main, "semua": allc}; rows = []; drows = []
for tag, S in sets.items():
    z = np.load(f"{OUT}/D_bruto_{tag}.npz", allow_pickle=True); names = list(z["names"]); full = z["full"]; w0 = int(z["w0"]); opt = z["opt"]; f2 = names.index("F2-adj"); rank_f2 = int((full > full[f2]).sum()) + 1
    rows.append(dict(himpunan=tag, K=len(names), pemenang=names[w0], auc_pemenang=full[w0], optimisme_rata=opt.mean(), optimisme_sd=opt.std(ddof=1), opt_p2_5=np.percentile(opt, 2.5), opt_p97_5=np.percentile(opt, 97.5), auc_terkoreksi=full[w0] - opt.mean(), auc_F2=full[f2], peringkat_F2=rank_f2, auc_F2_terkoreksi=full[f2] - opt.mean(), pemenang_boot_sama=float((z["win"] == w0).mean())))
    pm = []
    for r in ("Floor", "C1"):
        excl = [n for n in names if np.allclose(allc[n] if n in allc else main[n], ref[r], atol=1e-12)]; m = O.maxT(z, r, exclude=tuple(excl)); pm.append(m)
    ph = holm([m["p"] for m in pm])
    for m, h in zip(pm, ph):
        drows.append(dict(himpunan=tag, **m, p_holm=h))
    top = np.argsort(-full)[:6]; print(f"\n[{tag}] 6 kandidat teratas (AUC gabungan penuh): " + "; ".join(f"{names[i]} {full[i]:.4f}" for i in top))
Dt = pd.DataFrame(rows); Dm = pd.DataFrame(drows); Dt.to_csv(f"{OUT}/ringkas_D_optimisme.csv", index=False); Dm.to_csv(f"{OUT}/ringkas_D_maxT.csv", index=False)
print("\n== Part D: selection optimism (bootstrap 2,000) ==\n" + Dt.round(4).to_string(index=False)); print("\n== Part D: max-T test (studentized Reality Check; Holm over two references) ==\n" + Dm.round(4).to_string(index=False))
fp = f"{OUT}/D_perm.npy"
if os.path.exists(fp):
    r = np.load(fp); print(f"\n== Kalibrasi D (label dipermutasi, 40x): AUC pemenang {r[:, 0].mean():.4f}; optimisme {r[:, 1].mean():.4f}; AUC terkoreksi {r[:, 2].mean():.4f}; p max-T < 0.05: {(r[:, 3] < 0.05).mean():.3f} ==")
