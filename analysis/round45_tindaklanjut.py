#!/usr/bin/env python3
"""Round 45 (CPU): see PRAREGISTRASI_putaran45.md. A = F2-Clean vs F2 within strata (bruto_strata.npz); B = patient-level bootstrap sensitivity.
Usage: CUDA_VISIBLE_DEVICES="" python round45_tindaklanjut.py   Raw output to hasil45/*.csv, then a summary to stdout."""
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "hasil45"); os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, HERE); sys.path.insert(0, "/home/miftah_f/RTX-5090/tuep_fresh/code")
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
import round28_analisis as A
Y = A.Y; N = len(Y)

def pv(d):
    d = d[~np.isnan(d)]; return max(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())), 1 / len(d))
def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); h = np.empty(len(p)); run = 0
    for r, i in enumerate(o):
        run = max(run, min(1, p[i] * (len(p) - r))); h[i] = run
    return h

# ---- Analisis A
z = np.load(os.path.join(HERE, "hasil44/bruto_strata.npz"), allow_pickle=True); names = list(z["names"]); ks = list(z["ks"]); B = z["boot"]; P = z["point"]
rows = []
contr = [("F2-Clean", "F2-adj"), ("I2__rank-noAS-D2", None)]  # second contrast: I2-adj is not in bruto_strata -> computed separately
for a_, b_ in contr[:1]:
    for i, k in enumerate(ks[:3]):
        ja, jb = names.index(a_), names.index(b_); d = B[:, i, ja] - B[:, i, jb]
        rows.append(dict(kontras=f"{a_} - {b_}", k=int(k), titik=P[i, ja] - P[i, jb], lo=np.nanpercentile(d, 2.5), hi=np.nanpercentile(d, 97.5), p=pv(d), n_boot=int((~np.isnan(d)).sum())))
# I2 contrast: patient-level bootstrap again (seed 45001) with the pair function of round 41
import round41_strata_kaliper as S41
I2ADJ = np.load(os.path.join(HERE, "hasil29/B_prediksi_data_asli.npz"))["V4_AS"].astype(float)
MA = np.load(os.path.join(HERE, "hasil44/A_prediksi.npz")); I2R = np.asarray(MA["I2__rank-noAS-D2"], float)
g = S41.STR["year x device"]; groups = [np.where(g == s)[0] for s in np.unique(g)]
S2 = np.column_stack([np.nanmean(I2R, 0), np.nanmean(I2ADJ, 0)]); ei, ni = S41.make_pairs(Y, g, np.arange(N)); A0, _ = S41.auc_by_k(S2, ei, ni)
rng = np.random.default_rng(45001); Bb = np.full((2000, len(S41.KS), 2), np.nan)
for b in range(2000):
    idx = np.concatenate([rng.choice(v, len(v)) for v in groups]); e, n_ = S41.make_pairs(Y, g, idx); Bb[b] = S41.auc_by_k(S2, e, n_)[0]
for i, k in enumerate(S41.KS[:3]):
    d = Bb[:, i, 0] - Bb[:, i, 1]
    rows.append(dict(kontras="I2__rank-noAS-D2 - I2-adj (bootstrap baru seed 45001)", k=int(k), titik=A0[i, 0] - A0[i, 1], lo=np.nanpercentile(d, 2.5), hi=np.nanpercentile(d, 97.5), p=pv(d), n_boot=int((~np.isnan(d)).sum())))
RA = pd.DataFrame(rows); RA["p_holm"] = holm(RA["p"].values); RA.to_csv(os.path.join(OUT, "A_selisih_strata.csv"), index=False)
print("== A: AUC difference within strata (year x device, age caliper) ==\n" + RA.round(4).to_string(index=False))

# ---- Analisis B
h30 = np.load(os.path.join(HERE, "hasil30/eval_prediksi_v4.npz"))
S = {"F2-adj": h30["F2_adj"].astype(float), "I2-adj": I2ADJ, "C1": h30["C1"].astype(float), "Floor": h30["Floor"].astype(float), "CBraMod-FT": h30["CBraMod_FT"].astype(float)}
pos, neg = np.where(Y == 1)[0], np.where(Y == 0)[0]
def auc_all(Sc, idx):
    return np.mean([roc_auc_score(Y[idx], Sc[r][idx]) for r in range(Sc.shape[0])])
pt = {k: np.mean([roc_auc_score(Y, v[r]) for r in range(10)]) for k, v in S.items()}
rng = np.random.default_rng(45002); BB = {k: np.empty(2000) for k in S}
for b in range(2000):
    idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
    for k, v in S.items(): BB[k][b] = auc_all(v, idx)
nbrows = pd.read_csv(os.path.join(HERE, "hasil30/eval_selisih_per_lipatan.csv")).set_index("test")
nbmap = {"I2-adj": "F2-adj - I2-adj", "C1": "F2-adj - C1", "Floor": "F2-adj - Floor", "CBraMod-FT": "F2-adj - CBraMod-FT"}
rows = []
for ref in ["I2-adj", "C1", "Floor", "CBraMod-FT"]:
    d = BB["F2-adj"] - BB[ref]; nbr = nbrows.loc[nbmap[ref]]
    rows.append(dict(selisih=f"F2-adj - {ref}", titik=pt["F2-adj"] - pt[ref], lo=np.percentile(d, 2.5), hi=np.percentile(d, 97.5), p_boot=pv(d), lebar_boot=np.percentile(d, 97.5) - np.percentile(d, 2.5),
                     nb_diff=nbr["diff"], nb_lo=nbr["lo"], nb_hi=nbr["hi"], nb_p=nbr["p"], lebar_nb=nbr["hi"] - nbr["lo"]))
RB = pd.DataFrame(rows); RB["p_holm_boot"] = holm(RB["p_boot"].values); RB.to_csv(os.path.join(OUT, "B_bootstrap_pasien.csv"), index=False)
print("\n== B: patient-level bootstrap vs Nadeau-Bengio t-test (pooled AUC; per fold) ==\n" + RB.round(4).to_string(index=False))
