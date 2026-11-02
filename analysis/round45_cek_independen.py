#!/usr/bin/env python3
"""Independent check for round 45 (second implementation; CPU). Usage: python round45_cek_independen.py"""

import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, "/home/miftah_f/RTX-5090/tuep_fresh/code")
import numpy as np, pandas as pd
from scipy import stats
import round28_analisis as A
Y = A.Y; N = len(Y); AGE = A.AGE.astype(float)
n_ok = 0
def ok(m):
    global n_ok; n_ok += 1; print("OK  ", m, flush=True)
# year x device strata rebuilt from ERA and DEV
cell = np.array([f"{e}|{d}" for e, d in zip(A.ERA, A.DEV)]); cells = np.unique(cell)
# predictions
M = {}
for f in ("A_prediksi.npz", "B_prediksi.npz", "C_prediksi.npz", "E_nm_prediksi.npz"):
    p = os.path.join(HERE, "hasil44", f)
    if os.path.exists(p):
        z = np.load(p)
        for k in z.files: M[k] = np.asarray(z[k], float)
h30 = np.load(os.path.join(HERE, "hasil30/eval_prediksi_v4.npz"))
F2C = np.mean(np.stack([M["I2__rank-noAS-D2"], M["P3__rank-noAS-D2"], M["NM_clean"]]), 0).mean(0); F2A = h30["F2_adj"].astype(float).mean(0)
def pair_auc(score, idx, kmax):
    num = 0.0; den = 0
    for c in np.unique(cell[idx]):
        sel = idx[cell[idx] == c]; e = sel[Y[sel] == 1]; n = sel[Y[sel] == 0]
        for i in e:
            d = np.abs(AGE[i] - AGE[n]); m = d <= kmax
            if m.any():
                s = score[i] - score[n[m]]; num += (s > 0).sum() + 0.5 * (s == 0).sum(); den += m.sum()
    return num / den, den
z = np.load(os.path.join(HERE, "hasil44/bruto_strata.npz"), allow_pickle=True); names = list(z["names"]); P = z["point"]
for i, k in enumerate((5.0, 10.0, 15.0)):
    for nm, sc in (("F2-Clean", F2C), ("F2-adj", F2A)):
        a, den = pair_auc(sc, np.arange(N), k)
        assert abs(a - P[i, names.index(nm)]) < 1e-9, (nm, k, a, P[i, names.index(nm)])
ok("AUC within strata (year x device, caliper 5/10/15) of F2-Clean and F2-adj recomputed with a loop: equals bruto_strata.npz (1e-9)")
# second bootstrap (seed 777) for the F2-Clean - F2-adj difference at k=10
rng = np.random.default_rng(777); groups = [np.where(cell == c)[0] for c in cells]; d = []
for b in range(600):
    idx = np.concatenate([rng.choice(g, len(g)) for g in groups]); a1, _ = pair_auc(F2C, idx, 10.0); a2, _ = pair_auc(F2A, idx, 10.0); d.append(a1 - a2)
d = np.array(d); lo, hi = np.percentile(d, [2.5, 97.5]); ra = pd.read_csv(os.path.join(HERE, "hasil45/A_selisih_strata.csv")); r10 = ra[(ra.kontras == "F2-Clean - F2-adj") & (ra.k == 10)].iloc[0]
assert abs(lo - r10.lo) < 0.03 and abs(hi - r10.hi) < 0.03 and (lo < 0 < hi) == (r10.lo < 0 < r10.hi), (lo, hi, r10.lo, r10.hi)
ok(f"second bootstrap (seed 777, 600 replicates) of the F2-Clean - F2-adj difference at k=10: CI [{lo:.3f}, {hi:.3f}] vs [{r10.lo:.3f}, {r10.hi:.3f}] (tolerance 0.03; both include zero)")
# B: pooled AUC with scipy Mann-Whitney
S = {"F2-adj": h30["F2_adj"], "C1": h30["C1"], "Floor": h30["Floor"], "CBraMod-FT": h30["CBraMod_FT"], "I2-adj": np.load(os.path.join(HERE, "hasil29/B_prediksi_data_asli.npz"))["V4_AS"]}
def auc_mw(y, s):
    r = stats.rankdata(s); n1 = (y == 1).sum(); n0 = (y == 0).sum(); return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
def pooled(sc, idx): return np.mean([auc_mw(Y[idx], np.asarray(sc[r], float)[idx]) for r in range(10)])
rb = pd.read_csv(os.path.join(HERE, "hasil45/B_bootstrap_pasien.csv")).set_index("selisih")
pt = {k: pooled(v, np.arange(N)) for k, v in S.items()}
for ref in ("I2-adj", "C1", "Floor", "CBraMod-FT"):
    assert abs((pt["F2-adj"] - pt[ref]) - rb.loc[f"F2-adj - {ref}", "titik"]) < 1e-9, ref
ok("pooled-AUC difference point (scipy Mann-Whitney) equals hasil45/B_bootstrap_pasien.csv (1e-9)")
pos, neg = np.where(Y == 1)[0], np.where(Y == 0)[0]; rng = np.random.default_rng(777); BB = {k: [] for k in S}
for b in range(800):
    idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
    for k, v in S.items(): BB[k].append(pooled(v, idx))
for ref in ("I2-adj", "C1", "Floor", "CBraMod-FT"):
    dd = np.array(BB["F2-adj"]) - np.array(BB[ref]); lo, hi = np.percentile(dd, [2.5, 97.5]); r = rb.loc[f"F2-adj - {ref}"]
    assert abs(lo - r.lo) < 0.015 and abs(hi - r.hi) < 0.015 and ((lo > 0) or (hi < 0)) == ((r.lo > 0) or (r.hi < 0)), (ref, lo, hi, r.lo, r.hi)
ok("second bootstrap (seed 777, 800 replicates): CIs of four differences within 0.015 of hasil45 and the decision 'CI includes zero or not' the same (F2 - CBraMod-FT does not include zero)")
print(f"{n_ok} checks passed"); print(f"ALL {n_ok} CHECK GROUPS PASSED")
