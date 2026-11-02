#!/usr/bin/env python3
"""Round 46: independent check (second code). (A) The summary is recomputed from hasil46/H1_auc_per_lipatan.npz with plain numpy and compared with ringkas_R1.csv;
(B) three replicates (r = 0, 17, 53) are retrained sequentially (without joblib) with the AUC from a self-written Mann--Whitney rank and compared exactly with the raw output;
(C) the decision rule is reapplied to the V* columns of hasil29/A_kebocoran_replikasi.csv and matches the values of the leakage table (round 29: 50.6/56.0/56.1/55.7/52.5/50.4). CPU. Usage: python round46_cek_independen.py"""
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""; os.environ.setdefault("OMP_NUM_THREADS", "1")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
from scipy.stats import rankdata
import round28_analisis as A, round29_sensitivitas as S
OUT = os.path.join(HERE, "hasil46"); ok = fail = 0


GRP = {"A": [0, 0], "B": [0, 0], "C": [0, 0]}; g = "A"   # groups: A summary from raw output, B sequential retraining, C rule on the CSV of round 29


def chk(name, cond):
    global ok, fail
    ok += bool(cond); fail += (not cond); GRP[g][0 if cond else 1] += 1; print(("PASS  " if cond else "FAIL  ") + f"[{g}] " + name, flush=True)


def auc_rank(y, s):
    r = rankdata(s); n1 = int(y.sum()); n0 = len(y) - n1; return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def dec(m, se):
    if m <= 0.520 and m + 1.645 * se <= 0.540: return "tidak ada bukti"
    return "kebocoran" if m > 0.55 else "tidak pasti"


Z = np.load(f"{OUT}/H1_auc_per_lipatan.npz", allow_pickle=True); FA = Z["fold_auc"]; V = [str(x) for x in Z["variants"]]
chk("raw output shape 100 x 6 x 50 without NaN and in [0,1]", FA.shape == (100, 6, 50) and np.isfinite(FA).all() and FA.min() >= 0 and FA.max() <= 1)
chk("variants in order C1,V0,V1,V2,V3,V4", V == ["C1", "V0", "V1", "V2", "V3", "V4"])
R = pd.read_csv(f"{OUT}/ringkas_R1.csv").set_index("variant"); P = pd.read_csv(os.path.join(HERE, "hasil29/A_kebocoran_replikasi.csv")); P = P[P.scenario == "H1_all"].sort_values("rep")
for i, v in enumerate(V):
    fm = FA[:, i, :].mean(1); m = fm.mean(); se = fm.std(ddof=1) / 10.0
    chk(f"{v}: per-fold mean {m:.5f} = ringkasan", abs(m - R.loc[v, "fold_mean_mean"]) < 1e-12)
    chk(f"{v}: SE {se:.5f} = ringkasan", abs(se - R.loc[v, "fold_mean_se"]) < 1e-12)
    chk(f"{v}: per-fold decision = {dec(m, se)}", dec(m, se) == R.loc[v, "fold_mean_dec"])
    g = "C"; pm = P[v].mean(); pse = P[v].std(ddof=1) / 10.0
    chk(f"{v}: pooled decision {dec(pm, pse)} from the CSV of round 29 = summary", dec(pm, pse) == R.loc[v, "pooled_dec"] and abs(pm - R.loc[v, "pooled_mean"]) < 1e-12)
    chk(f"{v}: decision changes = {dec(m, se) != dec(pm, pse)}", (dec(m, se) != dec(pm, pse)) == bool(R.loc[v, "keputusan_berubah"]))
    g = "A"
g = "C"; chk("Leakage table of round 29 (pooled, percent, 1 decimal) C1 50.6 V0 56.0 V1 56.1 V2 55.7 V3 52.5 V4 50.4", [f"{100 * P[v].mean() + 1e-9:.1f}" for v in V] == ["50.6", "56.0", "56.1", "55.7", "52.5", "50.4"])
g = "B"
for r in (0, 17, 53):
    Xs = A.shuffle_cols(A.BP, [np.arange(A.N)], np.random.default_rng(29100 + r)); out = [S.fold_job(Xs, f, tr, te, S.VARIANTS, False) for f, (_, _, tr, te) in enumerate(A.J)]
    for i, v in enumerate(V):
        fa = np.array([auc_rank(A.Y[te], o[v]) for (_, _, _, te), o in zip(A.J, out)])
        chk(f"replicate {r} {v}: 50 per-fold AUCs retrained sequentially = raw output (max difference {np.abs(fa - FA[r, i]).max():.1e})", np.abs(fa - FA[r, i]).max() < 1e-9)
print(f"\n{ok} PASS, {fail} FAIL"); print({k: tuple(v) for k, v in GRP.items()}); ng = sum(1 for v in GRP.values() if v[0] and not v[1])
if not fail: print(f"ALL {ng} CHECK GROUPS PASSED")
sys.exit(1 if fail else 0)
