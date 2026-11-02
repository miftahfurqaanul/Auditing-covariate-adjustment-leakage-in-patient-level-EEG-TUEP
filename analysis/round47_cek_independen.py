#!/usr/bin/env python3
"""Round 47: independent check (second code). (A) the 50 per-fold AUCs of F2 without NM are recomputed from the stored predictions with a self-written Mann--Whitney rank and compared with the raw output;
(B) the summary (mean, NB CI, difference, p, Holm) is recomputed with plain numpy/scipy and compared with ringkas_*.csv; (C) F2_adj = mean of the three views; folds = canonical split (no train-test overlap). CPU."""
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "hasil47"); sys.path.insert(0, HERE); sys.path.insert(0, "/home/miftah_f/RTX-5090/tuep_fresh/code")
import numpy as np, pandas as pd
from scipy import stats
from scipy.stats import rankdata
import round28_analisis as A
ok = fail = 0; GRP = {"A": [0, 0], "B": [0, 0], "C": [0, 0]}; g = "A"
def chk(name, cond):
    global ok, fail
    ok += bool(cond); fail += (not cond); GRP[g][0 if cond else 1] += 1; print(("PASS  " if cond else "FAIL  ") + f"[{g}] " + name, flush=True)
def auc_rank(y, s):
    r = rankdata(s); n1 = int(y.sum()); n0 = len(y) - n1; return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
P = np.load(os.path.join(HERE, "hasil30/eval_prediksi_v4.npz")); R = np.load(f"{OUT}/auc_per_lipatan.npz"); Y = A.Y; J = A.J
src = {"F2_tanpa_NM": (P["I2_adj"] + P["P3_adj"]) / 2, "F2_adj": P["F2_adj"], "C1": P["C1"], "Floor": P["Floor"], "I2_adj": P["I2_adj"], "P3_adj": P["P3_adj"]}
for k, s in src.items():
    fa = np.array([auc_rank(Y[te], s[r, te]) for r, _, _, te in J]); chk(f"{k}: 50 per-fold AUCs recomputed with ranks = raw output (max difference {np.abs(fa - R['fold_' + k]).max():.1e})", np.abs(fa - R["fold_" + k]).max() < 1e-9)
    po = np.mean([auc_rank(Y[~np.isnan(s[r])], s[r][~np.isnan(s[r])]) for r in range(10)]); chk(f"{k}: AUC gabungan {po:.5f} = raw output", abs(po - float(R["pooled_" + k])) < 1e-9)
g = "B"; T = pd.read_csv(f"{OUT}/ringkas_f2_tanpa_nm.csv").set_index("model"); D = pd.read_csv(f"{OUT}/ringkas_selisih.csv").set_index("kontras"); tc = stats.t.ppf(0.975, 49)
for k in src:
    f = R["fold_" + k]; se = np.sqrt(f.var(ddof=1) * (1 / 50 + 0.25)); chk(f"{k}: mean {100 * f.mean():.2f} and NB CI = summary", abs(100 * f.mean() - T.loc[k, "auc_fold"]) < 1e-9 and abs(100 * (f.mean() - tc * se) - T.loc[k, "lo"]) < 1e-9 and abs(100 * (f.mean() + tc * se) - T.loc[k, "hi"]) < 1e-9)
pv = {}
for k in ["F2_adj", "C1", "Floor", "I2_adj"]:
    d = R["fold_F2_tanpa_NM"] - R["fold_" + k]; se = np.sqrt(d.var(ddof=1) * (1 / 50 + 0.25)); t = d.mean() / se; p = 2 * stats.t.sf(abs(t), 49); pv[k] = p
    row = D.loc[f"F2_tanpa_NM - {k}"]; chk(f"difference from {k}: {d.mean():+.4f}, p {p:.4f} = ringkasan", abs(d.mean() - row.selisih) < 1e-9 and abs(p - row.p) < 1e-9 and abs(d.mean() - tc * se - row.lo) < 1e-9)
q = np.array([pv["F2_adj"], pv["C1"], pv["Floor"]]); o = np.argsort(q); hh = np.empty(3); run = 0.0
for i_, i in enumerate(o): run = max(run, (3 - i_) * q[i]); hh[i] = min(1.0, run)
for i, k in enumerate(["F2_adj", "C1", "Floor"]): chk(f"Holm {k}: {hh[i]:.4f} = ringkasan", abs(hh[i] - D.loc[f"F2_tanpa_NM - {k}", "holm"]) < 1e-9)
g = "C"; chk("F2_adj = mean of I2_adj, P3_adj, NM_adj (1e-9)", np.abs(P["F2_adj"] - (P["I2_adj"] + P["P3_adj"] + P["NM_adj"]) / 3).max() < 1e-9)
chk("50 folds; no train-test overlap; each patient tested once per repetition", len(J) == 50 and all(not (set(tr) & set(te)) for _, _, tr, te in J) and (np.bincount(np.concatenate([te for r, _, _, te in J if r == 0]), minlength=200) == 1).all())
chk("full F2: per-fold mean = Table 4 (69.74)", abs(100 * R["fold_F2_adj"].mean() - 69.74) < 0.006)
print(f"\n{ok} PASS, {fail} FAIL"); print({k: tuple(v) for k, v in GRP.items()})
if not fail: print(f"ALL {sum(1 for v in GRP.values() if v[0] and not v[1])} CHECK GROUPS PASSED")
sys.exit(1 if fail else 0)
