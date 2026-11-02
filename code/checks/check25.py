import os, json, numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
D = "/home/miftah_f/RTX-5090/tuep_fresh/data"; R = "/home/miftah_f/RTX-5090/tuep_fresh/reports"; FAIL = []
def chk(ok, m): print(("PASS " if ok else "FAIL ") + m); (None if ok else FAIL.append(m))
F = pd.read_csv(f"{D}/folds_patient.csv"); T = np.load(f"{D}/p3_patient_table.npz", allow_pickle=True); S = list(T["subjects"]); y = T["y"]; pos = {s: i for i, s in enumerate(S)}
folds = [(r, k, np.array([pos[s] for s in F[(F.repeat == r) & (F.fold == k)].subject])) for r in range(10) for k in range(5)]
def load_fm(m):
    P = np.full((10, 200), np.nan); ok = True; seeds = set()
    for r, k, te in folds:
        d = json.load(open(f"{D}/round25/{m}/r{r}_k{k}.json")); ok &= set(d["test"]) == {S[i] for i in te} and d["seed"] == 1000 + 10 * r + k and d["model"] == m
        for s, v in d["test"].items(): P[r, pos[s]] = v
    return P, ok
LB, okl = load_fm("labram"); CB, okc = load_fm("cbramod")
chk(okl and okc and not np.isnan(LB).any() and not np.isnan(CB).any(), "100 JSON files: test patients = canonical fold, seeds as pre-registered, one prediction per patient per repeat")
I2 = np.load(f"{D}/round17/p1.npz")["I2"]; P3 = np.load(f"{D}/round17/p3.npz")["P3"]; NM = np.full((10, 200), np.nan)
for r, k, te in folds:
    d = json.load(open(f"{D}/round16/r{r}_k{k}.json"))["F"]["test"]; NM[r, te] = [d[S[i]] for i in te]
F2 = (I2 + P3 + NM) / 3
pooled = lambda P: np.mean([roc_auc_score(y, p) for p in P])
chk(abs(pooled(F2) - 0.72709) < 1e-5, f"F2 rebuilt from members: {pooled(F2):.5f}")
fa = lambda P: np.array([roc_auc_score(y[te], P[r, te]) for r, k, te in folds])
def nb(a, b):
    d = fa(a) - fa(b); t = d.mean() / np.sqrt(d.var(ddof=1) * (1 / 50 + 40 / 160)); return d.mean(), 2 * stats.t.sf(abs(t), 49), int((d > 0).sum())
res = [nb(F2, LB), nb(F2, CB)]; p = np.array([x[1] for x in res]); o = np.argsort(p); h = np.empty(2); run = 0
for j, i in enumerate(o): run = max(run, min(1, p[i] * (2 - j))); h[i] = run
pr = pd.read_csv(f"{R}/round25_primary.csv")
chk(np.allclose([x[0] for x in res], pr.dAUC) and np.allclose(p, pr.p) and np.allclose(h, pr.p_holm), f"primary: dAUC {[round(x[0], 4) for x in res]}, p {np.round(p, 4).tolist()}, Holm {np.round(h, 4).tolist()}, folds better {[x[2] for x in res]}")
chk(abs(pooled(LB) - 0.6098) < 5e-5 and abs(pooled(CB) - 0.6264) < 5e-5, f"pooled AUC LaBraM-FT {pooled(LB):.4f}, CBraMod-FT {pooled(CB):.4f}")
# interpretation rule part 2: within-year AUC difference > 0 (independent pair counter, repeat-mean scores)
T4 = np.load(f"{D}/p3_patient_table.npz", allow_pickle=True)
import sys; sys.path.insert(0, "/home/miftah_f/RTX-5090/tuep_fresh/code"); os.environ["CUDA_VISIBLE_DEVICES"] = ""
import matched_design as md
yr = md.subject_table().loc[S].year.values; g = np.where(yr <= 2009, "<=2009", yr.astype(int).astype(str))
def sauc(p):
    num = den = 0.0
    for s_ in np.unique(g):
        e, c = p[(g == s_) & (y == 1)], p[(g == s_) & (y == 0)]; dd = e[:, None] - c[None, :]; num += (dd > 0).sum() + 0.5 * (dd == 0).sum(); den += dd.size
    return num / den
w = [sauc(F2.mean(0)) - sauc(LB.mean(0)), sauc(F2.mean(0)) - sauc(CB.mean(0))]
chk(all(x > 0 for x in w), f"within-year AUC difference > 0: F2-LaBraM {w[0]:+.4f}, F2-CBraMod {w[1]:+.4f}")
print(f"\n{'ALL CHECKS PASSED' if not FAIL else str(len(FAIL)) + ' FAILED: ' + str(FAIL)}")
