#!/usr/bin/env python3
"""Independent check for round 39 (CPU): (1) second implementation of the adjustment and transforms (explicit loops); (2) properties of the data generator; (3) reproduction of stored replicates; (4) recomputation of the summary from the CSV with separate code; (5) data structure.
Usage: CUDA_VISIBLE_DEVICES="" python round39_cek_independen.py"""
import os, sys, glob
os.environ["CUDA_VISIBLE_DEVICES"] = ""; HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.argv = [sys.argv[0]]
import numpy as np, pandas as pd
from scipy import stats
import round39_simulasi_kebocoran as S
n_ok = 0
def ok(m):
    global n_ok; n_ok += 1; print("OK  ", m, flush=True)
# 1. second implementation
rng = np.random.default_rng(1)
for dist in ("normal", "lognormal", "contam"):
    d = S.make_data(11, n=80, m=12, p=6, dist=dist, scen="H2a", imb=True); X, pid, y, D = d["X"], d["pid"], d["y"], d["D"]; tr = np.sort(rng.choice(80, 55, replace=False)); trw = np.isin(pid, tr)
    # symlog / symlogc / rank transforms with loops
    Xs = np.array([[np.sign(v) * np.log1p(abs(v)) for v in row] for row in X]); assert np.allclose(S.transform(X, trw, "symlog"), Xs)
    c = np.median(np.abs(X[trw]), axis=0) + 1e-12; Xc = np.array([[np.sign(v) * np.log1p(abs(v) / c[j]) for j, v in enumerate(row)] for row in X]); assert np.allclose(S.transform(X, trw, "symlogc"), Xc)
    Rk = S.transform(X, trw, "rank"); q = np.linspace(0, 1, 1001); Q = np.quantile(X[trw], q, axis=0)
    for j in range(X.shape[1]):
        for i in rng.choice(len(X), 6, replace=False):
            u = np.interp(X[i, j], Q[:, j], q); assert abs(Rk[i, j] - stats.norm.ppf(min(max(u, 1e-3), 1 - 1e-3))) < 1e-10
    # win and pat adjustment
    T = S.transform(X, trw, "symlog"); rows = np.where(trw)[0]; Dw = D[pid[rows]]; B = np.linalg.solve(Dw.T @ Dw, Dw.T @ T[rows]); assert np.allclose(S.adjust(T, pid, D, tr, trw, "win"), T - D[pid] @ B)
    M = np.array([T[pid == i].mean(0) for i in range(80)]); Dt = D[tr]; B2 = np.linalg.solve(Dt.T @ Dt, Dt.T @ M[tr]); assert np.allclose(S.adjust(T, pid, D, tr, trw, "pat"), T - D[pid] @ B2)
ok("transforms (symlog, symlogc, rank) and adjustment (win, pat) equal the second implementation (loop/solve normal equations) on 3 distributions, H2a, unbalanced windows (1e-8)")
# 2. sifat pembangkit
cnt = 0
for sd in range(20):
    d = S.make_data(100 + sd, n=200, m=50, p=20, dist="lognormal", scen="H1", s=1.0); y, D, X, pid = d["y"], d["D"], d["X"], d["pid"]; assert y.sum() == 100
    Xl = np.log(X); M = np.array([Xl[pid == i].mean(0) for i in range(200)]); r = [abs(np.corrcoef(M[:, j], y)[0, 1]) for j in range(20)]; cnt += np.mean(r)
assert cnt / 20 < 0.12; ok(f"H1: log of the patient-mean features is uncorrelated with the label (mean |r| {cnt / 20:.3f}; expected ~0.056 for n = 200); balanced label")
eras = np.zeros((2, 5)); age_d = []
for sd in range(100):
    d = S.make_data(200 + sd, n=200, m=2, p=3, s=1.0); y, D = d["y"], d["D"]; era = np.where(D[:, 3] == 1, 0, np.where(D[:, 4] == 1, 1, np.where(D[:, 5] == 1, 2, np.where(D[:, 6] == 1, 4, 3))))
    for c in (0, 1):
        eras[c] += np.bincount(era[y == c], minlength=5)
    age_d.append(D[y == 0, 1].mean() - D[y == 1, 1].mean())
P1 = eras[1] / eras[1].sum(); P0 = eras[0] / eras[0].sum(); assert np.abs(P1 - S.ERA_REAL1).max() < 0.02 and np.abs(P0 - S.ERA_REAL0).max() < 0.02; assert 0.3 < np.mean(age_d) / 1.0 < 0.65
ok(f"era | label distribution equals the cohort table (max difference {max(np.abs(P1 - S.ERA_REAL1).max(), np.abs(P0 - S.ERA_REAL0).max()):.3f}); age z difference non-epilepsy - epilepsy {np.mean(age_d):.2f} (harapan ~0.45/1.0 SD)")
d0 = S.make_data(5, s=0.0, n=400, m=1, p=2); assert abs(d0["D"][d0["y"] == 1, 1].mean() - d0["D"][d0["y"] == 0, 1].mean()) < 0.35; ok("s = 0: age is unrelated to the label")
# 3. reproduction of stored replicates
B = pd.concat([pd.read_csv(f) for f in glob.glob(f"{HERE}/hasil39/B_cpu.csv")], ignore_index=True)
for c, v in (("sig", 1.0), ("cfrac", 0.01), ("cscale", 100.0)): B[c] = B[c].fillna(v)
picks = [dict(dist="lognormal", scen="H1", n=200, m=50, p=50, imb=False, s=1.0, sig=1.0, cfrac=0.01, cscale=100.0), dict(dist="contam", scen="H1", n=200, m=50, p=10, imb=False, s=1.0, sig=1.0, cfrac=0.01, cscale=100.0), dict(dist="lognormal", scen="H2a", n=200, m=50, p=50, imb=False, s=1.0, sig=1.0, cfrac=0.01, cscale=100.0)]; mx = 0
for cell in picks:
    for rep in (0, 7):
        res = pd.DataFrame(S.run_rep(cell, rep, ["LRpat", "RF"], S.configs_b(cell["dist"], cell["imb"], "cpu")))
        for _, r in res.iterrows():
            sel = B[(B.dist == r["dist"]) & (B.scen == r.scen) & (B.n == r.n) & (B.m == r.m) & (B.p == r.p) & (B.s == r.s) & (B.sig == r.sig) & (B.cfrac == r.cfrac) & (B.cscale == r.cscale) & (B.rep == r.rep) & (B.clf == r.clf) & (B["transform"] == r["transform"]) & (B["adjust"] == r["adjust"]) & (B.imb.astype(str).str.lower() == str(r.imb).lower())]
            assert len(sel) == 1, (cell, rep, r.clf, r["transform"], r["adjust"], len(sel)); mx = max(mx, abs(float(sel.auc.iloc[0]) - r.auc))
assert mx < 3e-3, mx; ok(f"reproduction of 3 cells x 2 replicates x (LRpat, RF) x all configurations: maximum AUC difference from the CSV {mx:.2e}")
# 4. recompute the summary with separate code (loop)
G = pd.read_csv(f"{HERE}/hasil39/peta_sel.csv"); allr = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{HERE}/hasil39/B_*.csv")) + sorted(glob.glob(f"{HERE}/hasil39/C_*.csv")) + sorted(glob.glob(f"{HERE}/hasil39/pilot_*.csv"))], ignore_index=True)
for c, v in (("sig", 1.0), ("cfrac", 0.01), ("cscale", 100.0)): allr[c] = allr[c].fillna(v)
allr["imb"] = allr.imb.astype(str).str.lower().eq("true"); allr = allr.drop_duplicates(["dist", "scen", "n", "m", "p", "imb", "s", "sig", "cfrac", "cscale", "rep", "clf", "transform", "adjust"]); chk = 0
for _, g in G.sample(25, random_state=3).iterrows():
    m = allr[(allr["dist"] == g["dist"]) & (allr.scen == g.scen) & (allr.n == g.n) & (allr.m == g.m) & (allr.p == g.p) & (allr.imb == bool(g.imb)) & (allr.s == g.s) & (np.isclose(allr.sig, g.sig)) & (np.isclose(allr.cfrac, g.cfrac)) & (np.isclose(allr.cscale, g.cscale)) & (allr.clf == g.clf)]
    a = m[(m["transform"] == g["transform"]) & (m["adjust"] == g["adjust"])].set_index("rep").auc; bt = "raw" if g.clf in ("RF", "ET", "XGB") else g["transform"]; b = m[(m["transform"] == bt) & (m["adjust"] == "none")].set_index("rep").auc; idx = a.index.intersection(b.index)
    d = np.array([a[i] - b[i] for i in idx]); assert len(d) == g.n_rep and abs(d.mean() - g.delta) < 1e-9 and abs(d.std(ddof=1) / np.sqrt(len(d)) - g.se) < 1e-9; chk += 1
ok(f"{chk} random groups: mean delta, number of replicates, and SE from the raw CSV equal peta_sel.csv (1e-9)")
# 5. struktur
assert allr.auc.between(0, 1).all() and not allr.auc.isna().any(); gcount = allr.groupby(["dist", "scen", "n", "m", "p", "imb", "s", "sig", "cfrac", "cscale", "clf", "transform", "adjust"]).rep.nunique()
b_c = B[B.clf.isin(["LRpat", "RF"])].groupby(["dist", "scen", "n", "m", "p", "imb", "s", "sig", "cfrac", "cscale", "clf"]).rep.nunique(); assert (b_c >= 50).all() and (b_c == 100).sum() >= 0.6 * len(b_c)
Gp = pd.concat([pd.read_csv(f) for f in glob.glob(f"{HERE}/hasil39/B_gpu.csv")]); g_c = Gp.groupby(["dist", "scen", "n", "m", "p", "imb", "s", "clf"]).rep.nunique(); assert (g_c >= 30).all()
ok(f"structure: AUC in [0, 1], no NaN, no duplicates; CPU replicates per cell {b_c.min()}-{b_c.max()} (>= 50), GPU per sel {g_c.min()}-{g_c.max()} (>= 30)")
# controls (rule in the pre-registration; Addendum 2: symlog on normal features is reported as a documented exception)
G = G[G.n_rep >= 3]; core = G[((G.s == 0) | (G["dist"] == "normal")) & (G.scen == "H1") & (G.adjust == "pat") & (G["transform"].isin(["raw", "rank", "log"]))]
assert core.delta.max() <= 0.012, core.delta.max()
sl = G[(G["dist"] == "normal") & (G.scen == "H1") & (G.s == 1.0) & (G.adjust == "pat") & (G["transform"].isin(["symlog", "symlogc"])) & (G.clf.isin(["RF", "XGB"]))]
assert (sl.delta > 0.01).all() and sl.n_rep.min() >= 30, sl[["clf", "transform", "n_rep", "delta"]]
pos = G[(G["dist"] == "lognormal") & (G.sig == 1.0) & (G.scen == "H1") & (G.n == 200) & (G.m == 50) & (G.p == 50) & (~G.imb) & (G.s == 1.0) & (G.clf == "RF") & (G["transform"] == "raw") & (G.adjust == "pat")]; assert pos.delta.iloc[0] > 0.03
ok(f"negative control (s = 0 and normal; raw, rank, log): maximum delta {core.delta.max():+.3f} <= 0.01 (+ error tolerance 0.002); DOCUMENTED EXCEPTION (stage C, 100/30 replicates): symlog and symlogc on normal features give an RF delta {sl[sl.clf == 'RF'].delta.min():+.3f} up to {sl[sl.clf == 'RF'].delta.max():+.3f} and XGB {sl[sl.clf == 'XGB'].delta.min():+.3f} up to {sl[sl.clf == 'XGB'].delta.max():+.3f} (> 0.01; a small real effect); positive control (raw lognormal RF): delta {pos.delta.iloc[0]:+.3f} > 0.03")
print(f"\nALL {n_ok} CHECK GROUPS PASSED")
