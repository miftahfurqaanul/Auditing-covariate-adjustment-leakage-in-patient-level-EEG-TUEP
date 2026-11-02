#!/usr/bin/env python3
"""Independent check for round 40 (addendum2 item 7). Every check uses an assert, except the soft check (reported). Usage: CUDA_VISIBLE_DEVICES="" python round40_cek_independen.py"""
import os, sys, glob
os.environ["CUDA_VISIBLE_DEVICES"] = ""
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); OUT = os.path.join(HERE, "hasil40")
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import round39_simulasi_kebocoran as R
import round40_simulasi_perlipatan as R40
import round40_ringkas as S
n_ok = 0; K = ["dist", "scen", "n", "m", "p", "imb", "s", "sig", "cfrac", "cscale", "rep", "clf", "transform", "adjust"]


def ok(m):
    global n_ok
    n_ok += 1; print("OK  ", m, flush=True)


# 1. built-in check functions (orthogonality, test data/labels not used, diagnostics) and identical to round 39
R.checks(); R40.checks(); ok("built-in checks of round39 and round40 (residual orthogonality, test data/labels do not change training features/predictions, diagnostics without test data, run_rep40 = run_rep of round 39)")

# 2. H40-0 reproduction CPU (hard) and GPU (soft)
for st, old in (("A", "B_cpu"), ("C", "C_cpu")):
    f = f"{OUT}/{st}_cpu.csv"
    if not os.path.exists(f):
        print("passed (not yet available):", f); continue
    a = pd.read_csv(f); b = pd.read_csv(f"{HERE}/hasil39/{old}.csv"); m = a[K + ["auc"]].merge(b[K + ["auc"]], on=K, suffixes=("_40", "_39")); assert len(m) == len(a), (st, len(m), len(a))
    d = (m["auc_40"] - m["auc_39"]).abs().max(); assert d <= 1e-9, (st, d); ok(f"H40-0 tahap {st}: {len(m)} rows, maximum pooled AUC difference {d:.1e} against {old}.csv (batas 1e-9)")
for st, old in (("GB", "B_gpu"), ("GC", "C_gpu")):
    f = f"{OUT}/{st}_gpu.csv"
    if os.path.exists(f):
        a = pd.read_csv(f); b = pd.read_csv(f"{HERE}/hasil39/{old}.csv"); m = a[K + ["auc"]].merge(b[K + ["auc"]], on=K, suffixes=("_40", "_39")); d = (m["auc_40"] - m["auc_39"]).abs()
        print(f"SOFT H40-0 GPU {st}: {len(m)}/{len(a)} paired rows; max difference {d.max():.4f}; mean {d.mean():.4f} (no hard threshold)")

# 3. recomputation of per-fold and pooled AUC from stored predictions (replicates 0-2, basic cell)
Df = pd.read_csv(f"{OUT}/A_cpu.csv") if os.path.exists(f"{OUT}/A_cpu.csv") else None; nchk = 0
if Df is not None:
    for fz in sorted(glob.glob(f"{OUT}/pred/*.npz")):
        z = np.load(fz); dist, rest = os.path.basename(fz)[:-4].split("_rep"); rep, clf = rest.split("_"); rep = int(rep); seed = int(z["seed"])
        data = R.make_data(seed, n=200, m=50, p=50, dist=dist, scen="H1", s=1.0, imb=False, sig=1.0, cfrac=0.01, cscale=100.0); y = data["y"]; folds = list(StratifiedKFold(5, shuffle=True, random_state=seed).split(np.zeros(200), y))
        sub = Df[(Df.dist == dist) & (Df.rep == rep) & (Df.clf == clf) & (Df.scen == "H1") & (Df.n == 200) & (Df.m == 50) & (Df.p == 50) & (~Df.imb) & (Df.s == 1.0) & (Df.sig == 1.0) & (Df.cfrac == 0.01) & (Df.cscale == 100.0)]
        for key in z.files:
            if key == "seed":
                continue
            c, t, a_ = key.split("__"); r = sub[(sub["transform"] == t) & (sub["adjust"] == a_)]
            if len(r) == 0:
                continue
            r = r.iloc[0]; P = z[key]; assert abs(roc_auc_score(y, P) - r["auc"]) < 1e-12; assert max(abs(roc_auc_score(y[te], P[te]) - r[f"fold_auc_{i}"]) for i, (tr, te) in enumerate(folds)) < 1e-12; nchk += 1
    ok(f"pooled AUC and 5 per-fold AUCs recomputed with sklearn from {nchk} stored out-of-fold predictions (1e-12)")

    # 4. summary recomputed from the raw rows
    T = pd.read_csv(f"{OUT}/ringkas_sel_R1R2.csv") if os.path.exists(f"{OUT}/ringkas_sel_R1R2.csv") else None
    if T is not None:
        D = S.load(); rng = np.random.default_rng(1); idx = rng.choice(len(T), size=min(25, len(T)), replace=False)
        for i in idx:
            r = T.iloc[i]; g = D[(D.dist == r.dist) & (D.scen == r.scen) & (D.n == r.n) & (D.m == r.m) & (D.p == r.p) & (D.imb == r.imb) & (D.s == r.s) & (D.sig == r.sig) & (D.cfrac == r.cfrac) & (D.cscale == r.cscale) & (D.clf == r.clf) & (D["transform"] == r["transform"]) & (D["adjust"] == r["adjust"])]
            col = "auc" if r.metrica == "auc" else "fold_mean"; assert len(g) == r.nrep and abs(g[col].mean() - r.auc_mean) < 1e-9
            fm = g[[f"fold_auc_{j}" for j in range(5)]].mean(axis=1); assert abs(fm.mean() - g["fold_mean"].mean()) < 1e-12
        ok(f"{len(idx)} random rows of ringkas_sel_R1R2.csv recomputed from the raw rows (1e-9)")

# 5. tahap NM
f = f"{OUT}/nm_perlipatan_replikasi.csv"
if os.path.exists(f):
    d = pd.read_csv(f); FCs = [f"fold_auc_{i}" for i in range(5)]; assert d[FCs].notna().all().all() and ((d[FCs] >= 0) & (d[FCs] <= 1)).all().all() and not d.duplicated(["scenario", "rep", "variant"]).any()
    cnt = d.groupby(["scenario", "variant"]).size(); assert (cnt == cnt.iloc[0]).all(); assert (d["auc"] - d[FCs].mean(axis=1)).abs().max() < 0.2
    ok(f"NM: {len(d)} rows, {cnt.iloc[0]} replicates per scenario x variant, AUC in [0,1], no duplicates, pooled AUC and fold mean consistent (max difference < 0.2)")
# 6. tahap ROH
f = f"{OUT}/rohira_perlipatan_replikasi.csv"
if os.path.exists(f):
    d = pd.read_csv(f); assert not d.duplicated(["cell", "scenario", "rep"]).any() and d[["auc", "fold_mean"]].notna().all().all()
    p = pd.read_csv(f"{HERE}/hasil36/paired_rohira_null.csv").rename(columns={"variant": "cell"}); m = d.merge(p, on=["cell", "scenario", "rep"], suffixes=("_40", "_36")); mx = (m["auc_40"] - m["auc_36"]).abs().max() if len(m) else np.nan
    q = pd.read_csv(f"{HERE}/hasil36/diagnosis_null_replikasi.csv"); m2 = d.merge(q, on=["cell", "scenario", "rep"], suffixes=("_40", "_36")); mx2 = (m2["auc_40"] - m2["auc_36"]).abs().max() if len(m2) else np.nan
    print(f"ROH reproduksi: RF {len(m)} pasang, dif max {mx:.2e}; LR {len(m2)} pasang, dif max {mx2:.2e}")
    assert (len(m) == 0 or mx < 1e-9) and (len(m2) == 0 or mx2 < 1e-9); ok("ROH: pooled AUC reproduces hasil36 (1e-9) on the available pairs")
print(f"\nALL {n_ok} CHECK GROUPS PASSED", flush=True)
