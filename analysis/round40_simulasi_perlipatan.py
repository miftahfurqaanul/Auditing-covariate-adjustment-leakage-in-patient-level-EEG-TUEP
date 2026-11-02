#!/usr/bin/env python3
"""Round 40 (stages P/A/C/GB/GC): repeats the round 39 simulation (identical seeds) with per-fold AUC output and H40-3 diagnostics. See PRAREGISTRASI_putaran40.md + addendum1.
Usage: python round40_simulasi_perlipatan.py --check | --run <P|A|C|GB|GC> <cpu|gpu> [n_rep]   (resumable; the cpu chain disables the GPU automatically)"""
import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import round39_simulasi_kebocoran as R                      # mengatur CUDA_VISIBLE_DEVICES menurut argumen --run <tahap> <rantai>
from scipy import stats
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

OUT = os.path.join(HERE, "hasil40"); os.makedirs(os.path.join(OUT, "pred"), exist_ok=True)
CPU40 = ["LRpat", "RF"]; GPU40 = R.GPU_CLF
BASE = dict(n=200, m=50, p=50, imb=False, s=1.0, sig=1.0, cfrac=0.01, cscale=100.0, scen="H1")


def diagnostics(X, pid, y, D, folds):
    """skew, Dq05, cusp (definitions in PRAREGISTRASI_putaran40.md); per fold only from the training windows and patients, averaged over folds. Does not use test data."""

    sk, dq, cu = [], [], []
    for tr, te in folds:
        trw = np.isin(pid, tr); Xt = X[trw]; pw = pid[trw]; sk.append(np.median(np.abs(stats.skew(Xt, axis=0))))
        B = np.linalg.lstsq(D[pw], Xt, rcond=None)[0]; Rr = Xt - D[pw] @ B
        eta = LogisticRegression(max_iter=500).fit(D[tr][:, 1:], y[tr]).decision_function(D[:, 1:]); qs = np.quantile(eta[tr], [1 / 3, 2 / 3]); lo = eta[pw] <= qs[0]; hi = eta[pw] >= qs[1]
        iqr = np.subtract(*np.percentile(Rr, [75, 25], axis=0)) + 1e-12; dq.append(np.median(np.abs(np.quantile(Rr[hi], .05, axis=0) - np.quantile(Rr[lo], .05, axis=0)) / iqr))
        med = np.median(Xt, axis=0); sc = np.subtract(*np.percentile(Xt, [75, 25], axis=0)) / 1.349 + 1e-12; cu.append(np.mean(np.mean(np.abs((Xt - med) / sc) < 0.25, axis=0)) / 0.1974)
    return float(np.mean(sk)), float(np.mean(dq)), float(np.mean(cu))


def run_rep40(cell, rep, clfs, configs, dev=None, want_pred=False):
    """Copy of run_rep of round 39 (identical data, folds, seeds) + per-fold AUC + diagnostics. 'auc' = pooled AUC, must be identical to round 39."""
    cell = dict(dict(sig=1.0, cfrac=0.01, cscale=100.0), **cell); key = (cell["dist"], cell["scen"], cell["n"], cell["m"], cell["p"], cell["imb"], cell["s"])
    if (cell["sig"], cell["cfrac"], cell["cscale"]) != (1.0, 0.01, 100.0):
        key = key + (cell["sig"], cell["cfrac"], cell["cscale"])
    seed = R.kseed(*key, rep)
    data = R.make_data(seed, **{k: cell[k] for k in ("n", "m", "p", "dist", "scen", "s", "imb", "sig", "cfrac", "cscale")}); X, pid, y, D = data["X"], data["pid"], data["y"], data["D"]; n = data["n"]
    kurt = float(np.median(stats.kurtosis(X, axis=0))); folds = list(StratifiedKFold(5, shuffle=True, random_state=seed).split(np.zeros(n), y)); res = []; allP = {}
    skew, dq05, cusp = diagnostics(X, pid, y, D, folds); extra = dict(skew=skew, Dq05=dq05, cusp=cusp)
    meta = np.zeros(n)
    for tr, te in folds:
        meta[te] = LogisticRegression(max_iter=500).fit(D[tr][:, 1:], y[tr]).predict_proba(D[te][:, 1:])[:, 1]
    meta_auc = roc_auc_score(y, meta)
    fa = lambda p: {f"fold_auc_{i}": float(roc_auc_score(y[te], p[te])) for i, (tr, te) in enumerate(folds)}
    for clf in clfs:
        want = [c for c in configs if not (clf in R.TREE and c[1] == "none" and c[0] != "raw")]; P = {c: np.zeros(n) for c in want}
        for fi, (tr, te) in enumerate(folds):
            trw = np.isin(pid, tr); cache = {}
            for t, a in want:
                if t not in cache:
                    cache[t] = R.transform(X, trw, t)
                Z = R.adjust(cache[t], pid, D, tr, trw, a); fs = R.kseed(seed, clf, fi)
                P[(t, a)][te] = R.clf_cpu(clf, Z, pid, y, tr, te, fs) if clf in R.CPU_CLF else R.clf_gpu(clf, Z, pid, y, tr, te, fs, dev)
        for c in want:
            res.append(dict(rep=rep, clf=clf, transform=c[0], adjust=c[1], auc=float(roc_auc_score(y, P[c])), **fa(P[c]), kurtosis=kurt, meta_auc=float(meta_auc), **extra, **cell))
            allP[f"{clf}|{c[0]}|{c[1]}"] = P[c]
        if clf in R.TREE:
            for c in configs:
                if c not in want:
                    res.append(dict(rep=rep, clf=clf, transform=c[0], adjust="none", auc=float(roc_auc_score(y, P[("raw", "none")])), **fa(P[("raw", "none")]), kurtosis=kurt, meta_auc=float(meta_auc), **extra, **cell))
    return (res, allP, seed) if want_pred else res


def stage_def(stage):
    if stage == "P":
        return [dict(BASE, dist="lognormal"), dict(BASE, dist="contam")], CPU40
    if stage == "A":
        return R.cells_b(), CPU40
    if stage == "C":
        return R.stage_cells("C"), CPU40
    if stage == "GB":
        return [dict(BASE, dist=d_, s=s_) for d_ in ("lognormal", "contam") for s_ in (1.0, 0.0)], GPU40
    if stage == "GC":
        return R.stage_cells("C"), GPU40
    raise ValueError(stage)


def checks():
    n = 0
    cell = dict(BASE, dist="contam", n=60, m=10, p=6); a = run_rep40(cell, 0, ["LRpat"], [("raw", "none"), ("raw", "pat"), ("rank", "pat")]); b = R.run_rep(cell, 0, ["LRpat"], [("raw", "none"), ("raw", "pat"), ("rank", "pat")])
    A_ = pd.DataFrame(a); B_ = pd.DataFrame(b); assert np.array_equal(A_["auc"].values, B_["auc"].values) and np.array_equal(A_["kurtosis"].values, B_["kurtosis"].values); n += 1                    # identical to run_rep of round 39
    fcols = [f"fold_auc_{i}" for i in range(5)]; assert A_[fcols].notna().all().all() and ((A_[fcols] >= 0) & (A_[fcols] <= 1)).all().all(); n += 1
    d = R.make_data(9, n=60, m=10, p=6, dist="lognormal"); folds = list(StratifiedKFold(5, shuffle=True, random_state=1).split(np.zeros(60), d["y"])); s1 = diagnostics(d["X"], d["pid"], d["y"], d["D"], folds)
    X2 = d["X"].copy(); te_all = np.unique(np.concatenate([te for _, te in folds])); _ = te_all
    ytest = d["y"].copy(); tr0, te0 = folds[0]; ytest[te0] = 1 - ytest[te0]; X3 = d["X"].copy(); X3[np.isin(d["pid"], te0)] *= 7
    s2 = diagnostics(X3, d["pid"], ytest, d["D"], folds[:1]); s3 = diagnostics(d["X"], d["pid"], d["y"], d["D"], folds[:1]); assert np.allclose(s2, s3); n += 1                                   # the diagnostics do not use test data/labels
    dn = R.make_data(5, n=100, m=40, p=20, dist="normal"); fn = list(StratifiedKFold(5, shuffle=True, random_state=1).split(np.zeros(100), dn["y"])); sk, dq, cu = diagnostics(dn["X"], dn["pid"], dn["y"], dn["D"], fn)
    assert abs(cu - 1) < 0.15 and sk < 0.3; n += 1                                                                                                                                              # normal: cusp ~ 1, skew ~ 0
    print(f"{n} round 40 checks passed (normal: skew {sk:.2f} Dq05 {dq:.2f} cusp {cu:.2f})", flush=True)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "--check"; t0 = time.time()
    if mode == "--check":
        R.checks(); checks()
    elif mode == "--run":
        stage, chain = sys.argv[2], sys.argv[3]; nrep = int(sys.argv[4]) if len(sys.argv) > 4 else 20; cells, clfs = stage_def(stage); path = f"{OUT}/{stage}_{chain}.csv"; done = R.done_set(path)
        if chain == "gpu":
            import torch
            dev = torch.device("cuda")
        for ci, cell in enumerate(cells):
            cell = dict(dict(sig=1.0, cfrac=0.01, cscale=100.0), **cell); configs = R.configs_b(cell["dist"], cell["imb"], chain)
            heavy = stage == "A" and (cell["m"] == 200 or cell["p"] == 200 or cell["n"] == 500); nrc = max(nrep // 2, 30) if heavy else nrep
            todo = [(r, c) for r in range(nrc) for c in clfs if (cell["dist"], cell["scen"], cell["n"], cell["m"], cell["p"], cell["imb"], cell["s"], cell["sig"], cell["cfrac"], cell["cscale"], r, c) not in done]
            if not todo:
                continue
            base_cell = (stage == "A" and cell["dist"] in ("lognormal", "contam") and cell["scen"] == "H1" and cell["n"] == 200 and cell["m"] == 50 and cell["p"] == 50 and not cell["imb"] and cell["s"] == 1.0 and cell["sig"] == 1.0 and cell["cfrac"] == 0.01 and cell["cscale"] == 100.0)
            if chain == "cpu":
                rows = Parallel(n_jobs=int(os.environ.get("NJOBS", 12)))(delayed(run_rep40)(cell, r, [c], configs, None, base_cell and r < 3) for r, c in todo)
                for (r, c), rr in zip(todo, rows):
                    if base_cell and r < 3:
                        rr, Pd, sd = rr; np.savez(f"{OUT}/pred/{cell['dist']}_rep{r}_{c}.npz", seed=sd, **{k.replace("|", "__"): v for k, v in Pd.items()})
                    R.write_rows(path, rr)
            else:
                for r, c in todo:
                    R.write_rows(path, run_rep40(cell, r, [c], configs, dev))
            print(f"cell {ci + 1}/{len(cells)} {cell['dist']} {cell['scen']} s={cell['s']}: done [{time.time() - t0:.0f}s]", flush=True)
    print(f"done {time.time() - t0:.0f}s")
