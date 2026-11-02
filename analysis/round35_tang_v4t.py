#!/usr/bin/env python3
"""Round 35 (GPU): Tang-F with V4t (B fitted ONLY on the tr_p patients that train the network; validation and test patients out-of-sample). See PRAREGISTRASI_putaran35.md.
One GPU process at a time. Usage: python round35_tang_v4t.py --check | --null [n=20] | --run   (resumable; evaluation: round35_eval_tang.py, CPU)"""
import os, sys, json, time, hashlib
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd, torch
from sklearn.metrics import roc_auc_score
import round34_tang_v4 as T
tf, r4 = T.tf, T.r4
OUT = os.path.join(HERE, "hasil35"); os.makedirs(f"{OUT}/V4t", exist_ok=True)


def offsets_t(S1, cnt, owner, tr_p, dev):
    return T.offsets(S1, cnt, owner, tr_p, [], dev)                                    # pool = tr_p only


def checks(X, y, subj, ylab, owner, S1, cnt):
    F = tf.folds_199(); te_p, tr_p, va_p = T.fold_patients(F, 0, 0, ylab); pool = np.array([T.POS[s] for s in tr_p]); B = T.coef(S1, cnt, pool)
    sel = np.isin(owner, pool); R = X[sel].reshape(-1, 506).astype(np.float64) - np.repeat(T.D[owner[sel]] @ B, X.shape[1], axis=0); Dd = np.repeat(T.D[owner[sel]], X.shape[1], axis=0)
    assert np.abs(Dd.T @ R).max() / len(R) < 1e-5, "tr_p residual not orthogonal to the design"
    S1b = S1.copy(); other = np.array([T.POS[s] for s in va_p + te_p]); S1b[other] = S1b[other] * 5 + 3; assert np.array_equal(T.coef(S1b, cnt, pool), B), "validation/test data change B"
    print("2 checks passed (tr_p residuals orthogonal to the design; validation/test data do not change B)", flush=True)


def null(X, y, subj, ylab, owner, S1, cnt, n):
    dev = torch.device("cuda"); F = tf.folds_199(); pats = [s for s in T.SUBJ if s in ylab.index]; pi = np.array([T.POS[s] for s in pats]); seqs = {p: np.where(owner == p)[0] for p in pi}
    path = f"{OUT}/tang_v4t_null_replikasi.csv"; done = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame(columns=["rep", "scenario", "variant", "auc"]); t0 = time.time()
    for rep in range(n):
        for scen in ("H1", "H2a"):
            if ((done.rep == rep) & (done.scenario == scen)).any():
                continue
            rng = np.random.default_rng((34000 if scen == "H1" else 34300) + rep); perm = np.arange(len(pi))                 # same seed as round 34 (identical random data)
            if scen == "H1":
                perm = rng.permutation(len(pi))
            else:
                for e in np.unique(T.ERA[pi]):
                    ii = np.where(T.ERA[pi] == e)[0]; perm[ii] = rng.permutation(ii)
            idx = np.concatenate([seqs[pi[perm[j]]] for j in range(len(pi))]); own_n = np.concatenate([np.full(len(seqs[pi[perm[j]]]), pi[j]) for j in range(len(pi))])
            subj_n = T.SUBJ[own_n]; y_n = r4.Y[own_n]; Xg = torch.from_numpy(X[idx]).to(dev); S1n = np.zeros_like(S1); cn = np.zeros_like(cnt); S1n[pi] = S1[pi[perm]]; cn[pi] = cnt[pi[perm]]; sc = {}
            for k in range(5):
                te_p, tr_p, va_p = T.fold_patients(F, 0, k, ylab); sc.update(T.train_fold(Xg, y_n, subj_n, tr_p, va_p, te_p, ylab, 2000 + k, dev, Off=offsets_t(S1n, cn, own_n, tr_p, dev))["test"])
            auc = roc_auc_score(ylab.loc[list(sc)].values, list(sc.values())); done = pd.concat([done, pd.DataFrame([dict(rep=rep, scenario=scen, variant="V4t", auc=auc)])]); done.to_csv(path, index=False)
            print(f"null rep {rep} {scen} V4t: AUC {auc:.3f} [{time.time() - t0:.0f}s]", flush=True); del Xg; torch.cuda.empty_cache()
    print(done.groupby(["scenario", "variant"]).auc.agg(["count", "mean", "std"]).round(4).to_string(), flush=True)


def run(X, y, subj, ylab, owner, S1, cnt):
    dev = torch.device("cuda"); Xg = torch.from_numpy(X).to(dev); F = tf.folds_199(); t0 = time.time()
    for r in range(10):
        for k in range(5):
            path = f"{OUT}/V4t/r{r}_f{k}.json"
            if os.path.exists(path):
                continue
            te_p, tr_p, va_p = T.fold_patients(F, r, k, ylab); seed = 2000 + 10 * r + k; t1 = time.time()
            res = T.train_fold(Xg, y, subj, tr_p, va_p, te_p, ylab, seed, dev, Off=offsets_t(S1, cnt, owner, tr_p, dev)); pt = pd.Series(res["test"]); auc = roc_auc_score(ylab.loc[pt.index].values, pt.values)
            print(f"V4t r{r} f{k}: {time.time() - t1:.0f}s, epoch {res['best_epoch']}, val AUC {res['val_auc']:.3f}, test AUC {auc:.3f}  [total {time.time() - t0:.0f}s]", flush=True)
            json.dump(dict(res, train_patients=len(tr_p), val_patients=len(va_p), variant="V4t", seed=seed), open(path, "w"))


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "--check"; X, y, subj, ylab, owner, S1, cnt = T.load_data(); print(f"{len(X)} sekuens; cuda {torch.cuda.is_available()}", flush=True); checks(X, y, subj, ylab, owner, S1, cnt)
    if mode == "--null":
        null(X, y, subj, ylab, owner, S1, cnt, int(sys.argv[2]) if len(sys.argv) > 2 else 20)
    elif mode == "--run":
        run(X, y, subj, ylab, owner, S1, cnt); print("RUN_OK", flush=True)
