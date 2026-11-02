#!/usr/bin/env python3
"""Round 37 (GPU): Tang-F with per-second features passed through symlog (sign(x) log1p(|x|); data-independent constant) and covariate adjustment. See PRAREGISTRASI_putaran37.md.
Variants: L = symlog without adjustment; Lp = symlog + adjustment with coefficients from the per-patient mean (unweighted; all training patients); Lw = symlog + second-weighted coefficients (exact).
One GPU process at a time. Usage: python round37_tang_log.py --check | --null <variant> [n=20] | --run <variant>   (resumable)"""
import os, sys, json, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd, torch
from scipy import stats
from sklearn.metrics import roc_auc_score
import round34_tang_v4 as T
tf, r4 = T.tf, T.r4
OUT = os.path.join(HERE, "hasil37"); [os.makedirs(f"{OUT}/{v}", exist_ok=True) for v in ("L", "Lp", "Lw")]


def symlog(X):
    return (np.sign(X) * np.log1p(np.abs(X))).astype(np.float32)


def load_data():
    X, y, subj, ylab, owner, S1, cnt = T.load_data(); X = symlog(X)
    S1 = np.zeros((len(T.SUBJ), 506)); c = np.zeros(len(T.SUBJ)); np.add.at(S1, owner, X.sum(1, dtype=np.float64)); np.add.at(c, owner, X.shape[1]); return X, y, subj, ylab, owner, S1, c


def offsets_v(var, S1, cnt, owner, tr_p, va_p, dev):
    """L: no offset; Lp: OLS on the per-patient mean (unweighted) over pool = tr_p + va_p; Lw: OLS weighted by the number of seconds (exact)."""
    if var == "L":
        return None
    pool = np.array([T.POS[s] for s in tr_p + va_p])
    if var == "Lw":
        B = T.coef(S1, cnt, pool)
    else:
        M = S1 / np.maximum(cnt, 1)[:, None]; B = np.linalg.lstsq(T.D[pool], M[pool], rcond=None)[0]
    return torch.from_numpy((T.D[owner] @ B).astype(np.float32)).to(dev)


def checks(X, y, subj, ylab, owner, S1, cnt):
    k = stats.kurtosis(X[::4].reshape(-1, 506).astype(np.float64), axis=0); print(f"symlog: median kurtosis {np.median(k):.2f}, max {k.max():.2f}; min value {X.min():.2f}, max {X.max():.2f}", flush=True)
    F = tf.folds_199(); te_p, tr_p, va_p = T.fold_patients(F, 0, 0, ylab); pool = np.array([T.POS[s] for s in tr_p + va_p]); M = S1 / np.maximum(cnt, 1)[:, None]
    B = np.linalg.lstsq(T.D[pool], M[pool], rcond=None)[0]; assert np.abs(T.D[pool].T @ (M[pool] - T.D[pool] @ B)).max() / len(pool) < 1e-6
    S1b = S1.copy(); other = np.array([T.POS[s] for s in te_p]); S1b[other] = S1b[other] * 5 + 3; Mb = S1b / np.maximum(cnt, 1)[:, None]; assert np.array_equal(np.linalg.lstsq(T.D[pool], Mb[pool], rcond=None)[0], B)
    print("2 checks passed (patient-level residuals orthogonal to the design; test data do not change B)", flush=True)


def null(var, X, y, subj, ylab, owner, S1, cnt, n):
    dev = torch.device("cuda"); F = tf.folds_199(); pats = [s for s in T.SUBJ if s in ylab.index]; pi = np.array([T.POS[s] for s in pats]); seqs = {p: np.where(owner == p)[0] for p in pi}
    path = f"{OUT}/tang_{var}_null_replikasi.csv"; done = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame(columns=["rep", "scenario", "variant", "auc"]); t0 = time.time()
    for rep in range(n):
        for scen in ("H1", "H2a"):
            if ((done.rep == rep) & (done.scenario == scen)).any():
                continue
            rng = np.random.default_rng((34000 if scen == "H1" else 34300) + rep); perm = np.arange(len(pi))
            if scen == "H1":
                perm = rng.permutation(len(pi))
            else:
                for e in np.unique(T.ERA[pi]):
                    ii = np.where(T.ERA[pi] == e)[0]; perm[ii] = rng.permutation(ii)
            idx = np.concatenate([seqs[pi[perm[j]]] for j in range(len(pi))]); own_n = np.concatenate([np.full(len(seqs[pi[perm[j]]]), pi[j]) for j in range(len(pi))])
            subj_n = T.SUBJ[own_n]; y_n = r4.Y[own_n]; Xg = torch.from_numpy(X[idx]).to(dev); S1n = np.zeros_like(S1); cn = np.zeros_like(cnt); S1n[pi] = S1[pi[perm]]; cn[pi] = cnt[pi[perm]]; sc = {}
            for k in range(5):
                te_p, tr_p, va_p = T.fold_patients(F, 0, k, ylab); sc.update(T.train_fold(Xg, y_n, subj_n, tr_p, va_p, te_p, ylab, 2000 + k, dev, Off=offsets_v(var, S1n, cn, own_n, tr_p, va_p, dev))["test"])
            auc = roc_auc_score(ylab.loc[list(sc)].values, list(sc.values())); done = pd.concat([done, pd.DataFrame([dict(rep=rep, scenario=scen, variant=var, auc=auc)])]); done.to_csv(path, index=False)
            print(f"null rep {rep} {scen} {var}: AUC {auc:.3f} [{time.time() - t0:.0f}s]", flush=True); del Xg; torch.cuda.empty_cache()
    print(done.groupby("scenario").auc.agg(["count", "mean", "std"]).round(4).to_string(), flush=True)


def run(var, X, y, subj, ylab, owner, S1, cnt):
    dev = torch.device("cuda"); Xg = torch.from_numpy(X).to(dev); F = tf.folds_199(); t0 = time.time()
    for r in range(10):
        for k in range(5):
            path = f"{OUT}/{var}/r{r}_f{k}.json"
            if os.path.exists(path):
                continue
            te_p, tr_p, va_p = T.fold_patients(F, r, k, ylab); seed = 2000 + 10 * r + k; t1 = time.time()
            res = T.train_fold(Xg, y, subj, tr_p, va_p, te_p, ylab, seed, dev, Off=offsets_v(var, S1, cnt, owner, tr_p, va_p, dev)); pt = pd.Series(res["test"]); auc = roc_auc_score(ylab.loc[pt.index].values, pt.values)
            print(f"{var} r{r} f{k}: {time.time() - t1:.0f}s, epoch {res['best_epoch']}, val AUC {res['val_auc']:.3f}, test AUC {auc:.3f}  [total {time.time() - t0:.0f}s]", flush=True)
            json.dump(dict(res, train_patients=len(tr_p), val_patients=len(va_p), variant=var, seed=seed), open(path, "w"))
    print("RUN_OK", flush=True)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "--check"; X, y, subj, ylab, owner, S1, cnt = load_data(); print(f"{len(X)} sekuens; cuda {torch.cuda.is_available()}", flush=True); checks(X, y, subj, ylab, owner, S1, cnt)
    if mode == "--null":
        null(sys.argv[2], X, y, subj, ylab, owner, S1, cnt, int(sys.argv[3]) if len(sys.argv) > 3 else 20)
    elif mode == "--run":
        run(sys.argv[2], X, y, subj, ylab, owner, S1, cnt)
