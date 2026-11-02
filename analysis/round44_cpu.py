#!/usr/bin/env python3
"""Round 44 (CPU): F2-Clean (variant without age/sex, rank-normal, age^2 term) for I2 and P3, IPW comparator (I2, RF), 30-window sensitivity, IPW-RF plasmode leakage test. See PRAREGISTRASI_putaran44.md.
Usage: CUDA_VISIBLE_DEVICES="" python round44_cpu.py --check | --run A | --run B | --run C | --run D   (A: I2/P3 50 folds; B: real IPW; C: 30 windows; D: IPW-RF plasmode)"""
import os, sys, time, zlib
os.environ["CUDA_VISIBLE_DEVICES"] = ""; os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ["PYTHONWARNINGS"] = "ignore"
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, "/home/miftah_f/RTX-5090/tuep_fresh/code")
import numpy as np, pandas as pd
from scipy.stats import norm
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import round28_analisis as A
import round30_p3_v5 as V5
import evalkit as ek
import round33_pembanding_v4 as R3
import round42_common as C42

OUT = os.path.join(HERE, "hasil44"); os.makedirs(OUT, exist_ok=True); NJOBS = int(os.environ.get("NJOBS", 8))
Y, N, J, SUBJ, ERA, D1, AS = A.Y, A.N, A.J, A.c17.SUBJ, A.ERA, A.D_I2, A.c17.AS.astype(float); AGE = A.AGE.astype(float)
D2 = np.column_stack([D1, ((AGE - AGE.mean()) / AGE.std()) ** 2]); DES = {"D1": D1, "D2": D2}; Q = np.linspace(0, 1, 1001)
VARS = {"asis": (False, "D1", True), "noAS": (False, "D1", False), "noAS-D2": (False, "D2", False), "rank-noAS-D1": (True, "D1", False), "rank-noAS-D2": (True, "D2", False)}


def kseed(*p):
    return zlib.crc32("|".join(map(str, p)).encode()) & 0x7fffffff


def rank_fit_apply(F, fit):
    Qs = np.quantile(F[fit], Q, axis=0); out = np.empty(F.shape)
    for j in range(F.shape[1]):
        out[:, j] = norm.ppf(np.clip(np.interp(F[:, j], Qs[:, j], Q), 1e-3, 1 - 1e-3))
    return out


def post_features(F, tr, rank, dname, use_as):
    """[rank-normal (fitted on training patients)] -> V4 (OLS on the design, fitted on training patients; residual / training SD) -> [+ age, sex]."""
    F = rank_fit_apply(F, tr) if rank else np.asarray(F, float); Dm = DES[dname]; B = np.linalg.lstsq(Dm[tr], F[tr], rcond=None)[0]; R = F - Dm @ B
    sd = R[tr].std(0, ddof=np.linalg.matrix_rank(Dm[tr])); R = R / np.where(sd < 1e-8, 1.0, sd)
    return np.hstack([R, AS]) if use_as else R


def fit_pred(F, tr, te, var):
    G = post_features(F, tr, *VARS[var]); return ek.nested_lr().fit(G[tr], Y[tr]).predict_proba(G[te])[:, 1]


def i2_fold(X, tr, te, variants):
    return {v: fit_pred(X, tr, te, v) for v in variants}


def p3_fold(Cw, pid, tr, te, variants):
    X = V5.features_v5(Cw, pid, tr); Zp = PCA(V5.p3.NPC, random_state=0).fit(X[tr]).transform(X); return {v: fit_pred(Zp, tr, te, v) for v in variants}


def ipw_weights(y, tr):
    """Inverse-propensity weights from the training patients: P(y=1 | D1[:,1:]) unpenalized logistic regression; w = 1/P(y_i | C_i); clipped at the 5th and 95th percentile of the training weights; mean 1 on the training set. Returns weights for ALL indices (only the training ones are used)."""
    m = LogisticRegression(penalty=None, max_iter=2000).fit(D1[tr][:, 1:], y[tr]); p1 = m.predict_proba(D1[:, 1:])[:, 1]; w = np.where(y == 1, 1 / p1, 1 / (1 - p1)); lo, hi = np.percentile(w[tr], [5, 95]); w = np.clip(w, lo, hi); return w / w[tr].mean()


def ipw_i2(X, tr, te, use_as):
    w = ipw_weights(Y, tr); G = np.hstack([X, AS]) if use_as else X
    return ek.nested_lr().fit(G[tr], Y[tr], logisticregressioncv__sample_weight=w[tr]).predict_proba(G[te])[:, 1]


def ipw_rf(Xw, owner, y, tr, te, seed):
    """Rohira RF (80% of training patients), per-patient IPW weights passed to the windows. Xw: raw windows; owner: patient index of each window."""
    w = ipw_weights(y, tr); tr_p, _ = C42.split_val(tr, y, seed); fit = np.isin(owner, tr_p); sel = np.isin(owner, te)
    m = RandomForestClassifier(n_estimators=100, max_depth=1000, class_weight="balanced", n_jobs=1, random_state=seed).fit(Xw[fit], y[owner[fit]], sample_weight=w[owner[fit]])
    return pd.Series(m.predict_proba(Xw[sel])[:, 1]).groupby(owner[sel]).mean().reindex(te).values


def run_A():
    Cw, pid = V5.p3.load_cache(NJOBS); t0 = time.time(); res = {}; vs = list(VARS)
    o = Parallel(n_jobs=NJOBS)(delayed(i2_fold)(A.BP.astype(float), J[f][2], J[f][3], vs) for f in range(len(J)))
    for v in vs:
        res[f"I2__{v}"] = A.c17.to_matrix(J, [x[v] for x in o])
    o = Parallel(n_jobs=NJOBS, max_nbytes="1M")(delayed(p3_fold)(Cw, pid, J[f][2], J[f][3], vs) for f in range(len(J)))
    for v in vs:
        res[f"P3__{v}"] = A.c17.to_matrix(J, [x[v] for x in o])
    np.savez(f"{OUT}/A_prediksi.npz", **res)
    for k, M in res.items():
        print(f"{k}: per-fold AUC {100 * A.c17.fold_aucs(J, M).mean():.2f}; pooled {100 * np.mean([roc_auc_score(Y, p) for p in M]):.2f}", flush=True)
    print(f"done A [{time.time() - t0:.0f}s]", flush=True)


def run_B():
    t0 = time.time(); res = {}; X = A.BP.astype(float)
    for tag, uas in (("IPW-I2", False), ("IPW-I2+AS", True)):
        o = Parallel(n_jobs=NJOBS)(delayed(ipw_i2)(X, J[f][2], J[f][3], uas) for f in range(len(J))); res[tag] = A.c17.to_matrix(J, o)
    XR, PIDX = R3.XR, R3.PIDX
    o = Parallel(n_jobs=NJOBS)(delayed(ipw_rf)(XR, PIDX, Y, J[f][2], J[f][3], kseed("e44rf", f)) for f in range(len(J))); res["IPW-RF"] = A.c17.to_matrix(J, o)
    np.savez(f"{OUT}/B_prediksi.npz", **res)
    for k, M in res.items():
        print(f"{k}: per-fold AUC {100 * A.c17.fold_aucs(J, M).mean():.2f}; pooled {100 * np.mean([roc_auc_score(Y, p) for p in M]):.2f}", flush=True)
    print(f"done B [{time.time() - t0:.0f}s]", flush=True)


def subset_positions():
    """Relative positions (within patient) of min(30, n) random windows without replacement; seed 44000 + patient index."""
    import eegbench_fixed_eval as ebf
    return lambda n, p: np.sort(np.random.default_rng(44000 + p).choice(n, size=min(30, n), replace=False))


def run_C():
    Cw, pid = V5.p3.load_cache(NJOBS); t0 = time.time(); sub = subset_positions()
    w = np.load("/home/miftah_f/RTX-5090/tuep_fresh/data/features_bandpower_windows.npz", allow_pickle=True); Xw = w["X"]; e = np.load("/home/miftah_f/RTX-5090/tuep_fresh/data/emb_labram_windows.npz", allow_pickle=True)
    import eegbench_fixed_eval as ebf
    first = ebf.first[e["rec"]]; subj = np.asarray(ebf.subj_r)[e["rec"]]; idxw = [np.where(first & (subj == p))[0] for p in SUBJ]; cnt_p3 = np.bincount(pid, minlength=N)
    assert all(len(idxw[p]) == cnt_p3[p] for p in range(N)), "number of P3 windows differs from the BP windows"
    starts = np.r_[0, np.cumsum(cnt_p3)[:-1]]; keep_p3, bp30 = [], []
    for p in range(N):
        pos = sub(cnt_p3[p], p); keep_p3.append(starts[p] + pos); bp30.append(np.median(Xw[idxw[p][pos]], 0))
    keep_p3 = np.concatenate(keep_p3); C30, pid30 = Cw[keep_p3], pid[keep_p3]; BP30 = np.vstack(bp30); v = "rank-noAS-D2"; res = {}
    o = Parallel(n_jobs=NJOBS)(delayed(i2_fold)(BP30, J[f][2], J[f][3], [v]) for f in range(len(J))); res["I2-30"] = A.c17.to_matrix(J, [x[v] for x in o])
    o = Parallel(n_jobs=NJOBS, max_nbytes="1M")(delayed(p3_fold)(C30, pid30, J[f][2], J[f][3], [v]) for f in range(len(J))); res["P3-30"] = A.c17.to_matrix(J, [x[v] for x in o])
    np.savez(f"{OUT}/C_prediksi.npz", **res)
    for k, M in res.items():
        print(f"{k}: per-fold AUC {100 * A.c17.fold_aucs(J, M).mean():.2f}; pooled {100 * np.mean([roc_auc_score(Y, p) for p in M]):.2f}", flush=True)
    print(f"done C [{time.time() - t0:.0f}s]; total windows {len(keep_p3)}", flush=True)


def unit_ipw_rf(rep, cond, gamma):
    import round42_cpu as P42
    y = C42.gen_labels(rep, gamma); perm = C42.gen_perm(rep, cond); folds = C42.folds_for(rep, gamma, y); X, owner = P42.windows_for(perm); P = np.zeros(N)
    for fi, (tr, te) in enumerate(folds):
        P[te] = ipw_rf(X, owner, y, tr, te, C42.kseed("plasmode_clf", rep, gamma, fi))
    fa = C42.fold_aucs(y, P, folds); return dict(fam="IPW-RF", rep=rep, cond=cond, gamma=gamma, auc=float(roc_auc_score(y, P)), **{f"fold_auc_{i}": v for i, v in enumerate(fa)})


def run_D(nrep=30):
    path = f"{OUT}/D_ipw_rf_plasmode.csv"; done = set(map(tuple, pd.read_csv(path, usecols=["rep", "cond", "gamma"]).drop_duplicates().values.tolist())) if os.path.exists(path) else set(); t0 = time.time()
    units = [(r, c, g) for r in range(nrep) for c in ("H1", "H2a") for g in C42.GAMMAS if (r, c, g) not in done]
    for i in range(0, len(units), NJOBS * 2):
        res = Parallel(n_jobs=NJOBS)(delayed(unit_ipw_rf)(r, c, g) for r, c, g in units[i:i + NJOBS * 2])
        pd.DataFrame(res).to_csv(path, mode="a", header=not os.path.exists(path), index=False); print(f"{min(i + NJOBS * 2, len(units))}/{len(units)} [{time.time() - t0:.0f}s]", flush=True)


def checks():
    n = 0; r, k, tr, te = J[0]; X = A.BP.astype(float)
    for v in VARS:
        X2 = X.copy(); X2[te] = X2[te] * 5 + 3; assert np.allclose(post_features(X, tr, *VARS[v])[tr], post_features(X2, tr, *VARS[v])[tr]); n += 1
    R = post_features(X, tr, True, "D2", False); assert np.abs(D2[tr].T @ R[tr]).max() / len(tr) < 1e-8; n += 1
    R = post_features(X, tr, False, "D1", False); assert np.abs(D1[tr].T @ R[tr]).max() / len(tr) < 1e-8; n += 1
    y0 = Y.copy(); Y[te] = 1 - Y[te]; a = fit_pred(X, tr, te, "rank-noAS-D2") if False else None; Y[:] = y0
    a = fit_pred(X, tr, te, "rank-noAS-D2"); Y[te] = 1 - Y[te]; b = fit_pred(X, tr, te, "rank-noAS-D2"); Y[:] = y0; assert np.allclose(a, b); n += 1                                   # test labels are not used (I2)
    w = ipw_weights(Y, tr); assert abs(w[tr].mean() - 1) < 1e-12 and w[tr].min() > 0; w2 = ipw_weights(np.where(np.isin(np.arange(N), te), 1 - Y, Y), tr); assert np.allclose(w[tr], w2[tr]); n += 1     # training weights do not depend on test labels
    a = ipw_i2(X, tr, te, False); y0 = Y.copy(); Y[te] = 1 - Y[te]; b = ipw_i2(X, tr, te, False); Y[:] = y0; assert np.allclose(a, b); n += 1
    print(f"{n} checks passed; design df D1 {D1.shape}, D2 {D2.shape}", flush=True)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "--check"
    if mode == "--check":
        checks()
    elif mode == "--run":
        {"A": run_A, "B": run_B, "C": run_C, "D": run_D}[sys.argv[2]]()
