#!/usr/bin/env python3
"""Round 42: shared plasmode module (synthetic labels from covariates, H1/H2a feature exchange, transforms, adjustment). See PRAREGISTRASI_putaran42.md. Not run directly."""
import os, sys, zlib
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, "/home/miftah_f/RTX-5090/tuep_fresh/code")
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import round28_analisis as A

Y, N, D, ERA, SUBJ = A.Y, A.N, A.D_I2, A.ERA, A.c17.SUBJ
GAMMAS = [0.0, 0.5, 1.0]


def kseed(*parts):
    return zlib.crc32("|".join(map(str, parts)).encode()) & 0x7fffffff


# eta: linear predictor of an unpenalized logistic regression of the real label on the covariates; only used to form the synthetic label
_lr = LogisticRegression(penalty=None, max_iter=2000).fit(D[:, 1:], Y); ETA = _lr.decision_function(D[:, 1:]); ETA = ETA - ETA.mean(); AUC_ETA_REAL = float(roc_auc_score(Y, ETA))


def gen_labels(rep, gamma):
    """y_sim = 1 for the 100 patients with the highest score gamma*eta + e (e ~ logistic(0,1), the same for all gamma within a replicate)."""
    e = np.random.default_rng(kseed("plasmode_e", rep)).logistic(size=N); score = gamma * ETA + e; y = np.zeros(N, int); y[np.argsort(-score, kind="stable")[: N // 2]] = 1; return y


def gen_perm(rep, cond):
    if cond == "nyata":
        return np.arange(N)
    rng = np.random.default_rng(kseed("plasmode_perm", rep, cond))
    if cond == "H1":
        return rng.permutation(N)
    pi = np.arange(N)                                                      # H2a: within era
    for e in np.unique(ERA):
        ii = np.where(ERA == e)[0]; pi[ii] = rng.permutation(ii)
    return pi


def folds_for(rep, gamma, y):
    return list(StratifiedKFold(5, shuffle=True, random_state=kseed("plasmode_cv", rep, gamma)).split(np.zeros(N), y))


def split_val(tr, y, seed):
    """20% stratified validation from the training patients (patient indices); the remainder (80%) is used to train RF/XGB."""
    g = np.random.default_rng(seed); va = []
    for c in (0, 1):
        ps = np.array(sorted(p for p in tr if y[p] == c)); va += list(g.permutation(ps)[: int(round(0.2 * len(ps)))])
    return np.array(sorted(set(tr) - set(va))), np.array(sorted(va))


# ------------------------------------------------------------------ transforms (fitted on training rows) and adjustment
Q = np.linspace(0, 1, 1001)


def make_transform(kind, Xfit):
    """Returns a function f(X) fitted ONLY on Xfit."""
    if kind == "raw":
        return lambda X: X
    if kind == "symlog":
        return lambda X: np.sign(X) * np.log1p(np.abs(X))
    if kind == "log":
        return lambda X: np.log(np.maximum(X, 1e-12))
    if kind == "rank":
        from scipy.stats import norm
        Qs = np.quantile(Xfit, Q, axis=0)

        def f(X):
            out = np.empty(X.shape, float)
            for j in range(X.shape[1]):
                out[:, j] = norm.ppf(np.clip(np.interp(X[:, j], Qs[:, j], Q), 1e-3, 1 - 1e-3))
            return out
        return f
    raise ValueError(kind)


def patient_means(Z, owner):
    S = np.zeros((N, Z.shape[1])); np.add.at(S, owner, Z); cnt = np.bincount(owner, minlength=N); return S / np.maximum(cnt, 1)[:, None]


def adjust_units(Z, owner, tr, kind):
    """Z: transformed unit (window/segment/patient); owner: owning patient; tr: training patient indices. none / win (OLS over all training units) / pat (OLS over the per-patient mean, unweighted)."""
    if kind == "none":
        return Z
    if kind == "win":
        rows = np.isin(owner, tr); B = np.linalg.lstsq(D[owner][rows], Z[rows], rcond=None)[0]
    elif kind == "pat":
        B = np.linalg.lstsq(D[tr], patient_means(Z, owner)[tr], rcond=None)[0]
    else:
        raise ValueError(kind)
    return Z - D[owner] @ B


def fold_aucs(y, P, folds):
    return [float(roc_auc_score(y[te], P[te])) for _, te in folds]
