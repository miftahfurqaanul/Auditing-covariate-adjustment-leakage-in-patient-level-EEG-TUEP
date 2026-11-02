#!/usr/bin/env python3
"""Round 49A + 49C (CPU): strict nesting (all data-dependent transforms are refitted in every inner fold) for I2, P3, C1, the metadata baseline,
and non-EEG vs non-EEG + EEG models (N0, N1, N2). See PRAREGISTRASI_putaran49.md.
Usage: CUDA_VISIBLE_DEVICES="" python round49_nesting_ketat.py --check | --run   (raw results to hasil49/ before summarizing: round49_ringkas.py)"""
import os, sys, time, json, hashlib
os.environ["CUDA_VISIBLE_DEVICES"] = ""; os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ["PYTHONWARNINGS"] = "ignore"
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, "/home/miftah_f/RTX-5090/tuep_fresh/code")
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import round28_analisis as A
import round9 as r9
import evalkit as ek
import round30_p3_v5 as P5
J, Y, N, BP, D = A.J, A.Y, A.N, A.BP, A.D_I2
AS = A.c17.AS; ACQ = r9.ACQ; CS = ek.CS; p3 = P5.p3
OUT = os.path.join(HERE, "hasil49"); os.makedirs(OUT, exist_ok=True)


def v4(X, fit):
    """OLS of the features on [1, age, sex, era] fitted on the `fit` patients; residuals divided by the SD of the fitting residuals (as round29 V4)."""
    B = np.linalg.lstsq(D[fit], X[fit], rcond=None)[0]; sd = (X[fit] - D[fit] @ B).std(0, ddof=np.linalg.matrix_rank(D[fit])); return (X - D @ B) / sd


def make_tf(model, C=None, pid=None):
    """Returns a function fit_idx -> feature matrix (N x d) that uses statistics only from the fit_idx patients."""
    if model == "I2":
        return lambda fit: np.hstack([v4(BP, fit), AS])
    if model == "C1":
        return lambda fit: np.hstack([BP, AS])
    if model == "Floor":
        return lambda fit: ACQ
    if model == "N0":
        return lambda fit: np.hstack([ACQ, AS])
    if model == "N1":
        return lambda fit: np.hstack([ACQ, AS, v4(BP, fit)])
    if model == "N2":
        return lambda fit: np.hstack([ACQ, AS, BP])
    if model == "P3":
        def f(fit):
            X = P5.features_v5(C, pid, fit); Zp = PCA(p3.NPC, random_state=0).fit(X[fit]).transform(X)
            B = np.linalg.lstsq(D[fit], Zp[fit], rcond=None)[0]; return np.hstack([Zp - D @ B, AS])
        return f
    raise ValueError(model)


def lr(C):
    return LogisticRegression(C=C, class_weight="balanced", max_iter=5000)


def nested(tf, tr, te, strict):
    """Selection of C by inner CV (StratifiedKFold(5) without shuffling, as LogisticRegressionCV). strict=True: transforms and scaler are refitted in every inner fold."""
    if not strict:
        F = tf(tr); sc = StandardScaler().fit(F[tr]); G = sc.transform(F)
    score = np.zeros(len(CS))
    for a, b in StratifiedKFold(5).split(tr, Y[tr]):
        ia, ib = tr[a], tr[b]
        if strict:
            F = tf(ia); assert not set(ib) & set(ia); sc = StandardScaler().fit(F[ia]); G = sc.transform(F)
        est = LogisticRegression(class_weight="balanced", max_iter=5000, warm_start=True)                   # ascending C path with warm start, like LogisticRegressionCV (identical inner scores)
        for i, c in enumerate(CS):
            score[i] += roc_auc_score(Y[ib], est.set_params(C=c).fit(G[ia], Y[ia]).decision_function(G[ib]))
    c = CS[int(np.argmax(score))]
    F = tf(tr); sc = StandardScaler().fit(F[tr]); G = sc.transform(F)
    return lr(c).fit(G[tr], Y[tr]).predict_proba(G[te])[:, 1], float(c)


def job(model, f, strict, C=None, pid=None):
    r, k, tr, te = J[f]; assert not set(tr) & set(te)
    p, c = nested(make_tf(model, C, pid), tr, te, strict); return f, p, c


def to_matrix(res):
    P = np.full((10, N), np.nan); cs = np.zeros(len(J))
    for f, p, c in res:
        P[J[f][0], J[f][3]] = p; cs[f] = c
    return P, cs


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def check():
    Z = np.load(os.path.join(HERE, "hasil30", "eval_prediksi_v4.npz")); n = 0
    for model, key in (("I2", "I2_adj"), ("C1", "C1"), ("Floor", "Floor")):
        for f in range(5):
            _, p, _ = job(model, f, False); r, _, _, te = J[f]; d = np.abs(p - Z[key][r, te]).max(); print(f"loose {model} fold {f}: max difference {d:.2e}", flush=True); assert d < 5e-3, (model, f, d); n += 1
    C, pid = p3.load_cache(8); t0 = time.time(); _, p, _ = job("P3", 0, False, C, pid); r, _, _, te = J[0]; d = np.abs(p - Z["P3_adj"][r, te]).max()
    print(f"loose P3 fold 0: max difference {d:.2e} ({time.time() - t0:.0f}s)", flush=True); assert d < 5e-3; n += 1
    tr = J[0][2]; a = tr[:100]; F1 = make_tf("I2")(a); B = BP.copy(); B[np.setdiff1d(np.arange(N), a)] *= 7.0
    F2 = np.hstack([v4(B, a), AS]); assert np.allclose(F1[a], F2[a]); n += 1                          # data outside the fitting set do not change the features of the fitting patients
    print(f"{n} checks passed", flush=True)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "--check"; t0 = time.time()
    json.dump(dict(prereg_sha256=sha(os.path.join(HERE, "PRAREGISTRASI_putaran49.md")), code_sha256=sha(__file__)), open(f"{OUT}/meta_A.json", "w"), indent=1)
    if mode == "--check":
        check(); sys.exit()
    out = {}
    for model in ("I2", "C1", "Floor", "N0", "N1", "N2"):
        res = Parallel(n_jobs=6)(delayed(job)(model, f, True) for f in range(len(J))); P, cs = to_matrix(res); out[model] = P; out[f"{model}__C"] = cs
        np.savez(f"{OUT}/A_prediksi_ketat.npz", **out); print(f"{model} strict done [{time.time() - t0:.0f}s]", flush=True)
    C, pid = p3.load_cache(8)
    res = Parallel(n_jobs=4, max_nbytes="1M")(delayed(job)("P3", f, True, C, pid) for f in range(len(J))); P, cs = to_matrix(res); out["P3"] = P; out["P3__C"] = cs
    np.savez(f"{OUT}/A_prediksi_ketat.npz", **out); open(f"{OUT}/A_SELESAI", "w").write(time.ctime()); print(f"P3 strict done [{time.time() - t0:.0f}s]", flush=True)
