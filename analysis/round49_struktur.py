#!/usr/bin/env python3
"""Round 49D (CPU): isolation of the window structure within patients. Features without disease signal (H1); factors ICC x m x marginal distribution (fixed for all ICC);
arm A (per-window adjustment, window RF), B (per patient, window RF), C (aggregate first, patient RF), and patient LR. See PRAREGISTRASI_putaran49.md section 49D.
Usage: CUDA_VISIBLE_DEVICES="" python round49_struktur.py --check | --run [n_rep=20]   (resumable; raw output hasil49/D_replikasi.csv)"""
import os, sys, time, json, hashlib, zlib, itertools
os.environ["CUDA_VISIBLE_DEVICES"] = ""; os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ["PYTHONWARNINGS"] = "ignore"
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
OUT = os.path.join(HERE, "hasil49"); os.makedirs(OUT, exist_ok=True)
ERA1 = np.array([.25, .16, .20, .24, .15]); ERA0 = np.array([.06, .16, .22, .24, .32])          # same as round39 (TUEP cohort table)
ICCS, MS, DISTS = (0.0, 0.3, 0.7, 0.95), (10, 50, 200), ("normal", "lognormal", "contam"); N, P, S = 200, 20, 1.0


def kseed(*parts):
    return zlib.crc32("|".join(map(str, parts)).encode()) & 0x7fffffff


def make_data(seed, icc, m, dist):
    """Covariates and label as round39.make_data (s = 1); latent N(0,1) marginal features for all ICC; features independent of the label/covariates."""
    rng = np.random.default_rng(seed); y = np.r_[np.zeros(N // 2), np.ones(N - N // 2)]; rng.shuffle(y)
    age = 55 + 15 * (rng.standard_normal(N) - 0.45 * S * (y - 0.5) * 2 * 0.5); sex = rng.binomial(1, 0.545, N).astype(float)
    base = (ERA1 + ERA0) / 2; p1 = (1 - S) * base + S * ERA1; p0 = (1 - S) * base + S * ERA0
    era = np.array([rng.choice(5, p=(p1 if yy == 1 else p0)) for yy in y])
    D = np.column_stack([np.ones(N), (age - age.mean()) / age.std(), sex] + [(era == k).astype(float) for k in (0, 1, 2, 4)])
    pid = np.repeat(np.arange(N), m); u = rng.standard_normal((N, P)); e = rng.standard_normal((N * m, P)); z = np.sqrt(icc) * u[pid] + np.sqrt(1 - icc) * e
    if dist == "normal":
        X = z
    elif dist == "lognormal":
        X = np.exp(z)
    else:
        X = z * np.where(rng.random(N * m) < 0.01, 100.0, 1.0)[:, None]
    return X, pid, y, D


def adjust_win(X, pid, D, trw):
    B = np.linalg.lstsq(D[pid[trw]], X[trw], rcond=None)[0]; return X - D[pid] @ B


def pmean(X, pid):
    Sm = np.zeros((N, X.shape[1])); np.add.at(Sm, pid, X); return Sm / np.bincount(pid, minlength=N)[:, None]


def adjust_pat(M, D, tr):
    B = np.linalg.lstsq(D[tr], M[tr], rcond=None)[0]; return M - D @ B


def rf_win(Z, pid, y, tr, te, seed):
    trw, tew = np.isin(pid, tr), np.isin(pid, te); m = RandomForestClassifier(n_estimators=100, class_weight="balanced", n_jobs=1, random_state=seed).fit(Z[trw], y[pid[trw]])
    return pd.Series(m.predict_proba(Z[tew])[:, 1]).groupby(pid[tew]).mean().reindex(te).values


def rf_pat(M, y, tr, te, seed):
    return RandomForestClassifier(n_estimators=100, class_weight="balanced", n_jobs=1, random_state=seed).fit(M[tr], y[tr]).predict_proba(M[te])[:, 1]


def lr_pat(M, y, tr, te):
    return make_pipeline(StandardScaler(), LogisticRegressionCV(Cs=np.logspace(-3, 1, 9), cv=5, scoring="roc_auc", class_weight="balanced", max_iter=2000)).fit(M[tr], y[tr]).predict_proba(M[te])[:, 1]


ARMS = ["RFwin_none", "RFwin_A", "RFwin_B", "RFpat_none", "RFpat_C", "LRpat_none", "LRpat_win", "LRpat_pat"]


def replicate(icc, m, dist, rep):
    seed = kseed("p49D", icc, m, dist, rep); X, pid, y, D = make_data(seed, icc, m, dist); P_ = {a: np.full(N, np.nan) for a in ARMS}
    for k, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed % 100000).split(np.zeros(N), y)):
        trw = np.isin(pid, tr); s2 = seed % 100000 + k; assert not set(tr) & set(te)
        Za = adjust_win(X, pid, D, trw); M0 = pmean(X, pid); Mp = adjust_pat(M0, D, tr); Zb = X - (M0 - Mp)[pid]          # B: patient-level offset from the OLS of the training patient means
        P_["RFwin_none"][te] = rf_win(X, pid, y, tr, te, s2); P_["RFwin_A"][te] = rf_win(Za, pid, y, tr, te, s2); P_["RFwin_B"][te] = rf_win(Zb, pid, y, tr, te, s2)
        P_["RFpat_none"][te] = rf_pat(M0, y, tr, te, s2); P_["RFpat_C"][te] = rf_pat(Mp, y, tr, te, s2)
        P_["LRpat_none"][te] = lr_pat(M0, y, tr, te); P_["LRpat_win"][te] = lr_pat(pmean(Za, pid), y, tr, te); P_["LRpat_pat"][te] = lr_pat(Mp, y, tr, te)
    row = dict(icc=icc, m=m, dist=dist, rep=rep); row.update({a: roc_auc_score(y, P_[a]) for a in ARMS}); return row


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "--check"; t0 = time.time()
    json.dump(dict(prereg_sha256=sha(os.path.join(HERE, "PRAREGISTRASI_putaran49.md")), code_sha256=sha(__file__)), open(f"{OUT}/meta_D.json", "w"), indent=1)
    if mode == "--check":                                   # properties of the generator and the adjustment; does not compute the AUC of any arm
        for icc in ICCS:
            X, pid, y, D = make_data(1, icc, 50, "normal"); M = pmean(X, pid); r = M.var(0).mean() - (1 - icc) / 50; print(f"ICC {icc}: marginal var {X.var(0).mean():.3f}; var of patient means between {M.var(0).mean():.3f} (expected {icc + (1 - icc) / 50:.3f})")
            assert abs(X.var(0).mean() - 1) < 0.05
        X, pid, y, D = make_data(2, 0.3, 50, "lognormal"); tr = np.arange(160); trw = np.isin(pid, tr); Za = adjust_win(X, pid, D, trw)
        assert np.abs(D[pid[trw]].T @ Za[trw]).max() / trw.sum() < 1e-8; M0 = pmean(X, pid); Mp = adjust_pat(M0, D, tr); assert np.abs(D[tr].T @ Mp[tr]).max() / len(tr) < 1e-8
        Zb = X - (M0 - Mp)[pid]; assert np.allclose(pmean(Zb, pid), Mp); X2 = X.copy(); X2[~trw] *= 9; assert np.allclose(adjust_win(X2, pid, D, trw)[trw], Za[trw])
        print(f"checks passed; one replicate m=200 ...", flush=True); t1 = time.time(); replicate(0.7, 200, "lognormal", 999); print(f"{time.time() - t1:.0f}s (AUC not printed)"); sys.exit()
    nrep = int(sys.argv[2]) if len(sys.argv) > 2 else 20; path = f"{OUT}/D_replikasi.csv"; done = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame(columns=["icc", "m", "dist", "rep"])
    todo = [(i, m, d, r) for d in DISTS for m in MS for i in ICCS for r in range(nrep) if not ((done.icc == i) & (done.m == m) & (done.dist == d) & (done.rep == r)).any()]
    nj = int(os.environ.get("NJOBS49", "6"))
    for b in range(0, len(todo), 2 * nj):
        rows = Parallel(n_jobs=nj)(delayed(replicate)(*t) for t in todo[b:b + 2 * nj]); done = pd.concat([done, pd.DataFrame(rows)], ignore_index=True); done.to_csv(path, index=False)
        print(f"{len(done)} rows saved; remaining {len(todo) - b - len(rows)} [{time.time() - t0:.0f}s]", flush=True)
    open(f"{OUT}/D_SELESAI", "w").write(time.ctime())
