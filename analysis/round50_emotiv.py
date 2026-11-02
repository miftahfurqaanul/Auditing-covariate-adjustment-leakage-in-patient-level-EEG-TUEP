#!/usr/bin/env python3
"""Round 50B: plasmode on the independent Emotiv cohort (Zenodo 1252141): real EEG features, synthetic labels from acquisition covariates, feature blocks swapped between subjects (H1).
Parallel to round 42 (subject-level LR; window-level RF/XGB). See PRAREGISTRASI_putaran50.md section 50B.
Usage: python round50_emotiv.py --check | --run <LR|RF|XGB> [n_rep]   (resumable; output hasil50/B_<fam>.csv)"""
import os, sys, time, zlib
FAM = sys.argv[sys.argv.index("--run") + 1] if "--run" in sys.argv else None
if FAM != "XGB":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ["PYTHONWARNINGS"] = "ignore"
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, "/home/miftah_f/RTX-5090/tuep_fresh/code")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from scipy.signal import welch
from scipy.stats import norm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import evalkit as ek

T = "/home/miftah_f/RTX-5090/tuep_fresh/data"; OUT = os.path.join(HERE, "hasil50"); os.makedirs(OUT, exist_ok=True)
FS, MINWIN, GAMMAS = 200, 5, [0.0, 0.5, 1.0]
BANDS_R = [(1, 4), (4, 7.5), (7.5, 13), (13, 16), (16, 30), (30, 40)]          # Rohira bands
PIPES_LR = [(t, a) for t in ("raw", "rank") for a in ("none", "pat")]
PIPES_W = [("raw", "none"), ("raw", "win"), ("raw", "pat"), ("symlog", "pat"), ("rank", "pat"), ("log", "pat")]
DEFREPS = {"LR": 30, "RF": 30, "XGB": 20}


def kseed(*parts):
    return zlib.crc32("|".join(map(str, parts)).encode()) & 0x7fffffff


def raw_bandpower(W, chunk=500):
    """W: (n, 10, 2000) at 200 Hz -> (n, 60) sum of the Welch PSD (Hann, nperseg 400, noverlap 200) per Rohira band, ordered channel x band."""
    out = []
    for i in range(0, len(W), chunk):
        f, P = welch(W[i:i + chunk].astype(np.float64), fs=FS, nperseg=2 * FS, noverlap=FS, window="hann", axis=2)
        out.append(np.stack([P[..., (f >= lo) & (f < hi)].sum(-1) for lo, hi in BANDS_R], axis=-1).reshape(P.shape[0], -1))
    return np.concatenate(out)


def build():
    cache = f"{OUT}/emotiv_fitur.npz"
    zw = np.load(f"{T}/ext_zenodo_whole_windows.npz"); sid = zw["sid"]; zb = np.load(f"{T}/ext_zenodo_whole_bp.npz"); assert (zb["sid"] == sid).all()
    u, c = np.unique(sid, return_counts=True); keep = u[c >= MINWIN]; w = np.isin(sid, keep)
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True); XR = z["XR"]; assert len(XR) == w.sum()
    else:
        XR = raw_bandpower(zw["W"][w]); np.savez(cache, XR=XR, sid=sid[w])
    S = sid[w]; pos = {s: i for i, s in enumerate(keep)}; owner = np.array([pos[s] for s in S])
    XB = zb["X"][w].astype(np.float64); PB = np.vstack([np.median(XB[owner == p], axis=0) for p in range(len(keep))])
    idx = pd.read_csv(f"{T}/ext_zenodo_subjects.csv").set_index("sid").loc[keep]
    t = pd.to_datetime(idx.startTime, dayfirst=True); day = (t.dt.normalize() - t.dt.normalize().groupby(idx.country).transform("min")).dt.days.astype(float)
    hour = t.dt.hour + t.dt.minute / 60.0; z = lambda v: (v - v.mean()) / v.std(ddof=0)
    D = np.column_stack([np.ones(len(keep)), (idx.country == "NG").astype(float), idx.eyes.str.lower().str.startswith("closed").astype(float), z(day.values), z(hour.values)])
    Y = (idx.group == "epilepsy").astype(int).values
    return dict(XR=XR, owner=owner, PB=PB, D=D, Y=Y, keep=keep, country=idx.country.values)


E = build(); XR, OWN, PB, D, Y = E["XR"], E["owner"], E["PB"], E["D"], E["Y"]; N = len(Y); BLOCKS = [np.where(OWN == q)[0] for q in range(N)]
_lr = LogisticRegression(penalty=None, max_iter=5000).fit(D[:, 1:], Y); ETA = _lr.decision_function(D[:, 1:]); ETA = ETA - ETA.mean(); AUC_ETA_REAL = float(roc_auc_score(Y, ETA))


def gen_labels(rep, gamma):
    e = np.random.default_rng(kseed("emotiv_e", rep)).logistic(size=N); score = gamma * ETA + e; y = np.zeros(N, int); y[np.argsort(-score, kind="stable")[: N // 2]] = 1; return y


def gen_perm(rep):
    return np.random.default_rng(kseed("emotiv_perm", rep)).permutation(N)


def folds_for(rep, gamma, y):
    return list(StratifiedKFold(5, shuffle=True, random_state=kseed("emotiv_cv", rep, gamma)).split(np.zeros(N), y))


def split_val(tr, y, seed):
    g = np.random.default_rng(seed); va = []
    for c in (0, 1):
        ps = np.array(sorted(p for p in tr if y[p] == c)); va += list(g.permutation(ps)[: int(round(0.2 * len(ps)))])
    return np.array(sorted(set(tr) - set(va))), np.array(sorted(va))


Q = np.linspace(0, 1, 1001)


def make_transform(kind, Xfit):
    if kind == "raw":
        return lambda X: X
    if kind == "symlog":
        return lambda X: np.sign(X) * np.log1p(np.abs(X))
    if kind == "log":
        return lambda X: np.log(np.maximum(X, 1e-12))
    if kind == "rank":
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


def windows_for(perm):
    X = np.vstack([XR[BLOCKS[perm[p]]] for p in range(N)]); owner = np.concatenate([np.full(len(BLOCKS[perm[p]]), p) for p in range(N)]); return X, owner


def unit_lr(rep, gamma):
    y = gen_labels(rep, gamma); perm = gen_perm(rep); folds = folds_for(rep, gamma, y); X = PB[perm]; P = {p: np.zeros(N) for p in PIPES_LR}
    for tr, te in folds:
        for t, a in PIPES_LR:
            f = make_transform(t, X[tr]); Z = adjust_units(f(X), np.arange(N), tr, a); P[(t, a)][te] = ek.nested_lr().fit(Z[tr], y[tr]).predict_proba(Z[te])[:, 1]
    return [dict(fam="LR", rep=rep, cond="H1", gamma=gamma, transform=t, adjust=a, auc=float(roc_auc_score(y, p)), **{f"fold_auc_{i}": v for i, v in enumerate(fold_aucs(y, p, folds))}, auc_cov=float(roc_auc_score(y, ETA))) for (t, a), p in P.items()]


def window_clf(fam, Z, owner, y, tr, te, seed):
    tr_p, _ = split_val(tr, y, seed); fit = np.isin(owner, tr_p); sel = np.isin(owner, te)
    if fam == "RF":
        m = RandomForestClassifier(n_estimators=100, max_depth=1000, class_weight="balanced", n_jobs=1, random_state=seed).fit(Z[fit], y[owner[fit]])
    else:
        import xgboost as xgb
        m = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, tree_method="hist", device="cuda", random_state=seed, n_jobs=1, verbosity=0).fit(Z[fit], y[owner[fit]])
    return pd.Series(m.predict_proba(Z[sel])[:, 1]).groupby(owner[sel]).mean().reindex(te).values


def unit_window(fam, rep, gamma):
    y = gen_labels(rep, gamma); perm = gen_perm(rep); folds = folds_for(rep, gamma, y); X, owner = windows_for(perm); P = {p: np.zeros(N) for p in PIPES_W}
    for fi, (tr, te) in enumerate(folds):
        seed = kseed("emotiv_clf", rep, gamma, fi); trw = np.isin(owner, tr); cache = {}
        for t, a in PIPES_W:
            if t not in cache:
                cache[t] = make_transform(t, X[trw])(X)
            P[(t, a)][te] = window_clf(fam, adjust_units(cache[t], owner, tr, a), owner, y, tr, te, seed)
    return [dict(fam=fam, rep=rep, cond="H1", gamma=gamma, transform=t, adjust=a, auc=float(roc_auc_score(y, p)), **{f"fold_auc_{i}": v for i, v in enumerate(fold_aucs(y, p, folds))}, auc_cov=float(roc_auc_score(y, ETA))) for (t, a), p in P.items()]


def checks():
    n = 0; assert np.isfinite(XR).all() and np.isfinite(PB).all() and (XR > 0).all(); n += 1
    assert np.linalg.matrix_rank(D) == D.shape[1]; n += 1
    for g in GAMMAS:
        y = gen_labels(3, g); assert y.sum() == N // 2
    assert np.array_equal(gen_labels(5, 0.5), gen_labels(5, 0.5)) and not np.array_equal(gen_perm(5), gen_perm(6)); n += 1
    tr = np.arange(0, int(0.8 * N)); X, owner = windows_for(gen_perm(1)); trw = np.isin(owner, tr); te_rows = ~trw
    for t in ("symlog", "log", "rank"):
        Xp = X.copy(); Xp[te_rows] = Xp[te_rows] * 5 + 3; assert np.allclose(make_transform(t, X[trw])(X[trw]), make_transform(t, Xp[trw])(Xp[trw])); n += 1
    Z = make_transform("rank", X[trw])(X)
    for a in ("win", "pat"):
        Xp = Z.copy(); Xp[te_rows] = Xp[te_rows] * 5 + 3; assert np.allclose(adjust_units(Z, owner, tr, a)[trw], adjust_units(Xp, owner, tr, a)[trw]); n += 1
    M = patient_means(adjust_units(Z, owner, tr, "pat"), owner); assert np.abs(D[tr].T @ M[tr]).max() / len(tr) < 1e-8; n += 1
    Zw = adjust_units(Z, owner, tr, "win"); rows = np.isin(owner, tr); assert np.abs(D[owner][rows].T @ Zw[rows]).max() / rows.sum() < 1e-8; n += 1
    y = gen_labels(4, 1.0); tr_, te_ = folds_for(4, 1.0, y)[0]; y2 = y.copy(); y2[te_] = 1 - y2[te_]; Xw, ow = windows_for(gen_perm(4)); trw_ = np.isin(ow, tr_)
    Zr = adjust_units(make_transform("rank", Xw[trw_])(Xw), ow, tr_, "pat"); assert np.allclose(window_clf("RF", Zr, ow, y, tr_, te_, 7), window_clf("RF", Zr, ow, y2, tr_, te_, 7)); n += 1
    Xl = PB[gen_perm(4)]; Zl = adjust_units(make_transform("rank", Xl[tr_])(Xl), np.arange(N), tr_, "pat")
    assert np.allclose(ek.nested_lr().fit(Zl[tr_], y[tr_]).predict_proba(Zl[te_])[:, 1], ek.nested_lr().fit(Zl[tr_], y2[tr_]).predict_proba(Zl[te_])[:, 1]); n += 1
    cal = {g: np.mean([roc_auc_score(gen_labels(i, g), ETA) for i in range(100)]) for g in GAMMAS}
    cnt = np.bincount(OWN, minlength=N)
    print(f"{n} checks passed; subjects {N} (epilepsy {Y.sum()}; Nigeria {int(D[:, 1].sum())}); windows {len(XR)} ({cnt.min()}--{cnt.max()} per subject, median {np.median(cnt):.0f}); window features {XR.shape[1]}, subject features {PB.shape[1]}", flush=True)
    print(f"AUC eta -> real label {AUC_ETA_REAL:.4f}; AUC eta -> y_sim (100 replicates): " + "; ".join(f"gamma {g:g}: {v:.3f}" for g, v in cal.items()), flush=True)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "--check"; t0 = time.time()
    if mode == "--check":
        checks()
    elif mode == "--run":
        nrep = int(sys.argv[3]) if len(sys.argv) > 3 else DEFREPS[FAM]; path = f"{OUT}/B_{FAM}.csv"
        done = set(map(tuple, pd.read_csv(path, usecols=["rep", "gamma"]).drop_duplicates().values.tolist())) if os.path.exists(path) else set()
        units = [(r, g) for r in range(nrep) for g in GAMMAS if (r, g) not in done]; print(f"{FAM}: {len(units)} units left of {nrep * 3}", flush=True)
        fn = unit_lr if FAM == "LR" else (lambda r, g: unit_window(FAM, r, g))
        if FAM == "XGB":
            for i, (r, g) in enumerate(units):
                pd.DataFrame(fn(r, g)).to_csv(path, mode="a", header=not os.path.exists(path), index=False)
                if i % 10 == 9:
                    print(f"{i + 1}/{len(units)} [{time.time() - t0:.0f}s]", flush=True)
        else:
            nj = int(os.environ.get("NJOBS", 12))
            for i in range(0, len(units), nj * 2):
                chunk = units[i:i + nj * 2]; res = Parallel(n_jobs=nj)(delayed(fn)(r, g) for r, g in chunk)
                for rr in res:
                    pd.DataFrame(rr).to_csv(path, mode="a", header=not os.path.exists(path), index=False)
                print(f"{min(i + nj * 2, len(units))}/{len(units)} [{time.time() - t0:.0f}s]", flush=True)
    print(f"done {time.time() - t0:.0f}s", flush=True)
