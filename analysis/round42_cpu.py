#!/usr/bin/env python3
"""Round 42 (CPU/GPU): plasmode LR-I2, RF-Rohira (CPU) and XGB-Rohira (GPU). See PRAREGISTRASI_putaran42.md.
Usage: python round42_cpu.py --check | --pilot | --run <LR|RF|XGB> [n_rep]   (resumable; LR/RF disable the GPU)"""
import os, sys, time
FAM = sys.argv[sys.argv.index("--run") + 1] if "--run" in sys.argv else None
if FAM != "XGB":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ["PYTHONWARNINGS"] = "ignore"
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import round42_common as C
import evalkit as ek
import round33_pembanding_v4 as R3

OUT = os.path.join(C.HERE, "hasil42"); os.makedirs(OUT, exist_ok=True)
N, D, Y = C.N, C.D, C.Y; XR = R3.XR; PIDX = R3.PIDX; BP = C.A.BP.astype(float); BLOCKS = [np.where(PIDX == q)[0] for q in range(N)]
PIPES_LR = [(t, a) for t in ("raw", "symlog", "rank") for a in ("none", "pat")]
PIPES_RF = [("raw", "none"), ("raw", "win"), ("raw", "pat"), ("symlog", "pat"), ("rank", "pat"), ("log", "pat")]
COND = {"LR": ["H1", "H2a", "nyata"], "RF": ["H1", "H2a", "nyata"], "XGB": ["H1", "H2a"]}; DEFREPS = {"LR": 30, "RF": 30, "XGB": 20}


def windows_for(perm):
    """The window block of source patient perm[p] is given to target patient p."""
    X = np.vstack([XR[BLOCKS[perm[p]]] for p in range(N)]); owner = np.concatenate([np.full(len(BLOCKS[perm[p]]), p) for p in range(N)]); return X, owner


def unit_lr(rep, cond, gamma):
    y = C.gen_labels(rep, gamma); perm = C.gen_perm(rep, cond); folds = C.folds_for(rep, gamma, y); X = BP[perm]; rows = []; auc_cov = float(roc_auc_score(y, C.ETA)); P = {p: np.zeros(N) for p in PIPES_LR}
    for tr, te in folds:
        for t, a in PIPES_LR:
            f = C.make_transform(t, X[tr]); Z = C.adjust_units(f(X), np.arange(N), tr, a); P[(t, a)][te] = ek.nested_lr().fit(Z[tr], y[tr]).predict_proba(Z[te])[:, 1]
    for (t, a), p in P.items():
        fa = C.fold_aucs(y, p, folds); rows.append(dict(fam="LR", rep=rep, cond=cond, gamma=gamma, transform=t, adjust=a, auc=float(roc_auc_score(y, p)), **{f"fold_auc_{i}": v for i, v in enumerate(fa)}, auc_cov=auc_cov))
    return rows


def window_clf(fam, Z, owner, y, tr, te, seed, dev=None):
    tr_p, _ = C.split_val(tr, y, seed); fit = np.isin(owner, tr_p); sel = np.isin(owner, te)
    if fam == "RF":
        m = RandomForestClassifier(n_estimators=100, max_depth=1000, class_weight="balanced", n_jobs=1, random_state=seed).fit(Z[fit], y[owner[fit]]); pr = m.predict_proba(Z[sel])[:, 1]
    else:
        import xgboost as xgb
        m = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, tree_method="hist", device="cuda", random_state=seed, n_jobs=1, verbosity=0).fit(Z[fit], y[owner[fit]]); pr = m.predict_proba(Z[sel])[:, 1]
    return pd.Series(pr).groupby(owner[sel]).mean().reindex(te).values


def unit_window(fam, rep, cond, gamma):
    y = C.gen_labels(rep, gamma); perm = C.gen_perm(rep, cond); folds = C.folds_for(rep, gamma, y); X, owner = windows_for(perm); auc_cov = float(roc_auc_score(y, C.ETA)); P = {p: np.zeros(N) for p in PIPES_RF}
    for fi, (tr, te) in enumerate(folds):
        seed = C.kseed("plasmode_clf", rep, gamma, fi); trw = np.isin(owner, tr); cache = {}
        for t, a in PIPES_RF:
            if t not in cache:
                cache[t] = C.make_transform(t, X[trw])(X)
            Z = C.adjust_units(cache[t], owner, tr, a); P[(t, a)][te] = window_clf(fam, Z, owner, y, tr, te, seed)
    rows = []
    for (t, a), p in P.items():
        fa = C.fold_aucs(y, p, folds); rows.append(dict(fam=fam, rep=rep, cond=cond, gamma=gamma, transform=t, adjust=a, auc=float(roc_auc_score(y, p)), **{f"fold_auc_{i}": v for i, v in enumerate(fa)}, auc_cov=auc_cov))
    return rows


def checks():
    n = 0
    for g in C.GAMMAS:
        y = C.gen_labels(3, g); assert y.sum() == 100 and len(y) == 200
    a0, a1 = roc_auc_score(C.gen_labels(3, 0.0), C.ETA), roc_auc_score(C.gen_labels(3, 1.0), C.ETA); assert a1 > a0; n += 1
    assert np.array_equal(C.gen_labels(5, 0.5), C.gen_labels(5, 0.5)) and np.array_equal(C.gen_perm(5, "H1"), C.gen_perm(5, "H1")) and not np.array_equal(C.gen_perm(5, "H1"), C.gen_perm(6, "H1")); n += 1
    pi = C.gen_perm(2, "H2a"); assert all(set(C.ERA[pi[C.ERA == e]]) == {e} for e in np.unique(C.ERA)) and sorted(pi) == list(range(N)); n += 1                                   # H2a: within era
    tr = np.arange(0, 160); X, owner = windows_for(C.gen_perm(1, "H1")); trw = np.isin(owner, tr); te_rows = ~trw
    for t in ("symlog", "log", "rank"):
        Xp = X.copy(); Xp[te_rows] = Xp[te_rows] * 5 + 3; f1 = C.make_transform(t, X[trw]); f2 = C.make_transform(t, Xp[trw]); assert np.allclose(f1(X[trw]), f2(Xp[trw])); n += 1           # test data do not change the training transform
    Z = C.make_transform("rank", X[trw])(X); assert abs(Z[trw].mean()) < 0.05 and abs(Z[trw].std() - 1) < 0.2
    for a in ("win", "pat"):
        Za = C.adjust_units(Z, owner, tr, a); Xp = Z.copy(); Xp[te_rows] = Xp[te_rows] * 5 + 3; Zb = C.adjust_units(Xp, owner, tr, a); assert np.allclose(Za[trw], Zb[trw]); n += 1                       # training offset does not depend on test data
    Zp = C.adjust_units(Z, owner, tr, "pat"); M = C.patient_means(Zp, owner); assert np.abs(D[tr].T @ M[tr]).max() / len(tr) < 1e-8; n += 1                                  # patient-mean residual orthogonal to the design
    Zw = C.adjust_units(Z, owner, tr, "win"); rows = np.isin(owner, tr); assert np.abs(D[owner][rows].T @ Zw[rows]).max() / rows.sum() < 1e-8; n += 1
    y = C.gen_labels(4, 1.0); folds = C.folds_for(4, 1, y); tr_, te_ = folds[0]; y2 = y.copy(); y2[te_] = 1 - y2[te_]; Xw, ow = windows_for(C.gen_perm(4, "H1")); trw_ = np.isin(ow, tr_); Zr = C.adjust_units(C.make_transform("rank", Xw[trw_])(Xw), ow, tr_, "pat")
    a = window_clf("RF", Zr, ow, y, tr_, te_, 7); b = window_clf("RF", Zr, ow, y2, tr_, te_, 7); assert np.allclose(a, b); n += 1                                                      # test labels are not used (RF)
    Xl = BP[C.gen_perm(4, "H1")]; Zl = C.adjust_units(C.make_transform("rank", Xl[tr_])(Xl), np.arange(N), tr_, "pat"); a = ek.nested_lr().fit(Zl[tr_], y[tr_]).predict_proba(Zl[te_])[:, 1]; b = ek.nested_lr().fit(Zl[tr_], y2[tr_]).predict_proba(Zl[te_])[:, 1]; assert np.allclose(a, b); n += 1
    print(f"{n} checks passed; AUC eta -> real label {C.AUC_ETA_REAL:.4f}; AUC eta -> y_sim (rep 3): gamma 0 {a0:.3f}, gamma 1 {a1:.3f}; median kurtosis XR {np.median(__import__('scipy.stats', fromlist=['x']).kurtosis(XR, axis=0)):.0f}", flush=True)


def calib_table(nrep=100):
    r = []
    for g in C.GAMMAS:
        v = [roc_auc_score(C.gen_labels(i, g), C.ETA) for i in range(nrep)]; r.append((g, np.mean(v), np.std(v)))
    return r


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "--check"; t0 = time.time()
    if mode == "--check":
        checks(); print("calibration AUC(eta -> y_sim) (100 replicates): " + "; ".join(f"gamma {g}: {m:.3f} (SD {s:.3f})" for g, m, s in calib_table()))
    elif mode == "--run":
        fam = sys.argv[2]; nrep = int(sys.argv[3]) if len(sys.argv) > 3 else DEFREPS[fam]; path = f"{OUT}/{fam}.csv"
        done = set(map(tuple, pd.read_csv(path, usecols=["rep", "cond", "gamma"]).drop_duplicates().values.tolist())) if os.path.exists(path) else set()
        units = [(r, c, g) for r in range(nrep) for c in COND[fam] for g in C.GAMMAS if (r, c, g) not in done]; print(f"{fam}: {len(units)} units remaining of {nrep * len(COND[fam]) * 3}", flush=True)
        fn = unit_lr if fam == "LR" else (lambda r, c, g: unit_window(fam, r, c, g))
        if fam == "XGB":
            for i, (r, c, g) in enumerate(units):
                pd.DataFrame(fn(r, c, g)).to_csv(path, mode="a", header=not os.path.exists(path), index=False)
                if i % 9 == 8:
                    print(f"{i + 1}/{len(units)} [{time.time() - t0:.0f}s]", flush=True)
        else:
            nj = int(os.environ.get("NJOBS", 12))
            for i in range(0, len(units), nj * 2):
                chunk = units[i:i + nj * 2]; res = Parallel(n_jobs=nj)(delayed(fn)(r, c, g) for r, c, g in chunk)
                for rr in res:
                    pd.DataFrame(rr).to_csv(path, mode="a", header=not os.path.exists(path), index=False)
                print(f"{min(i + nj * 2, len(units))}/{len(units)} [{time.time() - t0:.0f}s]", flush=True)
    print(f"done {time.time() - t0:.0f}s", flush=True)
