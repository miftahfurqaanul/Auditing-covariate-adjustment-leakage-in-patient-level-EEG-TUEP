#!/usr/bin/env python3
"""Putaran 52 (CPU): pembanding dengan penyesuaian ash pada label nyata (50 lipatan kanonis) dan pada kondisi nol H1/H2a:
DNM (tingkat pasien), Rohira (RF tingkat jendela), embedding beku BENDR/LaBraM/CBraMod (PCA 50 -> rank -> penyesuaian -> LR). Penyesuaian V4 (OLS) diganti ash
(round51_common.ash_coef; tingkat jendela lewat rata-rata per pasien). Kode putaran 33/43/51 tidak diubah. Lihat PRAREGISTRASI_putaran52.md.
Pemakaian: CUDA_VISIBLE_DEVICES="" python round52_cpu.py --check | --run real | --run null   (resumable; NJOBS default 12; hasil52/)"""
import os, sys, time, hashlib, json
os.environ["CUDA_VISIBLE_DEVICES"] = ""; os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ["PYTHONWARNINGS"] = "ignore"
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, "/home/miftah_f/RTX-5090/tuep_fresh/code")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import round33_pembanding_v4 as R33
import round43_fm_kompresi as R43
import round51_common as K
import evalkit as ek
A, Y, N, J, D, AS = R33.A, R33.Y, R33.N, R33.J, R33.D, R33.AS
OUT = os.path.join(HERE, "hasil52"); os.makedirs(f"{OUT}/cpu", exist_ok=True); NJOBS = int(os.environ.get("NJOBS", 12))
EMB = ["LaBraM-eq", "CBraMod-eq", "BENDR-eq", "BP-eq"]; NULL_EMB = ["BP-eq", "LaBraM-eq", "CBraMod-eq", "BENDR-eq"]


def adjust(X, fit, owner=None, how="ash"):
    """how='v4': R33.adjust (OLS); how='ash': koefisien ash atas rata-rata per pasien latih (owner=None: baris = pasien). Residu dibagi SD residu pada baris latih (seperti V4)."""
    if how == "v4":
        return R33.adjust(X, fit, owner)
    X = X.astype(np.float64)
    if owner is None:
        B = K.ash_coef(D[fit], X[fit])[0]; R = X - D @ B; rows = np.zeros(len(X), bool); rows[fit] = True; Dm = D
    else:
        B = K.ash_coef(D[fit], K.patient_means(X, owner, N)[fit])[0]; Dm = D[owner]; R = X - Dm @ B; rows = np.isin(owner, fit)
    sd = R[rows].std(0, ddof=np.linalg.matrix_rank(Dm[rows])); return R / np.where(sd < 1e-8, 1.0, sd)


def dnm_fold(X, how, tr, te, use_as=True):
    F = adjust(X, tr, None, how)
    if use_as:
        F = np.hstack([F, AS])
    return ek.nested_lr().fit(F[tr], Y[tr]).predict_proba(F[te])[:, 1]


def rohira_fold(Xw, pidx, how, use_as, r, k, tr, te):
    tr_p, _ = R33.split_val(list(R33.SUBJ[tr]), R33.YLAB, 100 * r + k); Z = adjust(Xw, tr, pidx, how)
    if use_as:
        Z = np.hstack([Z, AS[pidx]])
    fit = np.isin(R33.SUBJ[pidx], tr_p); m = RandomForestClassifier(n_estimators=100, max_depth=1000, class_weight="balanced", n_jobs=1, random_state=1000 * r + k)
    m.fit(Z[fit], Y[pidx][fit]); sel = np.isin(pidx, te)
    return pd.Series(m.predict_proba(Z[sel])[:, 1]).groupby(pidx[sel]).mean().loc[te].values


def feats(X, fit, use_pca, use_as, how):
    """Seperti round43_fm_kompresi.feats; how='v4' OLS, how='ash' ash (koefisien atas pasien fit)."""
    F = X.astype(float)
    if use_pca:
        sc = StandardScaler().fit(F[fit]); pca = PCA(n_components=min(R43.NPC, len(fit) - 1), random_state=0).fit(sc.transform(F[fit])); F = pca.transform(sc.transform(F))
    F = R43.rank_fit_apply(F, fit); B = np.linalg.lstsq(D[fit], F[fit], rcond=None)[0] if how == "v4" else K.ash_coef(D[fit], F[fit])[0]
    R = F - D @ B; sd = R[fit].std(0, ddof=np.linalg.matrix_rank(D[fit])); R = R / np.where(sd < 1e-8, 1.0, sd)
    if use_as:
        R = np.hstack([R, (AS - AS[fit].mean(0)) / AS[fit].std(0)])
    return R


def emb_fold(X, y, tr, te, use_pca, use_as, seed, how):
    sc = {C: [] for C in R43.CS}
    for a, b in StratifiedKFold(5, shuffle=True, random_state=seed).split(tr, y[tr]):
        ia, ib = tr[a], tr[b]; F = feats(X, ia, use_pca, use_as, how)
        for C in R43.CS:
            sc[C].append(roc_auc_score(y[ib], R43.lr(C).fit(F[ia], y[ia]).decision_function(F[ib])))
    best = sorted(R43.CS, key=lambda C: (-np.mean(sc[C]), C))[0]; F = feats(X, tr, use_pca, use_as, how)
    return R43.lr(best).fit(F[tr], y[tr]).predict_proba(F[te])[:, 1], best


def mat(preds):
    return A.c17.to_matrix(J, preds)


def check():
    n = 0; r, k, tr, te = J[0]
    X2 = R33.XD.copy(); X2[te] = X2[te] * 5 + 3; assert np.allclose(adjust(R33.XD, tr)[tr], adjust(X2, tr)[tr]); n += 1
    Xw2 = R33.XR.copy(); mt = np.isin(R33.PIDX, te); Xw2[mt] = Xw2[mt] * 5 + 3; assert np.allclose(adjust(R33.XR, tr, R33.PIDX)[~mt], adjust(Xw2, tr, R33.PIDX)[~mt]); n += 1
    # reproduksi V4: prediksi tersimpan putaran 33 dan 43 (lipatan 0 dan 17)
    Z = np.load(f"{R33.OUT}/pembanding_prediksi.npz"); E = np.load(f"{R43.OUT}/A_prediksi.npz"); M = R43.load_mats()
    for f in (0, 17):
        r, k, tr, te = J[f]; d = np.abs(dnm_fold(R33.XD, "v4", tr, te) - Z["DNM_V4"][r, te]).max(); assert d < 1e-8, d; n += 1
        d = np.abs(rohira_fold(R33.XR, R33.PIDX, "v4", True, r, k, tr, te) - Z["Rohira_V4_AS_AS"][r, te]).max(); assert d < 1e-8, d; n += 1
        d = np.abs(emb_fold(M["LaBraM"], Y, tr, te, True, True, R43.kseed("e43", f), "v4")[0] - E["LaBraM-eq__AS"][r, te]).max(); assert d < 1e-8, d; n += 1
    # ash: sisa tidak lagi ortogonal terhadap desain (disengaja), tetapi prediksi hanya memakai pasien latih dan label uji tidak dipakai
    y2 = Y.copy(); y2[te] = 1 - y2[te]; a = emb_fold(M["BENDR"], Y, tr, te, True, True, 3, "ash")[0]; b = emb_fold(M["BENDR"], y2, tr, te, True, True, 3, "ash")[0]; assert np.allclose(a, b); n += 1
    Fa = feats(M["LaBraM"], tr, True, False, "ash"); Fo = feats(M["LaBraM"], tr, True, False, "v4"); print(f"{n} pemeriksaan lulus; |korelasi sisa-desain| ash {np.abs(D[tr].T @ Fa[tr]).max() / len(tr):.3f} vs v4 {np.abs(D[tr].T @ Fo[tr]).max() / len(tr):.1e}", flush=True)


def run_real():
    t0 = time.time(); M = R43.load_mats(); jobs = []
    jobs.append(("DNM-ash", lambda f: dnm_fold(R33.XD, "ash", J[f][2], J[f][3])))
    jobs.append(("Rohira-ash-AS", lambda f: rohira_fold(R33.XR, R33.PIDX, "ash", True, J[f][0], J[f][1], J[f][2], J[f][3])))
    jobs.append(("Rohira-ash", lambda f: rohira_fold(R33.XR, R33.PIDX, "ash", False, J[f][0], J[f][1], J[f][2], J[f][3])))
    for name, fn in jobs:
        path = f"{OUT}/cpu/{name}.npy"
        if os.path.exists(path):
            continue
        np.save(path, mat(Parallel(n_jobs=NJOBS)(delayed(fn)(f) for f in range(len(J))))); print(name, f"[{time.time() - t0:.0f}s]", flush=True)
    for name in EMB:
        path = f"{OUT}/cpu/{name}__AS_ash.npy"
        if os.path.exists(path):
            continue
        key, pca = R43.MODELS[name]; out = Parallel(n_jobs=NJOBS)(delayed(emb_fold)(M[key], Y, J[f][2], J[f][3], pca, True, R43.kseed("e43", f), "ash") for f in range(len(J)))
        np.save(path, mat([o[0] for o in out])); np.save(path.replace(".npy", "__C.npy"), np.array([o[1] for o in out])); print(name, f"[{time.time() - t0:.0f}s]", flush=True)


def run_null():
    t0 = time.time(); M = R43.load_mats(); path = f"{OUT}/cpu_null.csv"; done = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame(columns=["model", "scen", "rep", "fold_mean", "auc"])
    def add(row):
        nonlocal done
        done = pd.concat([done, pd.DataFrame([row])], ignore_index=True); done.to_csv(path, index=False)
    def have(m, s, r):
        return ((done.model == m) & (done.scen == s) & (done.rep == r)).any()
    for rep in range(20):                                                   # embedding beku: benih dan permutasi identik dengan round43 run_C
        for scen in ("H1", "H2a"):
            rng = np.random.default_rng(43000 + rep + (0 if scen == "H1" else 500))
            if scen == "H1":
                perm = rng.permutation(N)
            else:
                perm = np.arange(N)
                for e_ in np.unique(R43.ERA):
                    ii = np.where(R43.ERA == e_)[0]; perm[ii] = rng.permutation(ii)
            for name in NULL_EMB:
                if have(name + "-ash", scen, rep):
                    continue
                key, pca = R43.MODELS[name]; Xs = M[key][perm]; out = Parallel(n_jobs=NJOBS)(delayed(emb_fold)(Xs, Y, J[f][2], J[f][3], pca, False, R43.kseed("e43null", rep, f), "ash") for f in range(5)); P = np.full(N, np.nan); fa = []
                for f in range(5):
                    P[J[f][3]] = out[f][0]; fa.append(roc_auc_score(Y[J[f][3]], out[f][0]))
                add(dict(model=name + "-ash", scen=scen, rep=rep, fold_mean=float(np.mean(fa)), auc=float(roc_auc_score(Y, P))))
        print(f"emb rep {rep} [{time.time() - t0:.0f}s]", flush=True)
    for nm_, n_rep in (("DNM", 20), ("Rohira", 10)):                         # benih 33000 + r, seperti round33 (pooled AUC sebagai nilai tunggal)
        for scen in ("H1", "H2a"):
            for r in range(n_rep):
                if have(nm_ + "-ash", scen, r):
                    continue
                perm = R33.perm_owner(scen, 33000 + r)
                if nm_ == "DNM":
                    X = R33.XD[perm]; v = R33.pooled(mat(Parallel(n_jobs=NJOBS)(delayed(dnm_fold)(X, "ash", J[f][2], J[f][3], False) for f in range(len(J)))))
                else:
                    blocks = [np.where(R33.PIDX == q)[0] for q in range(N)]; Xn = np.vstack([R33.XR[blocks[perm[p]]] for p in range(N)]); pn = np.concatenate([np.full(len(blocks[perm[p]]), p) for p in range(N)])
                    v = R33.pooled(mat(Parallel(n_jobs=NJOBS)(delayed(rohira_fold)(Xn, pn, "ash", False, J[f][0], J[f][1], J[f][2], J[f][3]) for f in range(len(J)))))
                add(dict(model=nm_ + "-ash", scen=scen, rep=r, fold_mean=np.nan, auc=float(v)))
            print(nm_, scen, f"[{time.time() - t0:.0f}s]", flush=True)


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "--check"
    if mode == "--check":
        check(); sys.exit()
    json.dump(dict(prereg_sha256=sha(os.path.join(HERE, "PRAREGISTRASI_putaran52.md")), code_sha256=sha(__file__), common_sha256=sha(os.path.join(HERE, "round51_common.py"))), open(f"{OUT}/meta_cpu.json", "w"), indent=1)
    run_real() if sys.argv[2] == "real" else run_null()
    print("selesai", flush=True)
