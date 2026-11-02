#!/usr/bin/env python3
"""Round 43 (CPU): equal-treatment pipeline for frozen embeddings (BENDR, LaBraM, CBraMod) and 50-feature anatomical compression (Comp50); R1 leakage test. See PRAREGISTRASI_putaran43.md.
Usage: CUDA_VISIBLE_DEVICES="" python round43_fm_kompresi.py --prep | --check | --run A | --run C   (resumable; NJOBS default 8)"""
import os, sys, time
os.environ["CUDA_VISIBLE_DEVICES"] = ""; os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ["PYTHONWARNINGS"] = "ignore"
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, "/home/miftah_f/RTX-5090/tuep_fresh/code")
import numpy as np, pandas as pd, zlib
from scipy.stats import norm
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import round28_analisis as A
import eegbench_fixed_eval as ebf

OUT = os.path.join(HERE, "hasil43"); os.makedirs(OUT, exist_ok=True); DATA = "/home/miftah_f/RTX-5090/tuep_fresh/data"
Y, N, J, SUBJ, ERA, D = A.Y, A.N, A.J, A.c17.SUBJ, A.ERA, A.D_I2; AS = A.c17.AS.astype(float); NJOBS = int(os.environ.get("NJOBS", 8))
CS = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]; Q = np.linspace(0, 1, 1001); NPC = 50
REGIONS = {"frontal": ["FP1", "FP2", "F7", "F3", "FZ", "F4", "F8"], "sentral": ["C3", "CZ", "C4"], "temporal": ["T3", "T4", "T5", "T6"], "parietal": ["P3", "PZ", "P4"], "oksipital": ["O1", "O2"]}; BANDS = ["delta", "theta", "alpha", "beta", "gamma"]
# model -> (patient matrix key, PCA?)
MODELS = {"BP-eq": ("BP", True), "BP-rank": ("BP", False), "LaBraM-eq": ("LaBraM", True), "CBraMod-eq": ("CBraMod_L12", True), "CBraMod-L4": ("CBraMod_L4", True), "CBraMod-L8": ("CBraMod_L8", True), "BENDR-eq": ("BENDR", True), "Comp50": ("Comp50", False), "Comp25": ("Comp25", False)}
NULL_MODELS = ["BP-eq", "LaBraM-eq", "CBraMod-eq", "BENDR-eq", "Comp50"]


def kseed(*p):
    return zlib.crc32("|".join(map(str, p)).encode()) & 0x7fffffff


def prep():
    """Patient-level matrix from the windows of the first session (26,968 windows)."""
    w = np.load(f"{DATA}/features_bandpower_windows.npz", allow_pickle=True); X, names = w["X"], list(w["names"]); e = np.load(f"{DATA}/emb_labram_windows.npz", allow_pickle=True); rec = e["rec"]
    first = ebf.first[rec]; subj = np.asarray(ebf.subj_r)[rec]; idx = {p: np.where(first & (subj == p))[0] for p in SUBJ}; assert sum(len(v) for v in idx.values()) == 26968
    out = {"BP": np.vstack([np.median(X[idx[p]], 0) for p in SUBJ]), "LaBraM": np.vstack([e["E"][idx[p]].mean(0) for p in SUBJ])}
    out["BENDR"] = np.vstack([np.load(f"{DATA}/emb_bendr_windows.npz", allow_pickle=True)["E"][idx[p]].mean(0) for p in SUBJ])
    c = np.load(f"{DATA}/emb_cbramod_layers_windows.npz", allow_pickle=True); Ec = c["E"]
    for lay in (3, 7, 11):
        out[f"CBraMod_L{lay + 1}"] = np.vstack([Ec[idx[p]][:, lay, :].mean(0) for p in SUBJ])
    cols = {(b, r): [names.index(f"rel_{b}_{ch}") for ch in chs] for r, chs in REGIONS.items() for b in BANDS}; med, iq = [], []
    for p in SUBJ:
        Xp = X[idx[p]]; R = np.column_stack([Xp[:, cols[(b, r)]].mean(1) for r in REGIONS for b in BANDS]); med.append(np.median(R, 0)); iq.append(np.subtract(*np.percentile(R, [75, 25], axis=0)))
    out["Comp25"] = np.vstack(med); out["Comp50"] = np.hstack([np.vstack(med), np.vstack(iq)])
    assert np.abs(out["BP"] - A.BP).max() < 1e-12
    np.savez(f"{OUT}/matrizes_pasien.npz", **out); print({k: v.shape for k, v in out.items()}, flush=True)


def load_mats():
    z = np.load(f"{OUT}/matrizes_pasien.npz"); return {k: z[k].astype(float) for k in z.files}


def rank_fit_apply(F, fit):
    Qs = np.quantile(F[fit], Q, axis=0); out = np.empty(F.shape)
    for j in range(F.shape[1]):
        out[:, j] = norm.ppf(np.clip(np.interp(F[:, j], Qs[:, j], Q), 1e-3, 1 - 1e-3))
    return out


def feats(X, fit, use_pca, use_as):
    """All fitting only on the `fit` patients: [standardization + PCA 50] -> rank-normal -> V4 (OLS on D, residual / training SD) -> [+ standardized age/sex]."""
    F = X.astype(float)
    if use_pca:
        sc = StandardScaler().fit(F[fit]); pca = PCA(n_components=min(NPC, len(fit) - 1), random_state=0).fit(sc.transform(F[fit])); F = pca.transform(sc.transform(F))
    F = rank_fit_apply(F, fit); B = np.linalg.lstsq(D[fit], F[fit], rcond=None)[0]; R = F - D @ B; sd = R[fit].std(0, ddof=np.linalg.matrix_rank(D[fit])); R = R / np.where(sd < 1e-8, 1.0, sd)
    if use_as:
        R = np.hstack([R, (AS - AS[fit].mean(0)) / AS[fit].std(0)])
    return R


def lr(C):
    return LogisticRegression(penalty="l2", C=C, class_weight="balanced", max_iter=5000)


def fit_predict(X, y, tr, te, use_pca, use_as, seed):
    """C is chosen from 6 values by 5-fold inner CV on the training patients; all steps are refitted inside every inner fold."""
    sc = {C: [] for C in CS}
    for a, b in StratifiedKFold(5, shuffle=True, random_state=seed).split(tr, y[tr]):
        ia, ib = tr[a], tr[b]; F = feats(X, ia, use_pca, use_as)
        for C in CS:
            sc[C].append(roc_auc_score(y[ib], lr(C).fit(F[ia], y[ia]).decision_function(F[ib])))
    best = sorted(CS, key=lambda C: (-np.mean(sc[C]), C))[0]; F = feats(X, tr, use_pca, use_as)
    return lr(best).fit(F[tr], y[tr]).predict_proba(F[te])[:, 1], best


def run_A():
    M = load_mats(); res = {}; t0 = time.time()
    for name, (key, pca) in MODELS.items():
        for use_as in (True, False):
            tag = f"{name}__{'AS' if use_as else 'noAS'}"
            out = Parallel(n_jobs=NJOBS)(delayed(fit_predict)(M[key], Y, J[f][2], J[f][3], pca, use_as, kseed("e43", f)) for f in range(len(J)))
            res[tag] = A.c17.to_matrix(J, [o[0] for o in out]); res[tag + "__C"] = np.array([o[1] for o in out])
            fa = A.c17.fold_aucs(J, res[tag]); print(f"{tag}: per-fold AUC {100 * fa.mean():.2f}; pooled {100 * np.mean([roc_auc_score(Y, p) for p in res[tag]]):.2f}; median chosen C {np.median(res[tag + '__C']):g} [{time.time() - t0:.0f}s]", flush=True)
            np.savez(f"{OUT}/A_prediksi.npz", **res)


def run_C(nrep=20):
    M = load_mats(); path = f"{OUT}/C_null_replikasi.csv"; done = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame(columns=["model", "scen", "rep", "fold_mean", "auc"]); t0 = time.time()
    for rep in range(nrep):
        for scen in ("H1", "H2a"):
            rng = np.random.default_rng(43000 + rep + (0 if scen == "H1" else 500))
            if scen == "H1":
                perm = rng.permutation(N)
            else:
                perm = np.arange(N)
                for e_ in np.unique(ERA):
                    ii = np.where(ERA == e_)[0]; perm[ii] = rng.permutation(ii)
            for name in NULL_MODELS:
                if ((done.model == name) & (done.scen == scen) & (done.rep == rep)).any():
                    continue
                key, pca = MODELS[name]; Xs = M[key][perm]
                out = Parallel(n_jobs=NJOBS)(delayed(fit_predict)(Xs, Y, J[f][2], J[f][3], pca, False, kseed("e43null", rep, f)) for f in range(5)); P = np.full(N, np.nan); fa = []
                for f in range(5):
                    P[J[f][3]] = out[f][0]; fa.append(roc_auc_score(Y[J[f][3]], out[f][0]))
                done = pd.concat([done, pd.DataFrame([dict(model=name, scen=scen, rep=rep, fold_mean=float(np.mean(fa)), auc=float(roc_auc_score(Y, P)))])], ignore_index=True); done.to_csv(path, index=False)
        print(f"rep {rep} [{time.time() - t0:.0f}s]", flush=True)


def checks():
    M = load_mats(); n = 0; r, k, tr, te = J[0]; X = M["Comp50"]; X2 = X.copy(); X2[te] = X2[te] * 5 + 3
    for use_pca, key in ((False, "Comp50"), (True, "LaBraM")):
        X = M[key]; X2 = X.copy(); X2[te] = X2[te] * 5 + 3; assert np.allclose(feats(X, tr, use_pca, True)[tr], feats(X2, tr, use_pca, True)[tr]); n += 1                                              # test data do not change the training features
        R = feats(X, tr, use_pca, False); assert np.abs(D[tr].T @ R[tr]).max() / len(tr) < 1e-8; n += 1                                                                                          # residual orthogonal to the design
        y2 = Y.copy(); y2[te] = 1 - y2[te]; a, _ = fit_predict(X, Y, tr, te, use_pca, True, 3); b, _ = fit_predict(X, y2, tr, te, use_pca, True, 3); assert np.allclose(a, b); n += 1                         # test labels are not used
    print(f"{n} checks passed", flush=True)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "--check"
    if mode == "--prep":
        prep()
    elif mode == "--check":
        checks()
    elif mode == "--run":
        run_A() if sys.argv[2] == "A" else run_C(int(sys.argv[3]) if len(sys.argv) > 3 else 20)
