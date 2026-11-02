#!/usr/bin/env python3
"""Round 31: FROZEN BENDR and LaBraM embeddings with V4 covariate adjustment (OLS on all training patients), nested_lr probe as for I2.
No GPU, no fine-tuning. Per-recording embeddings of the first session from tuep_fresh/data/eegbench_rep_D.npz (read-only); patient = mean over recordings (as in round7).
Variants per encoder: raw (without age/sex), raw+AS, V4, V4+AS. Protocol: 50 canonical folds, per-fold AUC (Nadeau-Bengio CI).
Usage (CPU): CUDA_VISIBLE_DEVICES="" python round31_fm_beku.py --dry-run | --run [n_null=20]"""
import os, sys, json, time, hashlib
os.environ["CUDA_VISIBLE_DEVICES"] = ""; os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ["PYTHONWARNINGS"] = "ignore"
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, "/home/miftah_f/RTX-5090/tuep_fresh/code")
import numpy as np, pandas as pd
from scipy import stats
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import round28_analisis as A, round29_sensitivitas as S
import eegbench_fixed_eval as ebf, evalkit as ek

OUT = os.path.join(HERE, "hasil31"); os.makedirs(OUT, exist_ok=True)
Y, N, J, SUBJ, ERA, DEV, AS, D = A.Y, A.N, A.J, A.c17.SUBJ, A.ERA, A.DEV, A.c17.AS, A.D_I2
YD = np.char.add(np.char.add(ERA, "|"), DEV); NJOBS = 12
ENC = {"BENDR": "bendr", "LaBraM": "labram"}; VAR = ["mentah", "mentah+AS", "V4", "V4+AS"]


def patient_matrices():
    z = np.load(f"{ek.D}/eegbench_rep_D.npz"); rec = z["rec"]; s = ebf.subj_r[rec]; fi = ebf.first[rec]; M = {}
    for name, key in ENC.items():
        X = z[key].astype(np.float32); M[name] = np.vstack([X[fi & (s == p)].mean(0) for p in SUBJ])
    return M


def adjust(X, tr):
    """V4: OLS regression on [intercept, age, sex, era indicator] fitted on ALL training patients; residuals divided by the training residual SD (as in round29 V4)."""
    X = X.astype(np.float64); B = np.linalg.lstsq(D[tr], X[tr], rcond=None)[0]; R = X - D @ B
    sd = R[tr].std(0, ddof=np.linalg.matrix_rank(D[tr])); return R / np.where(sd < 1e-8, 1.0, sd)


def probe(F, tr, te, use_as):
    """If the dimension is > 2000 (BENDR 8,704): StandardScaler + PCA 128 fitted on training patients (as ebf.lr_fp); then age/sex (optional) are appended; nested_lr."""
    if F.shape[1] > 2000:
        sc = StandardScaler().fit(F[tr]); pca = PCA(n_components=min(128, len(tr) - 1), random_state=0).fit(sc.transform(F[tr])); G = pca.transform(sc.transform(F))
    else:
        G = F
    if use_as:
        G = np.hstack([G, AS])
    return ek.nested_lr().fit(G[tr], Y[tr]).predict_proba(G[te])[:, 1]


def fold(X, tr, te, variants):
    out = {}
    for v in variants:
        F = adjust(X, tr) if v.startswith("V4") else X
        out[v] = probe(F, tr, te, v.endswith("+AS"))
    return out


def matrices(X, variants, folds=None):
    idx = range(len(J)) if folds is None else folds
    res = Parallel(n_jobs=NJOBS)(delayed(fold)(X, J[f][2], J[f][3], variants) for f in idx)
    return {v: A.c17.to_matrix([J[f] for f in idx], [o[v] for o in res]) for v in variants} if folds is None else res


def pooled(Mx, y=None):
    return float(np.mean([roc_auc_score(Y if y is None else y, p) for p in Mx]))


def perturbation_checks(X):
    """V4 must not use the data/labels of test patients: (i) changing the test X does not change the training features; (ii) flipping the test labels does not change the predictions."""

    n = 0
    for f in (0, 17, 33):
        _, _, tr, te = J[f]; X2 = X.copy(); X2[te] = X2[te] * 5 + 3
        assert np.allclose(adjust(X, tr)[tr], adjust(X2, tr)[tr]), f; n += 1
        a = fold(X, tr, te, ["V4+AS"])["V4+AS"]; y0 = Y.copy(); Y[te] = 1 - Y[te]; b = fold(X, tr, te, ["V4+AS"])["V4+AS"]; Y[:] = y0
        assert np.allclose(a, b), f; n += 1
        R = adjust(X, tr); assert np.abs(np.linalg.lstsq(D[tr], R[tr], rcond=None)[0]).max() < 1e-6, f; n += 1
    return n


def null_rep(X, scen, seed, variant):
    rng = np.random.default_rng(seed)
    if scen == "H1":
        perm = rng.permutation(N)
    else:
        perm = np.arange(N)
        for e in np.unique(ERA):
            ii = np.where(ERA == e)[0]; perm[ii] = rng.permutation(ii)
    return pooled(matrices(X[perm], [variant])[variant])


def nb(a, n1=160, n2=40):
    m = a.mean(); h = stats.t.ppf(0.975, len(a) - 1) * a.std(ddof=1) * np.sqrt(1 / len(a) + n2 / n1); return 100 * m, 100 * (m - h), 100 * (m + h)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "--dry-run"; n_null = int(sys.argv[2]) if len(sys.argv) > 2 else 20; t0 = time.time()
    PM = patient_matrices(); print("patient matrices:", {k: v.shape for k, v in PM.items()}, "| design", D.shape, "rank", np.linalg.matrix_rank(D), flush=True)
    assert all(np.isfinite(v).all() for v in PM.values())
    for nm, X in PM.items():
        print(f"{nm}: {perturbation_checks(X)} perturbation checks passed", flush=True)
    if mode == "--dry-run":
        for nm, X in PM.items():
            r = matrices(X, VAR, folds=[0, 1, 2]); print(nm, "first 3 folds, test AUC:", {v: [round(roc_auc_score(Y[J[f][3]], r[i][v]), 3) for i, f in enumerate([0, 1, 2])] for v in VAR})
        print(f"dry-run done {time.time() - t0:.0f}s"); sys.exit()
    # ---------- data asli
    P = {}
    for nm, X in PM.items():
        for v, M in matrices(X, VAR).items():
            P[f"{nm}|{v}"] = M
        # reproduksi pembanding round7 (LaBraM 0.640; BENDR 0.553)
        ref = A.c17.to_matrix(J, Parallel(n_jobs=NJOBS)(delayed(lambda tr, te: ebf.lr_fp(X[tr], Y[tr], X[te]))(tr, te) for _, _, tr, te in J)); P[f"{nm}|round7"] = ref
        want = {"BENDR": 0.553, "LaBraM": 0.640}[nm]; got = A.c17.pooled_auc(ref); print(f"reproduksi round7 {nm}: AUC gabungan {got:.4f} (target {want})", flush=True); assert abs(got - want) < 6e-4, (nm, got)
    EV = np.load(os.path.join(HERE, "hasil30", "eval_prediksi_v4.npz")); REF = {k: EV[k] for k in ["C1", "I2_adj", "F2_adj", "Floor", "LaBraM_FT", "CBraMod_FT"]}
    ALL = {**P, **REF}; FA = {k: A.c17.fold_aucs(J, m) for k, m in ALL.items()}; rows = []
    for k, m in ALL.items():
        a = FA[k]; mn, lo, hi = nb(a); rows.append(dict(model=k, auc_fold=mn, lo=lo, hi=hi, sd=100 * a.std(ddof=1), auc_pooled=100 * A.c17.pooled_auc(m),
                                                         strata_year_x_device=100 * A.r9.strat_auc(Y, m.mean(0), YD)))
    M1 = pd.DataFrame(rows); M1.to_csv(f"{OUT}/fm_beku_metrik.csv", index=False); print(M1.round(2).to_string(index=False), flush=True)
    pairs = []
    for nm in ENC:
        pairs += [(f"{nm}|V4+AS", f"{nm}|mentah+AS", "primary"), (f"{nm}|V4", f"{nm}|mentah", "descriptive"), (f"{nm}|V4+AS", f"{nm}|round7", "descriptive"),
                  (f"{nm}|V4+AS", "C1", "descriptive"), (f"{nm}|V4+AS", "I2_adj", "descriptive"), (f"{nm}|V4+AS", "LaBraM_FT", "descriptive"), (f"{nm}|V4+AS", "Floor", "descriptive")]
    rows = []
    for a, b, fam in pairs:
        d, lo, hi, p = S.nb_diff(FA[a], FA[b]); rows.append(dict(test=f"{a} - {b}", diff=d, lo=lo, hi=hi, p=p, folds_better=int((FA[a] > FA[b]).sum()), family=fam))
    T = pd.DataFrame(rows); pm = T.family == "primary"; T["p_holm"] = np.nan; T.loc[pm, "p_holm"] = A.holm(T.loc[pm, "p"].values)
    T["q_bh"] = stats.false_discovery_control(T.p.values); T.to_csv(f"{OUT}/fm_beku_selisih.csv", index=False); print(T.round(4).to_string(index=False), flush=True)
    np.savez(f"{OUT}/fm_beku_prediksi.npz", **{k.replace("|", "_"): v for k, v in P.items()})
    # ---------- leakage test: embedding features permuted between patients (without age/sex); H1 permuted globally, H2a permuted within era
    if n_null > 0:
        rows = []
        for nm, X in PM.items():
            for scen in ("H1", "H2a"):
                for v in ("mentah", "V4"):
                    vals = np.array(Parallel(n_jobs=1)(delayed(null_rep)(X, scen, 31000 + r, v) for r in range(n_null)))
                    rows.append(dict(encoder=nm, scenario=scen, variant=v, n=n_null, mean=vals.mean(), sd=vals.std(ddof=1), p95=np.percentile(vals, 95))); print(rows[-1], flush=True)
        pd.DataFrame(rows).to_csv(f"{OUT}/fm_beku_null.csv", index=False)
    json.dump(dict(code_sha256=hashlib.sha256(open(__file__, "rb").read()).hexdigest(), seconds=time.time() - t0), open(f"{OUT}/meta_fm_beku.json", "w"))
    print(f"done {time.time() - t0:.0f}s")
