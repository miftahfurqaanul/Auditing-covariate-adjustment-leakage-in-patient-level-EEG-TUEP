#!/usr/bin/env python3
"""Round 32 (tier 2): EQUAL tuning budget for all patient-level pipelines (CPU). See PRAREGISTRASI_putaran32_33.md.
Feature sources: BP (band power; I2/C1), frozen BENDR, frozen LaBraM, DNM. Adjustment: none or V4 (covariate OLS on all training patients). Age/sex are always appended.
ALL pipelines use the same search space (6 probe configurations) chosen by 5-fold inner CV within the training patients; V4 adjustment and PCA are refitted
inside every inner fold (no inner validation patient takes part in fitting a transform). The default configuration (LR-L2) is reported alongside.
Usage (CPU): CUDA_VISIBLE_DEVICES="" python round32_anggaran_setara.py --dry-run | --run"""
import os, sys, json, time, hashlib
os.environ["CUDA_VISIBLE_DEVICES"] = ""; os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ["PYTHONWARNINGS"] = "ignore"
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, "/home/miftah_f/RTX-5090/tuep_fresh/code")
import numpy as np, pandas as pd
from scipy import stats
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import round28_analisis as A, round29_sensitivitas as S
import eegbench_fixed_eval as ebf, evalkit as ek, round6_eval as r6

OUT = os.path.join(HERE, "hasil32"); os.makedirs(OUT, exist_ok=True)
Y, N, J, SUBJ, ERA, DEV, AS, D = A.Y, A.N, A.J, A.c17.SUBJ, A.ERA, A.DEV, A.c17.AS, A.D_I2
YD = np.char.add(np.char.add(ERA, "|"), DEV); NJOBS = 12
CONFIGS = ["LR-L2", "PCA16+LR", "PCA32+LR", "PCA64+LR", "LR-L1", "RF"]        # urutan = prioritas bila seri (default pertama)
CS_L1 = [0.01, 0.03, 0.1, 0.3, 1.0]


def sources():
    z = np.load(f"{ek.D}/eegbench_rep_D.npz"); rec = z["rec"]; s = ebf.subj_r[rec]; fi = ebf.first[rec]; M = {"BP": A.BP.astype(np.float64)}
    for name, key in (("BENDR", "bendr"), ("LaBraM", "labram")):
        X = z[key].astype(np.float32); M[name] = np.vstack([X[fi & (s == p)].mean(0) for p in SUBJ])
    assert (r6.r4.SUBJ == SUBJ).all(); M["DNM"] = r6.XD.astype(np.float64)
    return M


def adjust(X, fit):
    """V4 (identical to round29 V4 and round31): OLS on [intercept, age, sex, era indicator] fitted on the `fit` patients; residuals divided by the residual SD."""
    X = X.astype(np.float64); B = np.linalg.lstsq(D[fit], X[fit], rcond=None)[0]; R = X - D @ B
    sd = R[fit].std(0, ddof=np.linalg.matrix_rank(D[fit])); return R / np.where(sd < 1e-8, 1.0, sd)


def feats(X, adj, fit):
    """Features of all patients; all fitting only on `fit`. Dimension > 2000 (BENDR): StandardScaler + PCA 128."""
    F = adjust(X, fit) if adj else X.astype(np.float64)
    if F.shape[1] > 2000:
        sc = StandardScaler().fit(F[fit]); pca = PCA(n_components=min(128, len(fit) - 1), random_state=0).fit(sc.transform(F[fit])); F = pca.transform(sc.transform(F))
    return F


def fit_predict(cfg, G, fit, pred, seed):
    H = G
    if cfg.startswith("PCA"):
        k = min(int(cfg[3:cfg.index("+")]), G.shape[1], len(fit) - 1)
        sc = StandardScaler().fit(G[fit]); pca = PCA(n_components=k, random_state=0).fit(sc.transform(G[fit])); H = pca.transform(sc.transform(G))
    H = np.hstack([H, AS])
    if cfg == "RF":
        m = RandomForestClassifier(n_estimators=300, min_samples_leaf=3, class_weight="balanced_subsample", random_state=seed, n_jobs=1)
    elif cfg == "LR-L1":
        m = make_pipeline(StandardScaler(), LogisticRegressionCV(Cs=CS_L1, cv=5, scoring="roc_auc", penalty="l1", solver="liblinear", class_weight="balanced", max_iter=5000, random_state=seed))
    else:
        m = ek.nested_lr()
    return m.fit(H[fit], Y[fit]).predict_proba(H[pred])[:, 1]


def pipeline_fold(X, adj, tr, te, seed):
    sc = np.zeros((5, len(CONFIGS)))
    for i, (a, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(tr, Y[tr])):
        fit, val = tr[a], tr[b]; G = feats(X, adj, fit)                                  # transforms are refitted inside the inner fold
        for j, c in enumerate(CONFIGS):
            sc[i, j] = roc_auc_score(Y[val], fit_predict(c, G, fit, val, seed))
    best = int(np.argmax(sc.mean(0))); G = feats(X, adj, tr); dflt = fit_predict("LR-L2", G, tr, te, seed)
    return dict(tuned=dflt if best == 0 else fit_predict(CONFIGS[best], G, tr, te, seed), default=dflt, best=best, inner=sc.mean(0))


def fold(f, SRC, names=None):
    r, k, tr, te = J[f]; seed = 100 * r + k; out = {}
    for src, X in SRC.items():
        for adj in (False, True):
            nm = f"{src}|{'V4' if adj else 'none'}"
            if names is None or nm in names:
                out[nm] = pipeline_fold(X, adj, tr, te, seed)
    return out


def nb(a, n1=160, n2=40):
    m = a.mean(); h = stats.t.ppf(0.975, len(a) - 1) * a.std(ddof=1) * np.sqrt(1 / len(a) + n2 / n1); return 100 * m, 100 * (m - h), 100 * (m + h)


def perturbation_checks(SRC):
    """For BP|V4 and BENDR|V4 (folds 0 and 17): (i) changing the X of test patients changes neither the chosen configuration nor the inner validation score; (ii) flipping the test labels changes nothing (selection, tuned and default predictions)."""
    n = 0
    for nm in ("BP|V4", "BENDR|V4"):
        src = nm.split("|")[0]
        for f in (0, 17):
            _, _, tr, te = J[f]; a = fold(f, {src: SRC[src]}, names=[nm])[nm]
            S2 = {src: SRC[src].copy()}; S2[src][te] = S2[src][te] * 5 + 3; b = fold(f, S2, names=[nm])[nm]
            y0 = Y.copy(); Y[te] = 1 - Y[te]; c = fold(f, {src: SRC[src]}, names=[nm])[nm]; Y[:] = y0
            assert b["best"] == a["best"] and np.allclose(b["inner"], a["inner"], atol=1e-12), (nm, f, "test X changes the selection")   # the test X must not change selection/training
            assert c["best"] == a["best"] and np.allclose(c["inner"], a["inner"], atol=1e-12) and np.allclose(c["tuned"], a["tuned"], atol=1e-8) and np.allclose(c["default"], a["default"], atol=1e-8), (nm, f, "test label")
            n += 2
    return n


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "--dry-run"; t0 = time.time(); SRC = sources()
    print("fitur:", {k: v.shape for k, v in SRC.items()}, "| konfigurasi:", CONFIGS, flush=True); assert all(np.isfinite(v).all() for v in SRC.values())
    print(f"{perturbation_checks(SRC)} perturbation checks passed ({time.time() - t0:.0f}s)", flush=True)
    if mode == "--dry-run":
        o = fold(0, SRC); print({k: (CONFIGS[v["best"]], round(roc_auc_score(Y[J[0][3]], v["tuned"]), 3), round(roc_auc_score(Y[J[0][3]], v["default"]), 3)) for k, v in o.items()})
        print(f"dry-run done {time.time() - t0:.0f}s"); sys.exit()
    res = Parallel(n_jobs=NJOBS)(delayed(fold)(f, SRC) for f in range(len(J))); names = list(res[0].keys()); P, chosen = {}, []
    for nm in names:
        P[f"{nm}|tuned"] = A.c17.to_matrix(J, [o[nm]["tuned"] for o in res]); P[f"{nm}|default"] = A.c17.to_matrix(J, [o[nm]["default"] for o in res])
        chosen += [dict(pipeline=nm, fold=f, config=CONFIGS[o[nm]["best"]], **{f"inner_{c}": o[nm]["inner"][j] for j, c in enumerate(CONFIGS)}) for f, o in enumerate(res)]
    pd.DataFrame(chosen).to_csv(f"{OUT}/anggaran_konfigurasi_terpilih.csv", index=False)
    print(pd.DataFrame(chosen).groupby(["pipeline", "config"]).size().unstack(fill_value=0).to_string(), flush=True)
    EV = np.load(os.path.join(HERE, "hasil30", "eval_prediksi_v4.npz")); REF = {k: EV[k] for k in ["C1", "I2_adj", "F2_adj", "Floor", "LaBraM_FT", "CBraMod_FT"]}
    ALL = {**P, **REF}; FA = {k: A.c17.fold_aucs(J, m) for k, m in ALL.items()}
    # reproduce the default configuration against the existing results
    for mine, ref in (("BP|V4|default", "I2_adj"), ("BP|none|default", "C1")):
        d = np.abs(FA[mine] - FA[ref]).max(); print(f"reproduction {mine} vs {ref}: max per-fold AUC difference {d:.2e}", flush=True); assert d < 5e-3, (mine, d)
    z7 = np.load("/home/miftah_f/RTX-5090/tuep_fresh/reports/round7_fold_aucs.npz", allow_pickle=True); d = np.abs(FA["DNM|none|default"] - z7["DNM-F"]).max()
    print(f"reproduction DNM|none|default vs DNM-F round7: max difference {d:.2e}", flush=True); assert d < 5e-3, d
    rows = [dict(model=k, auc_fold=nb(a)[0], lo=nb(a)[1], hi=nb(a)[2], sd=100 * a.std(ddof=1), auc_pooled=100 * A.c17.pooled_auc(ALL[k]), strata_year_x_device=100 * A.r9.strat_auc(Y, ALL[k].mean(0), YD)) for k, a in FA.items()]
    M1 = pd.DataFrame(rows); M1.to_csv(f"{OUT}/anggaran_metrik.csv", index=False); print(M1.round(2).to_string(index=False), flush=True)
    pairs = [("BP|V4|tuned", "BP|none|tuned", "primary"), ("BP|V4|tuned", "BENDR|V4|tuned", "primary"), ("BP|V4|tuned", "LaBraM|V4|tuned", "primary"), ("BP|V4|tuned", "DNM|V4|tuned", "primary")]
    pairs += [(f"{s}|V4|tuned", f"{s}|none|tuned", "descriptive") for s in ("BP", "BENDR", "LaBraM", "DNM")]
    pairs += [(f"{s}|{a}|tuned", f"{s}|{a}|default", "descriptive") for s in ("BP", "BENDR", "LaBraM", "DNM") for a in ("none", "V4")]
    pairs += [("BP|V4|tuned", "BP|V4|default", "descriptive")] * 0 + [("BP|V4|tuned", "I2_adj", "descriptive"), ("BP|V4|tuned", "LaBraM_FT", "descriptive"), ("BP|V4|tuned", "Floor", "descriptive")]
    rows = []
    for a, b, fam in pairs:
        d, lo, hi, p = S.nb_diff(FA[a], FA[b]); rows.append(dict(test=f"{a} - {b}", diff=d, lo=lo, hi=hi, p=p, folds_better=int((FA[a] > FA[b]).sum()), family=fam))
    T = pd.DataFrame(rows); pm = T.family == "primary"; T["p_holm"] = np.nan; T.loc[pm, "p_holm"] = A.holm(T.loc[pm, "p"].values); T["q_bh"] = stats.false_discovery_control(T.p.values)
    T.to_csv(f"{OUT}/anggaran_selisih.csv", index=False); print(T.round(4).to_string(index=False), flush=True)
    np.savez(f"{OUT}/anggaran_prediksi.npz", **{k.replace("|", "_"): v for k, v in P.items()})
    json.dump(dict(code_sha256=hashlib.sha256(open(__file__, "rb").read()).hexdigest(), seconds=time.time() - t0), open(f"{OUT}/meta_anggaran.json", "w"))
    print(f"done {time.time() - t0:.0f}s")
