#!/usr/bin/env python3
"""Round 33 (tier 4): published comparators DNM and Rohira with V4 covariate adjustment (CPU). See PRAREGISTRASI_putaran32_33.md.
DNM: patient-level dynamic-network-model features (57-d) -> V4 (covariate OLS on all training patients) + age/sex -> nested_lr (same as DNM-F of round7, only V4 added).
Rohira: random forest on band power per 10 s window (set A: 8 channels x 6 bands = 48 features; Rohira-F recipe of round7: RF with 100 trees, trained on 80% of the training patients, patient score = mean
of the window probabilities). V4 is applied at window level (OLS of 48 features on the window-owner design, fitted on ALL windows of training patients; like NM-V4 on segments). Age/sex
(optional) are appended as per-window columns. Tang-F is NOT included (neural network, GPU).
Usage (CPU): CUDA_VISIBLE_DEVICES="" python round33_pembanding_v4.py --dry-run | --run [n_null_DNM=20] [n_null_Rohira=10]"""
import os, sys, json, time, hashlib
os.environ["CUDA_VISIBLE_DEVICES"] = ""; os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ["PYTHONWARNINGS"] = "ignore"
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, "/home/miftah_f/RTX-5090/tuep_fresh/code")
import numpy as np, pandas as pd
from scipy import stats
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import round28_analisis as A, round29_sensitivitas as S
import evalkit as ek, round6_eval as r6, round7 as r7

OUT = os.path.join(HERE, "hasil33"); os.makedirs(OUT, exist_ok=True)
Y, N, J, SUBJ, ERA, DEV, AS, D = A.Y, A.N, A.J, A.c17.SUBJ, A.ERA, A.DEV, A.c17.AS, A.D_I2
YD = np.char.add(np.char.add(ERA, "|"), DEV); NJOBS = 12; POS = {s: i for i, s in enumerate(SUBJ)}
assert (r6.r4.SUBJ == SUBJ).all() and (r7.SUBJ == SUBJ).all() and np.array_equal(r7.J200[0][3], J[0][3])
XD = r6.XD.astype(np.float64); XR, WS, WY, YLAB = r7.rohira_inputs(); PIDX = np.array([POS[s] for s in WS]); assert np.array_equal(WY, Y[PIDX])


def adjust(X, fit, owner=None):
    """V4: OLS on [intercept, age, sex, era indicator] fitted on the rows of the `fit` patients (owner = owner of each row; None if row = patient)."""
    X = X.astype(np.float64); Dm = D if owner is None else D[owner]; rows = np.isin(np.arange(len(X)) if owner is None else owner, fit)
    B = np.linalg.lstsq(Dm[rows], X[rows], rcond=None)[0]; R = X - Dm @ B; sd = R[rows].std(0, ddof=np.linalg.matrix_rank(Dm[rows])); return R / np.where(sd < 1e-8, 1.0, sd)


def split_val(tr_p, ylab, seed):
    """identical to tang_fixed.split_val (20% stratified validation from the training patients)."""
    g = np.random.default_rng(seed); va = []
    for c in [0, 1]:
        ps = np.array(sorted(p for p in tr_p if ylab[p] == c)); va += list(g.permutation(ps)[: int(round(0.2 * len(ps)))])
    return sorted(set(tr_p) - set(va)), sorted(va)


def dnm_fold(X, adj, tr, te, use_as=True):
    F = adjust(X, tr) if adj else X
    if use_as:
        F = np.hstack([F, AS])
    return ek.nested_lr().fit(F[tr], Y[tr]).predict_proba(F[te])[:, 1]


def rohira_fold(Xw, pidx, adj, use_as, r, k, tr, te):
    """As r7.rohira_job; pidx = owner of each window (patient index)."""
    tr_p, _ = split_val(list(SUBJ[tr]), YLAB, 100 * r + k); Z = Xw
    if adj:
        Z = adjust(Xw, tr, owner=pidx)
    if use_as:
        Z = np.hstack([Z, AS[pidx]])
    fit = np.isin(SUBJ[pidx], tr_p); m = RandomForestClassifier(n_estimators=100, max_depth=1000, class_weight="balanced", n_jobs=1, random_state=1000 * r + k)
    m.fit(Z[fit], Y[pidx][fit]); sel = np.isin(pidx, te)
    return pd.Series(m.predict_proba(Z[sel])[:, 1]).groupby(pidx[sel]).mean().loc[te].values


def perturbation_checks():
    n = 0
    for f in (0, 17):
        r, k, tr, te = J[f]
        X2 = XD.copy(); X2[te] = X2[te] * 5 + 3; assert np.allclose(adjust(XD, tr)[tr], adjust(X2, tr)[tr]); n += 1
        a = dnm_fold(XD, True, tr, te); y0 = Y.copy(); Y[te] = 1 - Y[te]; b = dnm_fold(XD, True, tr, te); Y[:] = y0; assert np.allclose(a, b, atol=1e-8); n += 1
        R0 = adjust(XD, tr); assert np.abs(np.linalg.lstsq(D[tr], R0[tr], rcond=None)[0]).max() < 1e-8; n += 1
        Xw2 = XR.copy(); mt = np.isin(PIDX, te); Xw2[mt] = Xw2[mt] * 5 + 3; assert np.allclose(adjust(XR, tr, PIDX)[~mt], adjust(Xw2, tr, PIDX)[~mt]); n += 1
        R1 = adjust(XR, tr, PIDX); rows = np.isin(PIDX, tr); assert np.abs(np.linalg.lstsq(D[PIDX][rows], R1[rows], rcond=None)[0]).max() < 1e-6; n += 1
        a = rohira_fold(XR, PIDX, True, True, r, k, tr, te); Y[te] = 1 - Y[te]; b = rohira_fold(XR, PIDX, True, True, r, k, tr, te); Y[:] = y0
        assert np.allclose(a, b, atol=1e-10); n += 1                                  # test labels flipped: Y[pidx][fit] uses only training patients
    return n


def nb(a, n1=160, n2=40):
    m = a.mean(); h = stats.t.ppf(0.975, len(a) - 1) * a.std(ddof=1) * np.sqrt(1 / len(a) + n2 / n1); return 100 * m, 100 * (m - h), 100 * (m + h)


def pooled(Mx):
    return float(np.mean([roc_auc_score(Y, p) for p in Mx]))


def perm_owner(scen, seed):
    rng = np.random.default_rng(seed)
    if scen == "H1":
        return rng.permutation(N)
    perm = np.arange(N)
    for e in np.unique(ERA):
        ii = np.where(ERA == e)[0]; perm[ii] = rng.permutation(ii)
    return perm


def null_dnm(scen, seed, adj):
    perm = perm_owner(scen, seed); X = XD[perm]
    return pooled(A.c17.to_matrix(J, Parallel(n_jobs=NJOBS)(delayed(dnm_fold)(X, adj, J[f][2], J[f][3], False) for f in range(len(J)))))


def null_rohira(scen, seed, adj):
    """the windows of patient perm[p] are given to patient p (the new owner carries its own label/covariates/era); without age/sex."""
    perm = perm_owner(scen, seed); blocks = [np.where(PIDX == q)[0] for q in range(N)]
    Xn = np.vstack([XR[blocks[perm[p]]] for p in range(N)]); pn = np.concatenate([np.full(len(blocks[perm[p]]), p) for p in range(N)])
    return pooled(A.c17.to_matrix(J, Parallel(n_jobs=NJOBS)(delayed(rohira_fold)(Xn, pn, adj, False, J[f][0], J[f][1], J[f][2], J[f][3]) for f in range(len(J)))))


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "--dry-run"; nn_d = int(sys.argv[2]) if len(sys.argv) > 2 else 20; nn_r = int(sys.argv[3]) if len(sys.argv) > 3 else 10; t0 = time.time()
    print(f"DNM {XD.shape}; Rohira windows {XR.shape} from {len(np.unique(PIDX))} patients; design {D.shape}", flush=True); assert np.isfinite(XD).all() and np.isfinite(XR).all()
    print(f"{perturbation_checks()} perturbation checks passed ({time.time() - t0:.0f}s)", flush=True)
    if mode == "--dry-run":
        r, k, tr, te = J[0]
        print("DNM-F / DNM-V4 fold 0:", [round(roc_auc_score(Y[te], dnm_fold(XD, a, tr, te)), 3) for a in (False, True)])
        print("Rohira-F, +AS, V4, V4+AS fold 0:", [round(roc_auc_score(Y[te], rohira_fold(XR, PIDX, a, s, r, k, tr, te)), 3) for a, s in ((False, False), (False, True), (True, False), (True, True))])
        print(f"dry-run done {time.time() - t0:.0f}s"); sys.exit()
    P = {}
    P["DNM-F"] = A.c17.to_matrix(J, Parallel(n_jobs=NJOBS)(delayed(dnm_fold)(XD, False, J[f][2], J[f][3]) for f in range(len(J))))
    P["DNM-V4"] = A.c17.to_matrix(J, Parallel(n_jobs=NJOBS)(delayed(dnm_fold)(XD, True, J[f][2], J[f][3]) for f in range(len(J))))
    for nm, a, s in (("Rohira-F", False, False), ("Rohira+AS", False, True), ("Rohira-V4", True, False), ("Rohira-V4+AS", True, True)):
        P[nm] = A.c17.to_matrix(J, Parallel(n_jobs=NJOBS)(delayed(rohira_fold)(XR, PIDX, a, s, J[f][0], J[f][1], J[f][2], J[f][3]) for f in range(len(J)))); print(nm, "done", f"{time.time() - t0:.0f}s", flush=True)
    EV = np.load(os.path.join(HERE, "hasil30", "eval_prediksi_v4.npz")); REF = {k: EV[k] for k in ["C1", "I2_adj", "F2_adj", "Floor", "LaBraM_FT", "CBraMod_FT"]}
    ALL = {**P, **REF}; FA = {k: A.c17.fold_aucs(J, m) for k, m in ALL.items()}
    z7 = np.load("/home/miftah_f/RTX-5090/tuep_fresh/reports/round7_fold_aucs.npz", allow_pickle=True)
    for mine, ref in (("DNM-F", "DNM-F"), ("Rohira-F", "Rohira-F")):
        d = np.abs(FA[mine] - z7[ref]).max(); print(f"reproduction {mine} vs round7: max per-fold AUC difference {d:.2e}", flush=True); assert d < 5e-3, (mine, d)
    rows = [dict(model=k, auc_fold=nb(a)[0], lo=nb(a)[1], hi=nb(a)[2], sd=100 * a.std(ddof=1), auc_pooled=100 * A.c17.pooled_auc(ALL[k]), strata_year_x_device=100 * A.r9.strat_auc(Y, ALL[k].mean(0), YD)) for k, a in FA.items()]
    M1 = pd.DataFrame(rows); M1.to_csv(f"{OUT}/pembanding_metrik.csv", index=False); print(M1.round(2).to_string(index=False), flush=True)
    pairs = [("DNM-V4", "DNM-F", "primary"), ("Rohira-V4+AS", "Rohira+AS", "primary"), ("I2_adj", "DNM-V4", "primary"), ("I2_adj", "Rohira-V4+AS", "primary"),
             ("Rohira-V4", "Rohira-F", "descriptive"), ("Rohira+AS", "Rohira-F", "descriptive"), ("DNM-V4", "C1", "descriptive"), ("Rohira-V4+AS", "C1", "descriptive"),
             ("I2_adj", "DNM-F", "descriptive"), ("I2_adj", "Rohira-F", "descriptive"), ("DNM-V4", "Floor", "descriptive"), ("Rohira-V4+AS", "Floor", "descriptive")]
    rows = []
    for a, b, fam in pairs:
        d, lo, hi, p = S.nb_diff(FA[a], FA[b]); rows.append(dict(test=f"{a} - {b}", diff=d, lo=lo, hi=hi, p=p, folds_better=int((FA[a] > FA[b]).sum()), family=fam))
    T = pd.DataFrame(rows); pm = T.family == "primary"; T["p_holm"] = np.nan; T.loc[pm, "p_holm"] = A.holm(T.loc[pm, "p"].values); T["q_bh"] = stats.false_discovery_control(T.p.values)
    T.to_csv(f"{OUT}/pembanding_selisih.csv", index=False); print(T.round(4).to_string(index=False), flush=True)
    np.savez(f"{OUT}/pembanding_prediksi.npz", **{k.replace("+", "_AS_").replace("-", "_"): v for k, v in P.items()})
    rows = []
    for name, fn, n in (("DNM", null_dnm, nn_d), ("Rohira", null_rohira, nn_r)):
        for scen in (("H1", "H2a") if n > 0 else ()):
            for adj, v in ((False, "mentah"), (True, "V4")):
                vals = np.array([fn(scen, 33000 + r, adj) for r in range(n)]); rows.append(dict(pembanding=name, scenario=scen, variant=v, n=n, mean=vals.mean(), sd=vals.std(ddof=1), p95=np.percentile(vals, 95))); print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(f"{OUT}/pembanding_null.csv", index=False)
    json.dump(dict(code_sha256=hashlib.sha256(open(__file__, "rb").read()).hexdigest(), seconds=time.time() - t0), open(f"{OUT}/meta_pembanding.json", "w"))
    print(f"done {time.time() - t0:.0f}s")
