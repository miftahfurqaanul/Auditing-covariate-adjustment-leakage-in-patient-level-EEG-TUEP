#!/usr/bin/env python3
"""Round 38 (GPU): paired null test of NM: N0 (standard normalization without covariate regression) vs V4 (regression on the segments of all training patients) vs V4p (coefficients from the per-patient mean of segments, unweighted; optional) on permuted segments\n(H1 global; H2a within era). See PRAREGISTRASI_putaran37_38.md. One GPU process at a time.
Usage: python round38_nm_paired.py --null [n=20] [variants separated by commas; default N0,V4]"""
import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); CODE = "/home/miftah_f/RTX-5090/tuep_fresh/code"; sys.path.insert(0, HERE); sys.path.insert(0, CODE)
import numpy as np, pandas as pd, torch
from scipy import stats
from sklearn.metrics import roc_auc_score
import round30_nm as M
nm, tf, r4 = M.nm, M.tf, M.r4
XSEG, PID, Y, AS, DES, N, POS, JOBS, SUBJ = M.XSEG, M.PID, M.Y, M.AS, M.DES, M.N, M.POS, M.JOBS, M.SUBJ
ERA = np.asarray(r4.YEAR_G); OUT = M.OUT


def tokens_v(X, pid, tr, variant):
    if variant == "V4":
        return M.tokens(X, pid, tr, "V4", no_as=True)
    fm = np.isin(pid, tr)
    if variant == "N0":
        mu = X[fm].mean(0); sd = X[fm].std(0, ddof=1); T = (X - mu) / sd
    else:                                                                    # V4p: OLS on the per-patient mean of the segments (unweighted), then residuals divided by the training segment SD
        cnt = np.bincount(pid, minlength=N); S = np.zeros((N, X.shape[1])); np.add.at(S, pid, X.astype(np.float64)); Mn = S / np.maximum(cnt, 1)[:, None]
        B = np.linalg.lstsq(DES[tr], Mn[tr], rcond=None)[0]; R = X - DES[pid] @ B; sd = R[fm].std(0, ddof=np.linalg.matrix_rank(DES[tr])); T = R / sd
    toks = [T[pid == i].astype(np.float32) for i in range(N)]
    summ = np.column_stack([np.vstack([np.median(t, 0) for t in toks]), (AS - AS[tr].mean(0)) / AS[tr].std(0)]).astype(np.float32); summ[:, 114:] = 0.0
    return toks, summ


def null_fold(X, pid, r, k, tr, te, variant, max_epochs=20):
    ylab = {s: int(Y[POS[s]]) for s in SUBJ}; tr_p, va_p = tf.split_val(list(SUBJ[tr]), ylab, 100 * r + k)
    tr_i, va_i = np.array([POS[s] for s in tr_p]), np.array([POS[s] for s in va_p]); toks, summ = tokens_v(X, pid, tr, variant)
    m, va, ep = nm.train_one(toks, summ, tr_i, va_i, 32, 32, 1000 * r + 10 * k, True, max_epochs); return nm.predict(m, toks, summ, te, 32)


def shuffled(scen, seed):
    rng = np.random.default_rng(seed)
    if scen == "H1":
        pi = rng.permutation(N)
    else:
        pi = np.arange(N)
        for e in np.unique(ERA):
            ii = np.where(ERA == e)[0]; pi[ii] = rng.permutation(ii)
    return np.vstack([XSEG[PID == pi[i]] for i in range(N)]), np.concatenate([np.full(int((PID == pi[i]).sum()), i) for i in range(N)])


def decision(d):
    se = d.std(ddof=1) / np.sqrt(len(d)); m = d.mean(); return "tidak ada bukti kebocoran" if (m <= 0.01 and m + 1.645 * se <= 0.03) else ("kebocoran" if m > 0.03 else "pemanfaatan kecil/tidak pasti")


if __name__ == "__main__":
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 20; variants = (sys.argv[3] if len(sys.argv) > 3 else "N0,V4").split(","); path = f"{OUT}/nm_paired_replikasi.csv"; t0 = time.time()
    done = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame(columns=["scenario", "rep", "variant", "auc"])
    # check: tokens_v V4 equals M.tokens V4; N0 contains no covariates
    _, _, tr, te = JOBS[0]; a, _ = tokens_v(XSEG, PID, tr, "V4"); b, _ = M.tokens(XSEG, PID, tr, "V4", no_as=True); assert all(np.array_equal(a[i], b[i]) for i in range(0, N, 20)); print("V4 token checks passed", flush=True)
    for scen in ("H1", "H2a"):
        for rep in range(n):
            todo = [v for v in variants if not ((done.scenario == scen) & (done.rep == rep) & (done.variant == v)).any()]
            if not todo:
                continue
            X2, pid2 = shuffled(scen, (30500 if scen == "H1" else 30900) + rep)
            for v in todo:
                P = np.full(N, np.nan)
                for r, k, tr, te in JOBS[:5]:
                    P[te] = null_fold(X2, pid2, r, k, tr, te, v)
                done = pd.concat([done, pd.DataFrame([dict(scenario=scen, rep=rep, variant=v, auc=float(roc_auc_score(Y, P)))])]); done.to_csv(path, index=False)
            print(f"{scen} rep {rep}: " + " ".join(f"{v} {done[(done.scenario == scen) & (done.rep == rep) & (done.variant == v)].auc.iloc[0]:.3f}" for v in variants if ((done.scenario == scen) & (done.rep == rep) & (done.variant == v)).any()) + f" [{time.time() - t0:.0f}s]", flush=True)
    P = done.pivot_table(index=["scenario", "rep"], columns="variant", values="auc").reset_index(); rows = []
    for scen, g in P.groupby("scenario"):
        for v in [x for x in variants if x != "N0"]:
            d = g[v] - g["N0"]; se = d.std(ddof=1) / np.sqrt(len(d)); rows.append(dict(scenario=scen, variant=v, n=len(g), auc_N0=g["N0"].mean(), auc_var=g[v].mean(), delta=d.mean(), se=se, delta_p95=d.mean() + 1.645 * se, t_p=stats.ttest_1samp(d, 0).pvalue, keputusan_R2=decision(d),
                                                                                    R1_mean_plus_1645SE=(g[v].mean() + 1.645 * g[v].std(ddof=1) / np.sqrt(len(g))) if scen == "H1" else np.nan))
    S1 = pd.DataFrame(rows); S1.to_csv(f"{OUT}/nm_paired_ringkasan.csv", index=False); print(S1.round(4).to_string(index=False)); print("NM PAIRED DONE", flush=True)
