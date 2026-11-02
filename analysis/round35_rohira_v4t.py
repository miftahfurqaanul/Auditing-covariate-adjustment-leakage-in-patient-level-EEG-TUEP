#!/usr/bin/env python3
"""Round 35 (CPU): Rohira with V4t (adjustment fitted ONLY on the tr_p patients that train the RF). See PRAREGISTRASI_putaran35.md.
Usage: CUDA_VISIBLE_DEVICES="" python round35_rohira_v4t.py --null [n=20] | --run"""
import os, sys, json, time, hashlib
os.environ["CUDA_VISIBLE_DEVICES"] = ""; os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ["PYTHONWARNINGS"] = "ignore"
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, "/home/miftah_f/RTX-5090/tuep_fresh/code")
import numpy as np, pandas as pd
from scipy import stats
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import round33_pembanding_v4 as R3
import round29_sensitivitas as S
A, J, Y, N, SUBJ, AS, D, XR, PIDX, YLAB = R3.A, R3.J, R3.Y, R3.N, R3.SUBJ, R3.AS, R3.D, R3.XR, R3.PIDX, R3.YLAB
OUT = os.path.join(HERE, "hasil35"); os.makedirs(OUT, exist_ok=True)


def rohira_fold_t(Xw, pidx, use_as, r, k, tr, te):
    tr_p, _ = R3.split_val(list(SUBJ[tr]), YLAB, 100 * r + k); fit_idx = np.array([R3.POS[s] for s in tr_p]); Z = R3.adjust(Xw, fit_idx, owner=pidx)      # V4t: B from tr_p only
    if use_as:
        Z = np.hstack([Z, AS[pidx]])
    fit = np.isin(SUBJ[pidx], tr_p); m = RandomForestClassifier(n_estimators=100, max_depth=1000, class_weight="balanced", n_jobs=1, random_state=1000 * r + k)
    m.fit(Z[fit], Y[pidx][fit]); sel = np.isin(pidx, te)
    return pd.Series(m.predict_proba(Z[sel])[:, 1]).groupby(pidx[sel]).mean().loc[te].values


def checks():
    n = 0
    for f in (0, 17):
        r, k, tr, te = J[f]; tr_p, va_p = R3.split_val(list(SUBJ[tr]), YLAB, 100 * r + k); fit_idx = np.array([R3.POS[s] for s in tr_p]); R = R3.adjust(XR, fit_idx, owner=PIDX); rows = np.isin(PIDX, fit_idx)
        assert np.abs(np.linalg.lstsq(D[PIDX][rows], R[rows], rcond=None)[0]).max() < 1e-6; n += 1                       # tr_p residual orthogonal to the design
        vi = np.array([R3.POS[s] for s in va_p]); X2 = XR.copy(); mv = np.isin(PIDX, np.concatenate([vi, te])); X2[mv] = X2[mv] * 5 + 3
        assert np.allclose(R3.adjust(XR, fit_idx, PIDX)[rows], R3.adjust(X2, fit_idx, PIDX)[rows]); n += 1               # validation/test data do not change the training features
    return n


def null_rep(scen, seed):
    perm = R3.perm_owner(scen, seed); blocks = [np.where(PIDX == q)[0] for q in range(N)]
    Xn = np.vstack([XR[blocks[perm[p]]] for p in range(N)]); pn = np.concatenate([np.full(len(blocks[perm[p]]), p) for p in range(N)])
    return R3.pooled(A.c17.to_matrix(J, Parallel(n_jobs=12)(delayed(rohira_fold_t)(Xn, pn, False, J[f][0], J[f][1], J[f][2], J[f][3]) for f in range(len(J)))))


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "--null"; t0 = time.time(); print(f"{checks()} checks passed", flush=True)
    if mode == "--null":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 20; rows = []
        for scen in ("H1", "H2a"):
            vals = np.array([null_rep(scen, 33000 + r) for r in range(n)]); se = vals.std(ddof=1) / np.sqrt(n)
            rows.append(dict(scenario=scen, variant="V4t", n=n, mean=vals.mean(), sd=vals.std(ddof=1), p95_rata2=vals.mean() + 1.645 * se,
                             keputusan="tidak ada bukti kebocoran" if vals.mean() <= 0.52 and vals.mean() + 1.645 * se <= 0.54 else ("kebocoran" if vals.mean() > 0.55 else "pemanfaatan kecil/tidak pasti"))); print(rows[-1], f"[{time.time() - t0:.0f}s]", flush=True)
        pd.DataFrame(rows).to_csv(f"{OUT}/rohira_v4t_null.csv", index=False)
    else:
        P = {nm: A.c17.to_matrix(J, Parallel(n_jobs=12)(delayed(rohira_fold_t)(XR, PIDX, a, J[f][0], J[f][1], J[f][2], J[f][3]) for f in range(len(J)))) for nm, a in (("Rohira-V4t", False), ("Rohira-V4t+AS", True))}
        Z = np.load(os.path.join(HERE, "hasil33", "pembanding_prediksi.npz")); EV = np.load(os.path.join(HERE, "hasil30", "eval_prediksi_v4.npz"))
        P.update({"Rohira-F": Z["Rohira_F"], "Rohira+AS": Z["Rohira_AS_AS"], "I2_adj": EV["I2_adj"], "C1": EV["C1"], "Floor": EV["Floor"]}); FA = {k: A.c17.fold_aucs(J, m) for k, m in P.items()}
        YD = np.char.add(np.char.add(A.ERA, "|"), A.DEV); nb = lambda a: (100 * a.mean(), 100 * (a.mean() - stats.t.ppf(.975, 49) * a.std(ddof=1) * np.sqrt(1 / 50 + .25)), 100 * (a.mean() + stats.t.ppf(.975, 49) * a.std(ddof=1) * np.sqrt(1 / 50 + .25)))
        M1 = pd.DataFrame([dict(model=k, auc_fold=nb(a)[0], lo=nb(a)[1], hi=nb(a)[2], auc_pooled=100 * A.c17.pooled_auc(P[k]), strata_year_x_device=100 * A.r9.strat_auc(Y, P[k].mean(0), YD)) for k, a in FA.items()]); M1.to_csv(f"{OUT}/rohira_v4t_metrik.csv", index=False); print(M1.round(2).to_string(index=False), flush=True)
        pairs = [("Rohira-V4t+AS", "Rohira+AS", "primary"), ("Rohira-V4t", "Rohira-F", "descriptive"), ("I2_adj", "Rohira-V4t+AS", "descriptive"), ("Rohira-V4t+AS", "C1", "descriptive")]; rows = []
        for a, b, fam in pairs:
            d, lo, hi, p = S.nb_diff(FA[a], FA[b]); rows.append(dict(test=f"{a} - {b}", diff=d, lo=lo, hi=hi, p=p, folds_better=int((FA[a] > FA[b]).sum()), family=fam))
        T = pd.DataFrame(rows); T.to_csv(f"{OUT}/rohira_v4t_selisih.csv", index=False); print(T.round(4).to_string(index=False), flush=True)
        np.savez(f"{OUT}/rohira_v4t_prediksi.npz", **{k.replace("+", "_AS_").replace("-", "_"): v for k, v in P.items() if "V4t" in k})
    print(f"done {time.time() - t0:.0f}s")
