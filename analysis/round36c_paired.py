#!/usr/bin/env python3
"""Round 36c (CPU, Rohira): paired null test (RF without adjustment vs RF-logpat) and real Rohira-logV4p data. See PRAREGISTRASI_putaran37_38.md.
Usage: CUDA_VISIBLE_DEVICES="" python round36c_paired.py --null [n=30] | --run"""
import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
from scipy import stats
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
import round36_diagnosis_rohira as R6
import round29_sensitivitas as S
R3, A, J, N, XR, PIDX, SUBJ, YLAB, Y, D, AS = R6.R3, R6.A, R6.J, R6.N, R6.XR, R6.PIDX, R6.SUBJ, R6.YLAB, R6.Y, R6.D, R6.R3.AS
OUT = R6.OUT; XL = np.log(np.maximum(XR.astype(np.float64), 1e-12))


def rf_fold_as(Xw, pidx, XB, kind, use_as, r, k, tr, te):
    tr_p, _ = R3.split_val(list(SUBJ[tr]), YLAB, 100 * r + k); Z = Xw.astype(np.float64) - R6.offsets(Xw, pidx, XB, tr, kind, 100 * r + k)[pidx] if kind != "none" else Xw.astype(np.float64)
    if use_as:
        Z = np.hstack([Z, AS[pidx]])
    fit = np.isin(SUBJ[pidx], tr_p); m = RandomForestClassifier(n_estimators=100, max_depth=1000, class_weight="balanced", n_jobs=1, random_state=1000 * r + k)
    m.fit(Z[fit], Y[pidx][fit]); sel = np.isin(pidx, te)
    return pd.Series(m.predict_proba(Z[sel])[:, 1]).groupby(pidx[sel]).mean().loc[te].values


def null_cell(variant, scen, seed):
    perm = R3.perm_owner(scen, seed); blocks = [np.where(PIDX == q)[0] for q in range(N)]
    Xn = np.vstack([XL[blocks[perm[p]]] for p in range(N)]); pn = np.concatenate([np.full(len(blocks[perm[p]]), p) for p in range(N)]); XB, _ = R6.patient_means(Xn, pn)
    kind = "none" if variant == "RF-none" else "pat"
    return R3.pooled(A.c17.to_matrix(J, Parallel(n_jobs=12)(delayed(rf_fold_as)(Xn, pn, XB, kind, False, J[f][0], J[f][1], J[f][2], J[f][3]) for f in range(len(J)))))


def decision(g):
    d = g.delta; se = d.std(ddof=1) / np.sqrt(len(d)); m = d.mean()
    return "tidak ada bukti kebocoran" if (m <= 0.01 and m + 1.645 * se <= 0.03) else ("kebocoran" if m > 0.03 else "pemanfaatan kecil/tidak pasti")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "--null"; t0 = time.time()
    if mode == "--null":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 30; path = f"{OUT}/paired_rohira_null.csv"; done = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame(columns=["variant", "scenario", "rep", "auc"])
        old = pd.read_csv(f"{OUT}/diagnosis_log_null_replikasi.csv"); old = old[old.cell == "RF-logpat"]                      # the first 10 replicates of RF-logpat (identical seed) are reused
        for scen in ("H1", "H2a"):
            for r in range(n):
                for var in ("RF-none", "RF-logpat"):
                    if ((done.variant == var) & (done.scenario == scen) & (done.rep == r)).any():
                        continue
                    if var == "RF-logpat" and ((old.scenario == scen) & (old.rep == r)).any():
                        a = float(old[(old.scenario == scen) & (old.rep == r)].auc.iloc[0])
                    else:
                        a = null_cell(var, scen, 33000 + r)
                    done = pd.concat([done, pd.DataFrame([dict(variant=var, scenario=scen, rep=r, auc=a)])]); done.to_csv(path, index=False)
                if r % 5 == 4:
                    print(f"{scen} rep {r}: [{time.time() - t0:.0f}s]", flush=True)
        P = done.pivot_table(index=["scenario", "rep"], columns="variant", values="auc").reset_index(); P["delta"] = P["RF-logpat"] - P["RF-none"]; rows = []
        for scen, g in P.groupby("scenario"):
            se = g.delta.std(ddof=1) / np.sqrt(len(g)); rows.append(dict(scenario=scen, n=len(g), auc_none=g["RF-none"].mean(), auc_logpat=g["RF-logpat"].mean(), delta=g.delta.mean(), se=se, delta_p95=g.delta.mean() + 1.645 * se,
                                                                         t_p=stats.ttest_1samp(g.delta, 0).pvalue, keputusan_R2=decision(g), R1_H1_mean_plus_1645SE=(g["RF-logpat"].mean() + 1.645 * g["RF-logpat"].std(ddof=1) / np.sqrt(len(g))) if scen == "H1" else np.nan))
        S1 = pd.DataFrame(rows); S1.to_csv(f"{OUT}/paired_rohira_ringkasan.csv", index=False); print(S1.round(4).to_string(index=False))
    else:
        XB_l, _ = R6.patient_means(XL, PIDX); P = {}
        for nm, kind, use_as, X, XB in (("Rohira-logV4p", "pat", False, XL, XB_l), ("Rohira-logV4p+AS", "pat", True, XL, XB_l)):
            P[nm] = A.c17.to_matrix(J, Parallel(n_jobs=12)(delayed(rf_fold_as)(X, PIDX, XB, kind, use_as, J[f][0], J[f][1], J[f][2], J[f][3]) for f in range(len(J)))); print(nm, "done", f"{time.time() - t0:.0f}s", flush=True)
        Z = np.load(os.path.join(HERE, "hasil33", "pembanding_prediksi.npz")); EV = np.load(os.path.join(HERE, "hasil30", "eval_prediksi_v4.npz")); P.update({"Rohira-F": Z["Rohira_F"], "Rohira+AS": Z["Rohira_AS_AS"], "I2_adj": EV["I2_adj"], "C1": EV["C1"]})
        FA = {k: A.c17.fold_aucs(J, m) for k, m in P.items()}; nb = lambda a: (100 * a.mean(), 100 * (a.mean() - stats.t.ppf(.975, 49) * a.std(ddof=1) * np.sqrt(1 / 50 + .25)), 100 * (a.mean() + stats.t.ppf(.975, 49) * a.std(ddof=1) * np.sqrt(1 / 50 + .25)))
        M1 = pd.DataFrame([dict(model=k, auc_fold=nb(a)[0], lo=nb(a)[1], hi=nb(a)[2], auc_pooled=100 * A.c17.pooled_auc(P[k])) for k, a in FA.items()]); M1.to_csv(f"{OUT}/paired_rohira_metrik.csv", index=False); print(M1.round(2).to_string(index=False))
        rows = []
        for a, b in (("Rohira-logV4p+AS", "Rohira+AS"), ("Rohira-logV4p", "Rohira-F"), ("I2_adj", "Rohira-logV4p+AS"), ("Rohira+AS", "Rohira-F")):
            d, lo, hi, p = S.nb_diff(FA[a], FA[b]); rows.append(dict(test=f"{a} - {b}", diff=d, lo=lo, hi=hi, p=p, folds_better=int((FA[a] > FA[b]).sum())))
        T = pd.DataFrame(rows); T.to_csv(f"{OUT}/paired_rohira_selisih.csv", index=False); print(T.round(4).to_string(index=False)); np.savez(f"{OUT}/paired_rohira_prediksi.npz", **{k.replace("+", "_AS_").replace("-", "_"): v for k, v in P.items() if "log" in k})
    print(f"done {time.time() - t0:.0f}s")
