"""Round 24 evaluation (reports/round24_prediction.md, commit 48092a0). Reads data/round24/r{1,2,3}.npz. CPU only.
Primary family (Holm over 3): R1 F2w - F2 if the gate passed (else p = 1, no test); R2 P3-KS - P3; R3 AUC in the 41-60 age stratum, R3 - I2
(repeat-averaged scores, paired patient bootstrap 2000x inside the stratum). Secondary/descriptive: metrics with CIs, pooled tests,
within-stratum AUC, negative controls, 169-patient analyses, F2 with the new members swapped in."""
import round17_common as c17
import sys, numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score
import evalkit as ek, round4 as r4, round9 as r9, round10 as r10, round14 as r14, round18_fusion as f18, round22_eval as e22

J, Y, N = c17.JOBS, c17.Y, c17.N
D24 = f"{ek.D}/round24"; R1, R2, R3 = (np.load(f"{D24}/r{i}.npz") for i in (1, 2, 3))
AGE = r4.AGE; MID = (AGE > 40) & (AGE <= 60)


def boot_p(d):
    d = np.asarray(d); return max(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())), 1 / len(d))


def stratum_auc_test(A, B, mask, seed, nb=2000):
    a, b = np.nanmean(A, 0), np.nanmean(B, 0); idx = np.where(mask)[0]; pt = roc_auc_score(Y[idx], a[idx]) - roc_auc_score(Y[idx], b[idx])
    rng = np.random.default_rng(seed); d = []
    for _ in range(nb):
        s = rng.choice(idx, len(idx))
        if len(np.unique(Y[s])) == 2:
            d.append(roc_auc_score(Y[s], a[s]) - roc_auc_score(Y[s], b[s]))
    return pt, np.percentile(d, 2.5), np.percentile(d, 97.5), boot_p(d), roc_auc_score(Y[idx], a[idx]), roc_auc_score(Y[idx], b[idx])


if __name__ == "__main__":
    c17.header("round24_eval")
    M = f18.load_members(); M["C1"] = np.load(f"{c17.OUT}/p1.npz")["C1"]
    ok = [c17.pf(abs(c17.pooled_auc(M[m]) - v) < 1e-5, f"{m} reproduces {v}") for m, v in [("I2", 0.71210), ("C1", 0.66627), ("P3", 0.68318), ("NM", 0.70509)]]
    F2 = f18.fuse(M, ["I2", "P3", "NM"]); ok.append(c17.pf(abs(c17.pooled_auc(F2) - 0.72709) < 1e-5, "F2 reproduces 0.72709"))
    if not all(ok):
        sys.exit("STOP")
    gate = bool(R1["gate"]); P = {"I2": M["I2"], "C1": M["C1"], "P3": M["P3"], "NM": M["NM"], "F2": F2, "W": R1["W"], "PLV": R1["PLV"],
                                  "P3-KS": R2["P3KS"], "P3-K": R2["P3K"], "R3": R3["R3"],
                                  "F2 (P3-KS)": f18.fuse({**M, "P3": R2["P3KS"]}, ["I2", "P3", "NM"]), "F2 (R3 for I2)": f18.fuse({**M, "I2": R3["R3"]}, ["I2", "P3", "NM"])}
    if gate:
        P["F2w"] = f18.fuse({**M, "W": R1["W"]}, ["I2", "P3", "NM", "W"])
    P["Metadata floor"] = c17.to_matrix(J, Parallel(n_jobs=12)(delayed(r9.lr_acq)(tr, te) for _, _, tr, te in J))
    FA = {m: c17.fold_aucs(J, P[m]) for m in P}
    print(f"\n== R1 gate (training data only): inner-OOF AUC mean {R1['gate_auc'].mean():.4f}, Spearman with I2 {R1['gate_rho'].mean():.4f} -> {'PASSED' if gate else 'FAILED'} ==")
    # primary family
    if gate:
        d1, p1 = ek.corrected_ttest(FA["F2w"], FA["F2"]); prim1 = dict(test="R1 F2w - F2", dAUC=d1, p=p1, note=f"folds better {(FA['F2w'] > FA['F2']).sum()}/50")
    else:
        prim1 = dict(test="R1 F2w - F2", dAUC=np.nan, p=1.0, note="gate FAILED: rejected without test (pre-registered)")
    d2, p2 = ek.corrected_ttest(FA["P3-KS"], FA["P3"]); prim2 = dict(test="R2 P3-KS - P3", dAUC=d2, p=p2, note=f"folds better {(FA['P3-KS'] > FA['P3']).sum()}/50")
    pt, lo, hi, p3_, aR, aI = stratum_auc_test(P["R3"], P["I2"], MID, 24400)
    prim3 = dict(test="R3 AUC in age 41-60: R3 - I2", dAUC=pt, p=p3_, note=f"AUC R3 {aR:.4f} vs I2 {aI:.4f}; CI [{lo:+.4f}, {hi:+.4f}]; n {int(MID.sum())}")
    PR = pd.DataFrame([prim1, prim2, prim3]); PR["p_holm"] = f18.holm(PR.p.values)
    print("\n== primary family (Holm over 3) =="); print(PR.round(4).to_string(index=False))
    # metrics
    pts = {m: r14.rep_metrics(P[m], np.arange(N)) for m in P}; CI = dict(zip(P, Parallel(n_jobs=12)(delayed(r14.boot)(P[m], 24100 + i) for i, m in enumerate(P))))
    rows = []
    for m in P:
        row = dict(method=m)
        for k_ in ["auc", "acc", "sens", "spec", "prec", "f1", "bac", "brier", "calib_slope"]:
            row[k_], row[k_ + "_lo"], row[k_ + "_hi"] = pts[m][k_], CI[m].loc[0.025, k_], CI[m].loc[0.975, k_]
        rows.append(row)
    T = pd.DataFrame(rows); print("\n== strict P3, 200 patients, threshold 0.5 [95% CI] =="); e22.show(T)
    # secondary pooled tests
    sec = [("W", "I2"), ("W", "C1"), ("PLV", "W"), ("P3-K", "P3"), ("R3", "I2"), ("F2 (P3-KS)", "F2"), ("F2 (R3 for I2)", "F2")]
    S = pd.DataFrame([dict(test=f"{x} - {y_}", dAUC=ek.corrected_ttest(FA[x], FA[y_])[0], p=ek.corrected_ttest(FA[x], FA[y_])[1], folds_better=int((FA[x] > FA[y_]).sum())) for x, y_ in sec])
    print("\n== secondary pooled tests (descriptive) =="); print(S.round(4).to_string(index=False))
    # within strata
    wt = [("P3-KS", "P3"), ("R3", "I2"), ("W", "Metadata floor")] + ([("F2w", "F2")] if gate else [])
    W_ = r10.within(P, np.arange(N), c17.STRATA, wt, 24200); print("\n== within-stratum AUC (descriptive) =="); e22.within_show(W_)
    # age strata
    AS_ = pd.DataFrame([dict(stratum=g, n=int(m_.sum()), **{m: roc_auc_score(Y[m_], np.nanmean(P[m], 0)[m_]) for m in ("I2", "R3", "F2", "F2 (R3 for I2)")})
                        for g, m_ in [("<=40", AGE <= 40), ("41-60", MID), (">60", AGE > 60)]])
    print("\n== AUC per age stratum (repeat-mean scores) =="); print(AS_.round(4).to_string(index=False))
    # negative controls
    perm = R3["R3_agepermuted"]; pa = np.array([c17.pooled_auc(p) for p in perm]); pm = np.array([roc_auc_score(Y[MID], np.nanmean(p, 0)[MID]) for p in perm])
    NC = pd.DataFrame([dict(model="W", control="year shuffled 20x", observed=c17.pooled_auc(P["W"]), null_mean=R1["shuffle_auc"].mean(), null_p95=np.percentile(R1["shuffle_auc"], 95)),
                       dict(model="P3-KS", control="year shuffled 20x", observed=c17.pooled_auc(P["P3-KS"]), null_mean=R2["shuffle_auc"].mean(), null_p95=np.percentile(R2["shuffle_auc"], 95)),
                       dict(model="R3", control="ages permuted in variance model 20x (pooled AUC)", observed=c17.pooled_auc(P["R3"]), null_mean=pa.mean(), null_p95=np.percentile(pa, 95)),
                       dict(model="R3", control="ages permuted in variance model 20x (AUC 41-60)", observed=aR, null_mean=pm.mean(), null_p95=np.percentile(pm, 95))])
    print("\n== negative controls =="); print(NC.round(4).to_string(index=False))
    # 169 patients
    I169, P169 = np.load(f"{c17.OUT}/p1.npz")["I2_169"], np.load(f"{c17.OUT}/p3.npz")["P3_169"]
    rows = [dict(model=m, auc=c17.pooled_auc(v)) for m, v in [("I2-169", I169), ("P3-169", P169), ("W-169", R1["W_169"]), ("P3-KS-169", R2["P3KS_169"]), ("R3-169", R3["R3_169"])]]
    for x, y_, nm in [(R2["P3KS_169"], P169, "P3-KS - P3 (169)"), (R3["R3_169"], I169, "R3 - I2 (169)")]:
        d, p = ek.corrected_ttest(c17.fold_aucs(c17.J169, x), c17.fold_aucs(c17.J169, y_), 135, 34); rows.append(dict(model=nm, dAUC=d, p=p))
    A169 = pd.DataFrame(rows); print("\n== 169 patients without <=2009 (descriptive) =="); print(A169.round(4).to_string(index=False))
    PR.to_csv(f"{c17.R}/round24_primary.csv", index=False); T.to_csv(f"{c17.R}/round24_results.csv", index=False); S.to_csv(f"{c17.R}/round24_secondary.csv", index=False)
    W_.to_csv(f"{c17.R}/round24_within.csv", index=False); AS_.to_csv(f"{c17.R}/round24_age_strata.csv", index=False); NC.to_csv(f"{c17.R}/round24_negative_controls.csv", index=False)
    A169.to_csv(f"{c17.R}/round24_169.csv", index=False); np.savez(f"{c17.R}/round24_fold_aucs.npz", **{k_.replace(" ", "_"): v for k_, v in FA.items()})
