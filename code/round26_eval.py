"""Round 26 evaluation (reports/round26_prediction.md, commit 69df653). Reads data/round26/preds.npz and features.npz. CPU only.
Primary (Holm over 2): E1 I2-WP - I2, E2 I2-SM - I2 (Nadeau-Bengio corrected t-test, 160/40). Secondary/descriptive: metrics with CIs,
within-stratum AUC, 169 patients, sensitivity variants, F2 with the new I2, negative controls, drowsiness and photic descriptives."""
import round17_common as c17
import sys, numpy as np, pandas as pd
from joblib import Parallel, delayed
import evalkit as ek, round4 as r4, round9 as r9, round10 as r10, round14 as r14, round18_fusion as f18, round22_eval as e22

J, Y, N = c17.JOBS, c17.Y, c17.N
PR_ = np.load(f"{ek.D}/round26/preds.npz")
with np.load(f"{ek.D}/round26/features.npz", allow_pickle=True) as _Z:
    Z = {k: _Z[k] for k in _Z.files}

if __name__ == "__main__":
    c17.header("round26_eval")
    M = f18.load_members(); F2 = f18.fuse(M, ["I2", "P3", "NM"])
    ok = [c17.pf(abs(c17.pooled_auc(PR_["I2"]) - 0.71210) < 1e-5, "I2 reproduces 0.71210"), c17.pf(np.allclose(PR_["I2"], M["I2"]), "I2 predictions identical to round 17"),
          c17.pf(abs(c17.pooled_auc(F2) - 0.72709) < 1e-5, "F2 reproduces 0.72709")]
    if not all(ok):
        sys.exit("STOP")
    P = {"I2": PR_["I2"], "I2-WP": PR_["WP"], "I2-SM": PR_["SM"], "E1b WP all patients": PR_["WP_all"], "E1c WP no age": PR_["WP_noage"],
         "E2a photic only": PR_["PHOTIC_ONLY"], "E2b state only": PR_["STATE_ONLY"], "E2c state concat": PR_["STATE_CONCAT"], "I2-WP+SM": PR_["WP_SM"],
         "F2": F2, "F2-WP": f18.fuse({**M, "I2": PR_["WP"]}, ["I2", "P3", "NM"]), "F2-SM": f18.fuse({**M, "I2": PR_["SM"]}, ["I2", "P3", "NM"]),
         "F2-WP+SM": f18.fuse({**M, "I2": PR_["WP_SM"]}, ["I2", "P3", "NM"])}
    P["Metadata floor"] = c17.to_matrix(J, Parallel(n_jobs=12)(delayed(r9.lr_acq)(tr, te) for _, _, tr, te in J))
    FA = {m: c17.fold_aucs(J, P[m]) for m in P}
    rows = []
    for a, b in [("I2-WP", "I2"), ("I2-SM", "I2")]:
        d, p = ek.corrected_ttest(FA[a], FA[b]); rows.append(dict(test=f"{a} - {b}", dAUC=d, p=p, folds_better=int((FA[a] > FA[b]).sum()), pooled_a=c17.pooled_auc(P[a]), pooled_b=c17.pooled_auc(P[b])))
    PRIM = pd.DataFrame(rows); PRIM["p_holm"] = f18.holm(PRIM.p.values)
    print("\n== primary family (Holm over 2) =="); print(PRIM.round(4).to_string(index=False))
    T = e22.metrics_table(P, 26100); print("\n== strict P3, 200 patients, threshold 0.5 [95% CI] =="); e22.show(T)
    sec = [("E1b WP all patients", "I2"), ("E1c WP no age", "I2"), ("E2a photic only", "I2"), ("E2b state only", "I2"), ("E2c state concat", "I2"), ("I2-WP+SM", "I2"),
           ("F2-WP", "F2"), ("F2-SM", "F2"), ("F2-WP+SM", "F2")]
    S = pd.DataFrame([dict(test=f"{a} - {b}", dAUC=ek.corrected_ttest(FA[a], FA[b])[0], p=ek.corrected_ttest(FA[a], FA[b])[1], folds_better=int((FA[a] > FA[b]).sum())) for a, b in sec])
    print("\n== secondary tests (descriptive) =="); print(S.round(4).to_string(index=False))
    W_ = r10.within(P, np.arange(N), c17.STRATA, [("I2-WP", "I2"), ("I2-SM", "I2"), ("F2-WP", "F2"), ("F2-SM", "F2"), ("I2-SM", "Metadata floor")], 26200)
    print("\n== within-stratum AUC (descriptive) =="); e22.within_show(W_)
    rows = []
    for v, base in [("WP", "I2"), ("SM", "I2")]:
        d, p = ek.corrected_ttest(c17.fold_aucs(c17.J169, PR_[f"{v}_169"]), c17.fold_aucs(c17.J169, PR_["I2_169"]), 135, 34)
        rows.append(dict(test=f"I2-{v} - I2 (169, retrained)", auc_new=c17.pooled_auc(PR_[f"{v}_169"]), auc_I2=c17.pooled_auc(PR_["I2_169"]), dAUC=d, p=p))
    A169 = pd.DataFrame(rows); print("\n== 169 patients without <=2009 (descriptive) =="); print(A169.round(4).to_string(index=False))
    pa = np.array([c17.pooled_auc(p) for p in PR_["WP_perm"]]); fa_perm = np.array([c17.fold_aucs(J, p).mean() for p in PR_["WP_perm"]])
    rh = np.array([c17.pooled_auc(p) for p in PR_["SM_randhalf"]]); fa_rh = np.array([c17.fold_aucs(J, p).mean() for p in PR_["SM_randhalf"]])
    NC = pd.DataFrame([dict(model="I2-WP", control="within-person pairing broken 20x", observed_pooled=c17.pooled_auc(P["I2-WP"]), null_mean=pa.mean(), null_p95=np.percentile(pa, 95),
                            observed_fold_mean=FA["I2-WP"].mean(), null_fold_mean=fa_perm.mean(), rank_p=(1 + (fa_perm >= FA["I2-WP"].mean()).sum()) / (1 + len(fa_perm))),
                       dict(model="I2-SM", control="random half of windows 20x", observed_pooled=c17.pooled_auc(P["I2-SM"]), null_mean=rh.mean(), null_p95=np.percentile(rh, 95),
                            observed_fold_mean=FA["I2-SM"].mean(), null_fold_mean=fa_rh.mean(), rank_p=(1 + (fa_rh >= FA["I2-SM"].mean()).sum()) / (1 + len(fa_rh)))])
    print("\n== negative controls =="); print(NC.round(4).to_string(index=False))
    # descriptives: drowsiness (posterior alpha index < 0) and photic status
    wp_, ai, ph = Z["win_patient"], Z["win_aidx"], Z["win_photic"]
    drowsy = np.array([(ai[wp_ == i] < 0).mean() for i in range(N)]); phot = np.array([ph[wp_ == i].mean() for i in range(N)])
    DS = pd.DataFrame(dict(group=np.where(Y == 1, "epilepsi", "non-epilepsi"), era=r4.YEAR_G, drowsy=drowsy, photic=phot))
    print("\n== drowsy-window fraction (posterior alpha index < 0) and photic-window fraction, first session (descriptive) ==")
    print(DS.groupby("group")[["drowsy", "photic"]].agg(["mean", "median"]).round(3).to_string())
    print(DS.pivot_table(index="era", columns="group", values=["drowsy", "photic"], aggfunc="mean").round(3).to_string())
    from scipy.stats import mannwhitneyu
    print(f"drowsy fraction epilepsy vs non-epilepsy: Mann-Whitney p = {mannwhitneyu(drowsy[Y == 1], drowsy[Y == 0]).pvalue:.4f}")
    PRIM.to_csv(f"{c17.R}/round26_primary.csv", index=False); T.to_csv(f"{c17.R}/round26_results.csv", index=False); S.to_csv(f"{c17.R}/round26_secondary.csv", index=False)
    W_.to_csv(f"{c17.R}/round26_within.csv", index=False); A169.to_csv(f"{c17.R}/round26_169.csv", index=False); NC.to_csv(f"{c17.R}/round26_negative_controls.csv", index=False)
    DS.to_csv(f"{c17.R}/round26_state_descriptives.csv", index=False); np.savez(f"{c17.R}/round26_fold_aucs.npz", **{k.replace(" ", "_"): v for k, v in FA.items()})
