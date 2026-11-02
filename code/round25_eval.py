"""Round 25 evaluation (reports/round25_prediction.md, commit ec52215). Reads data/round25/{cbramod,labram}/r{r}_k{k}.json. CPU only.
Primary (Holm over 2, cross-track): F2 - LaBraM-FT, F2 - CBraMod-FT (Nadeau-Bengio corrected t-test, 160/40, 50 fold AUCs).
Secondary/descriptive: full metrics with CIs, BAC at the validation threshold, other pairwise tests, within-stratum AUC, 169-patient subset of
the same predictions, hybrid F2 + FM fusions, per-fold diagnostics. With --partial, runs on the folds finished so far (no primary claim)."""
import round17_common as c17
import sys, json, os, numpy as np, pandas as pd
from joblib import Parallel, delayed
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
import evalkit as ek, round9 as r9, round10 as r10, round18_fusion as f18, round22_eval as e22

J, Y, N, SUBJ = c17.JOBS, c17.Y, c17.N, c17.SUBJ
D25 = f"{ek.D}/round25"; FM = {"LaBraM-FT": "labram", "CBraMod-FT": "cbramod"}


def load(model):
    P = np.full((10, N), np.nan); info = []; pos = {s: i for i, s in enumerate(SUBJ)}
    for r, k, tr, te in J:
        f = f"{D25}/{model}/r{r}_k{k}.json"
        if not os.path.exists(f):
            continue
        d = json.load(open(f)); assert set(d["test"]) == set(SUBJ[te]), f"{f}: test patients differ from the canonical fold"
        P[r, te] = [d["test"][s] for s in SUBJ[te]]; info.append(dict(r=r, k=k, best_epoch=d["best_epoch"], val_auc=d["val_auc"], thr=d["val_threshold"],
                                                                    test_auc=roc_auc_score(Y[te], P[r, te]), minutes=d["seconds"] / 60))
    return P, pd.DataFrame(info)


if __name__ == "__main__":
    partial = "--partial" in sys.argv; c17.header("round25_eval" + (" (PARTIAL)" if partial else ""))
    M = f18.load_members(); M["C1"] = np.load(f"{c17.OUT}/p1.npz")["C1"]; M["LaBraM-beku"] = np.load(f"{c17.OUT}/p5.npz")["LaBraM_raw"]
    M["F2-noAS-partial"] = np.load(f"{ek.D}/round21/age_preds.npz")["F2-noAS-partial"]
    ok = [c17.pf(abs(c17.pooled_auc(M[m]) - v) < 1e-5, f"{m} reproduces {v}") for m, v in [("I2", 0.71210), ("C1", 0.66627), ("P3", 0.68318), ("NM", 0.70509), ("LaBraM-beku", 0.63997)]]
    F2 = f18.fuse(M, ["I2", "P3", "NM"]); ok.append(c17.pf(abs(c17.pooled_auc(F2) - 0.72709) < 1e-5, "F2 reproduces 0.72709"))
    if not all(ok):
        sys.exit("STOP")
    FT, INFO = {}, {}
    for name, m in FM.items():
        FT[name], INFO[name] = load(m)
    done = [f for f, (r, k, _, te) in enumerate(J) if all(not np.isnan(FT[n][r, te]).any() for n in FT)]
    print(f"folds finished for both models: {len(done)}/50")
    if len(done) < 2:
        sys.exit("STOP: fewer than 2 folds finished for both models")
    if len(done) < 50 and not partial:
        sys.exit("STOP: not all 50 folds finished (use --partial for a descriptive look)")
    Jd = [J[f] for f in done]; reps = sorted({r for r, _, _, _ in Jd if sum(1 for r2, _, _, _ in Jd if r2 == r) == 5})
    P = {"F2": F2, "I2": M["I2"], "C1": M["C1"], "F2-noAS-partial": M["F2-noAS-partial"], "LaBraM-beku": M["LaBraM-beku"], **FT,
         "F2 + LaBraM-FT": (M["I2"] + M["P3"] + M["NM"] + FT["LaBraM-FT"]) / 4, "F2 + CBraMod-FT": (M["I2"] + M["P3"] + M["NM"] + FT["CBraMod-FT"]) / 4}
    P["Metadata floor"] = c17.to_matrix(J, Parallel(n_jobs=12)(delayed(r9.lr_acq)(tr, te) for _, _, tr, te in J))
    FA = {m: c17.fold_aucs(Jd, P[m]) for m in P}
    # primary
    rows = []
    for fm in FT:
        d, p = ek.corrected_ttest(FA["F2"], FA[fm]); rows.append(dict(test=f"F2 - {fm}", dAUC=d, p=p, folds_better=int((FA["F2"] > FA[fm]).sum()), n_folds=len(Jd)))
    PR = pd.DataFrame(rows); PR["p_holm"] = f18.holm(PR.p.values)
    print(f"\n== primary family (Holm over 2, cross-track){' -- PARTIAL, not the pre-registered test' if partial else ''} =="); print(PR.round(4).to_string(index=False))
    # metrics on complete repeats
    Pm = {m: P[m][reps] for m in P}
    if not reps:
        print("\n(no complete repeat yet: metrics, within-stratum and 169-patient tables skipped)"); reps = None
    if reps is None:
        sys.exit()
    print(f"\n== strict P3, 200 patients, threshold 0.5 [95% CI]; complete repeats {reps} =="); T = e22.metrics_table(Pm, 25100); e22.show(T)
    for fm in FT:                                                              # BAC at each fold's own validation threshold
        I = INFO[fm].set_index(["r", "k"]); b = []
        for r in reps:
            pred = np.zeros(N, bool)
            for k in range(5):
                te = [j[3] for j in J if j[0] == r and j[1] == k][0]; pred[te] = FT[fm][r, te] > I.loc[(r, k), "thr"]
            b.append(balanced_accuracy_score(Y, pred))
        print(f"{fm}: BAC at the validation threshold {np.mean(b):.4f} (mean over repeats)")
    # secondary
    sec = [("I2", "LaBraM-FT"), ("I2", "CBraMod-FT"), ("C1", "LaBraM-FT"), ("C1", "CBraMod-FT"), ("LaBraM-FT", "LaBraM-beku"), ("CBraMod-FT", "LaBraM-FT"),
           ("F2-noAS-partial", "LaBraM-FT"), ("F2-noAS-partial", "CBraMod-FT"), ("F2 + LaBraM-FT", "F2"), ("F2 + CBraMod-FT", "F2")]
    S = pd.DataFrame([dict(test=f"{a} - {b}", dAUC=ek.corrected_ttest(FA[a], FA[b])[0], p=ek.corrected_ttest(FA[a], FA[b])[1], folds_better=int((FA[a] > FA[b]).sum())) for a, b in sec])
    print("\n== secondary pooled tests (descriptive) =="); print(S.round(4).to_string(index=False))
    # within strata (complete repeats)
    wt = [("LaBraM-FT", "Metadata floor"), ("CBraMod-FT", "Metadata floor"), ("F2", "LaBraM-FT"), ("F2", "CBraMod-FT")]
    W_ = r10.within(Pm, np.arange(N), c17.STRATA, wt, 25200); print("\n== within-stratum AUC (descriptive) =="); e22.within_show(W_)
    # 169-patient subset of the same predictions (no retraining)
    K = c17.KEEP169; A169 = pd.DataFrame([dict(model=m, auc_200=c17.pooled_auc(Pm[m]), auc_169=float(np.mean([roc_auc_score(Y[K], p[K]) for p in Pm[m]]))) for m in
                                          ("F2", "I2", "LaBraM-FT", "CBraMod-FT", "LaBraM-beku")])
    print("\n== 169 patients without <=2009, same predictions (descriptive) =="); print(A169.round(4).to_string(index=False))
    # per-fold diagnostics
    print("\n== per-fold diagnostics ==")
    for fm in FT:
        I = INFO[fm]; rho = spearmanr(I.val_auc, I.test_auc).correlation
        print(f"{fm}: folds {len(I)}; best epoch median {I.best_epoch.median():.0f} (range {I.best_epoch.min()}-{I.best_epoch.max()}); "
              f"Spearman(val AUROC, test fold AUC) {rho:.3f}; minutes per fold {I.minutes.mean():.1f} (total {I.minutes.sum() / 60:.1f} h)")
    if not partial:
        PR.to_csv(f"{c17.R}/round25_primary.csv", index=False); T.to_csv(f"{c17.R}/round25_results.csv", index=False); S.to_csv(f"{c17.R}/round25_secondary.csv", index=False)
        W_.to_csv(f"{c17.R}/round25_within.csv", index=False); A169.to_csv(f"{c17.R}/round25_169.csv", index=False)
        pd.concat([INFO[fm].assign(model=fm) for fm in FT]).to_csv(f"{c17.R}/round25_folds.csv", index=False)
        np.savez(f"{c17.R}/round25_fold_aucs.npz", **{k_.replace(" ", "_"): v for k_, v in FA.items()})
