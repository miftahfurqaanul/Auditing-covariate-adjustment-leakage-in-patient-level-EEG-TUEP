"""Round 21 (reports/round21_prediction.md, commit 321462c): age sensitivity analysis. Demographic floor (age + sex -> nested_lr), models without
age/sex as DIRECT classifier inputs (I2-noAS, P3-noAS; the normative layer still adjusts for age/sex/era from training controls), fusions,
within-age-strata AUC (<=40, 41-60, >60; round10 estimator). NM is not retrained (still uses age/sex in its summary branch). CPU only.
Usage: python round21_age.py --dry-run | --jobs 12"""
import round17_common as c17
import time, numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
import evalkit as ek, round4 as r4, round9 as r9, round10 as r10, round14 as r14, round17_p3 as p3, round18_fusion as f18

J, Y, N, AS = c17.JOBS, c17.Y, c17.N, c17.AS
AGE = r4.AGE
AGE3 = np.where(AGE <= 40, "<=40", np.where(AGE <= 60, "41-60", ">60"))
STRATA = {"age3": AGE3, "age3 x era": np.char.add(np.char.add(AGE3.astype(str), "|"), r4.YEAR_G)}


def demo_predict(tr, te):
    return ek.nested_lr().fit(AS[tr], Y[tr]).predict_proba(AS[te])[:, 1]


def i2_noas_features(tr):
    return r4.normative_z(r4.DESIGN["I2"], tr)                                      # 114 z, no age/sex columns


def i2_noas_predict(tr, te):
    X = i2_noas_features(tr); return ek.nested_lr().fit(X[tr], Y[tr]).predict_proba(X[te])[:, 1]


def p3_noas_predict(C, pid, tr, te):
    X, _, _ = p3.p3_features(C, pid, tr, r4.YEAR_G, "era")
    pca = PCA(p3.NPC, random_state=0).fit(X[tr]); F = pca.transform(X)                # PCA 50 only, no age/sex
    return ek.nested_lr().fit(F[tr], Y[tr]).predict_proba(F[te])[:, 1]


def checks(C, pid):
    r, k, tr, te = J[0]; ok = []
    X = i2_noas_features(tr); full = r4.features("I2", tr)
    ok.append(c17.pf(X.shape == (N, 114) and np.array_equal(full[:, :114], X) and np.array_equal(full[:, 114:], AS),
                     "I2-noAS matrix = the 114 normative z columns of the I2 matrix, without the age/sex columns"))
    Xp, _, _ = p3.p3_features(C, pid, tr, r4.YEAR_G, "era"); F = PCA(p3.NPC, random_state=0).fit(Xp[tr]).transform(Xp)
    ok.append(c17.pf(F.shape == (N, p3.NPC), f"P3-noAS matrix has {F.shape[1]} columns (PCA only, no age/sex)"))
    rng = np.random.default_rng(0); B0 = r4.BP.copy(); r4.BP[te] += rng.normal(0, 3, r4.BP[te].shape).astype(r4.BP.dtype); Xb = i2_noas_features(tr)[tr]; r4.BP[:] = B0
    ok.append(c17.pf(np.allclose(Xb, i2_noas_features(tr)[tr]), "leakage: perturbing test patients leaves I2-noAS training features unchanged"))
    print("   patients per age stratum and class:"); print(pd.crosstab(AGE3, np.where(Y == 1, "epilepsy", "non-epilepsy")).to_string())
    return all(ok)


if __name__ == "__main__":
    a = c17.args(); c17.header("round21_age"); t0 = time.time()
    C, pid = p3.load_cache(a.jobs)
    if not checks(C, pid):
        raise SystemExit("STOP: checks failed")
    if a.dry_run:
        r, k, tr, te = J[0]
        print(f"dry-run fold r{r}k{k}: Demo {roc_auc_score(Y[te], demo_predict(tr, te)):.4f} | I2-noAS {roc_auc_score(Y[te], i2_noas_predict(tr, te)):.4f} | "
              f"P3-noAS {roc_auc_score(Y[te], p3_noas_predict(C, pid, tr, te)):.4f} | {time.time() - t0:.0f} s"); raise SystemExit
    M = f18.load_members(); P = {"I2": M["I2"], "P3": M["P3"], "NM": M["NM"]}
    P["Demo"] = c17.to_matrix(J, Parallel(n_jobs=a.jobs)(delayed(demo_predict)(tr, te) for _, _, tr, te in J))
    P["I2-noAS"] = c17.to_matrix(J, Parallel(n_jobs=a.jobs)(delayed(i2_noas_predict)(tr, te) for _, _, tr, te in J))
    P["P3-noAS"] = c17.to_matrix(J, Parallel(n_jobs=a.jobs, max_nbytes="1M")(delayed(p3_noas_predict)(C, pid, tr, te) for _, _, tr, te in J))
    P["F1"] = (P["I2"] + P["P3"]) / 2; P["F1-noAS"] = (P["I2-noAS"] + P["P3-noAS"]) / 2
    P["F2"] = (P["I2"] + P["P3"] + P["NM"]) / 3; P["F2-noAS-partial"] = (P["I2-noAS"] + P["P3-noAS"] + P["NM"]) / 3
    P["Metadata floor"] = c17.to_matrix(J, Parallel(n_jobs=a.jobs)(delayed(r9.lr_acq)(tr, te) for _, _, tr, te in J))
    ok = [c17.pf(abs(c17.pooled_auc(P[m]) - v) < 1e-5, f"{m} == {v}") for m, v in [("I2", 0.71210), ("P3", 0.68318), ("F2", 0.72709), ("Metadata floor", 0.64790)]]
    if not all(ok):
        raise SystemExit("STOP: reproduction failed")
    import os; os.makedirs(f"{ek.D}/round21", exist_ok=True); np.savez(f"{ek.D}/round21/age_preds.npz", **P)
    FA = {m: c17.fold_aucs(J, P[m]) for m in P}; pts = {m: r14.rep_metrics(P[m], np.arange(N)) for m in P}
    CI = dict(zip(P, Parallel(n_jobs=a.jobs)(delayed(r14.boot)(P[m], 21100 + i) for i, m in enumerate(P))))
    rows = []
    for m in P:
        row = dict(method=m)
        for k_ in ["auc", "acc", "sens", "spec", "prec", "f1", "bac", "brier", "calib_slope"]:
            row[k_], row[k_ + "_lo"], row[k_ + "_hi"] = pts[m][k_], CI[m].loc[0.025, k_], CI[m].loc[0.975, k_]
        rows.append(row)
    pd.DataFrame(rows).to_csv(f"{c17.R}/round21_age_results.csv", index=False); np.savez(f"{c17.R}/round21_age_fold_aucs.npz", **FA)
    f = lambda m, k_: f"{100 * pts[m][k_]:.1f} [{100 * CI[m].loc[0.025, k_]:.1f}-{100 * CI[m].loc[0.975, k_]:.1f}]"
    print("\n== strict P3, 200 patients, threshold 0.5, mean over 10 repeats [95% CI] ==")
    for m in P:
        print(f"{m:16s} AUC {f(m, 'auc')} | acc {f(m, 'acc')} | sens {f(m, 'sens')} | spec {f(m, 'spec')} | prec {f(m, 'prec')} | F1 {f(m, 'f1')} | BAC {f(m, 'bac')} | "
              f"Brier {pts[m]['brier']:.3f} | slope {pts[m]['calib_slope']:.2f}")
    prim = [("I2-noAS", "I2"), ("F1-noAS", "F1"), ("F2-noAS-partial", "F2")]
    pr = [ek.corrected_ttest(FA[x], FA[y_]) for x, y_ in prim]; hh = f18.holm([p for _, p in pr])
    TT = pd.DataFrame([dict(test=f"{x} - {y_}", dAUC=d, p=p, p_holm=h, folds_better=int((FA[x] > FA[y_]).sum())) for (x, y_), (d, p), h in zip(prim, pr, hh)])
    TT.to_csv(f"{c17.R}/round21_age_tests.csv", index=False)
    print("\n== reliance on age/sex as direct inputs (corrected t-test 160/40, Holm over 3) =="); print(TT.round(4).to_string(index=False))
    tests = [(m, "Demo") for m in ("I2", "I2-noAS", "F2", "F2-noAS-partial")] + [(m, "Metadata floor") for m in ("I2", "F2")]
    W = r10.within(P, np.arange(N), STRATA, tests, 21200); W.to_csv(f"{c17.R}/round21_age_within.csv", index=False)
    print("\n== within-age-strata AUC (round10 estimator; Holm over this table; 'age3 x era' is secondary) ==")
    print(W.assign(auc_a=[r_[f"auc_{r_['test'].split(' - ')[0]}"] for _, r_ in W.iterrows()], auc_b=[r_[f"auc_{r_['test'].split(' - ')[1]}"] for _, r_ in W.iterrows()])
          [["strata", "test", "auc_a", "auc_b", "diff", "lo", "hi", "p", "diff_repeat_mean", "p_holm"]].round(4).to_string(index=False))
    rows = []
    for g in ["<=40", "41-60", ">60"]:
        i = AGE3 == g
        rows.append(dict(stratum=g, n=int(i.sum()), epilepsy=int(Y[i].sum()), **{m: roc_auc_score(Y[i], np.nanmean(P[m], 0)[i]) for m in ("Demo", "I2", "I2-noAS", "F2", "F2-noAS-partial")}))
    PS = pd.DataFrame(rows); PS.to_csv(f"{c17.R}/round21_age_per_stratum.csv", index=False)
    print("\n== AUC inside each age stratum separately (repeat-mean scores; descriptive) =="); print(PS.round(4).to_string(index=False))
    print(f"done in {(time.time() - t0) / 60:.1f} min")
