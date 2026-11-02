"""Round 20 part A (reports/round20_prediction.md, commit a03c37f): standard whole-sample harmonisation vs the control-referenced layer.
Spectral view (114 band-power features, batch = YEAR_G as I2): H-olsall (I2 design OLS fitted on ALL training patients), H-zall (per-era
z-score from all training patients of the era), H-combat (ComBat of round 11, batch YEAR_G, covariates age+sex, fitted on all training
patients). Spatial view: P3-all (per-era re-centring reference from all training windows of the era; otherwise identical to round17_p3).
Holm over 4: I2-H-olsall, I2-H-zall, I2-H-combat, P3-P3-all. CPU only.  Usage: python round20_harmonisation.py --dry-run | --jobs 12"""
import round17_common as c17
import time, numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
import evalkit as ek, round4 as r4, round9 as r9, round10 as r10, round11 as r11, round14 as r14, round17_p3 as p3, round18_fusion as f18

J, Y, N, BP, AS, YG = c17.JOBS, c17.Y, c17.N, c17.BP, c17.AS, r4.YEAR_G


# ------------------------------------------------------------------ spectral view
def spectral_features(tr, method):
    if method == "I2":
        return np.hstack([r4.normative_z(r4.DESIGN["I2"], tr), AS])
    if method == "C1":
        return np.hstack([BP, AS])
    if method == "H-olsall":
        return np.hstack([r11.ols_z(BP, r4.DESIGN["I2"], tr), AS])
    if method == "H-zall":
        B64 = BP.astype(np.float64); Z = np.empty_like(B64)                              # float64: BP is stored as float32
        for g in np.unique(YG):
            ref = tr[YG[tr] == g]; mu, sd = B64[ref].mean(0), B64[ref].std(0, ddof=1); Z[YG == g] = (B64[YG == g] - mu) / sd
        return np.hstack([Z, AS])
    if method == "H-combat":
        P = r11.combat_fit(BP[tr], YG[tr], AS[tr]); return np.hstack([r11.combat_apply(P, BP, YG, AS), AS])     # same dtype path as round-11 E1b
    raise ValueError(method)


def spectral_predict(tr, te, method):
    X = spectral_features(tr, method); return ek.nested_lr().fit(X[tr], Y[tr]).predict_proba(X[te])[:, 1]


# ------------------------------------------------------------------ spatial view: round17_p3.p3_features with the reference set as a parameter
def p3_features_ref(C, pid, tr, yg, ref):
    """Identical to round17_p3.p3_features(mode='era') except that the reference windows are those of patients in `ref`
    (round 17: ref = training controls; here: ref = all training patients)."""
    cw = np.isin(pid, ref); era = yg[pid]; out = np.empty((len(pid), 5 * 171))
    for b in range(5):
        Cb = C[:, b]; Mg = Cb[cw].mean(0); Rc = np.empty_like(Cb)
        for g in np.unique(era):
            sel = era == g; rw = cw & sel; M = Mg if rw.sum() < p3.MIN_WIN else Cb[rw].mean(0)
            W, _ = p3.inv_sqrt(M); Rc[sel] = W @ Cb[sel] @ W
        out[:, b * 171:(b + 1) * 171] = p3.tangent_windows(Rc)
    cnt = np.bincount(pid, minlength=N); return np.add.reduceat(out, np.r_[0, np.cumsum(cnt)[:-1]], axis=0) / cnt[:, None]


def p3_all_predict(C, pid, tr, te):
    X = p3_features_ref(C, pid, tr, YG, tr)
    pca = PCA(p3.NPC, random_state=0).fit(X[tr]); F = np.hstack([pca.transform(X), AS])
    return ek.nested_lr().fit(F[tr], Y[tr]).predict_proba(F[te])[:, 1]


def checks(C, pid):
    r, k, tr, te = J[0]; c = tr[Y[tr] == 0]; ok = []
    ok.append(c17.pf(np.allclose(p3_features_ref(C, pid, tr, YG, c), p3.p3_features(C, pid, tr, YG, "era")[0], atol=1e-12),
                     "p3_features_ref with ref = training controls reproduces round17_p3.p3_features exactly"))
    Xo = spectral_features(tr, "H-olsall")[:, :114]; ok.append(c17.pf(np.abs(Xo[tr].mean(0)).max() < 1e-8, f"H-olsall: z of ALL training patients has mean 0 ({np.abs(Xo[tr].mean(0)).max():.1e})"))
    Xz = spectral_features(tr, "H-zall")[:, :114]
    w = max(max(np.abs(Xz[tr][YG[tr] == g].mean(0)).max(), np.abs(Xz[tr][YG[tr] == g].std(0, ddof=1) - 1).max()) for g in np.unique(YG))
    ok.append(c17.pf(w < 1e-8, f"H-zall: per-era mean 0 and sd 1 over training patients ({w:.1e})"))
    P = r11.combat_fit(BP[c], r11.YEAR[c], AS[c]); H = r11.combat_apply(P, BP, r11.YEAR, AS)
    ok.append(c17.pf(np.allclose(H, r11.e1b_features(tr)[1]), "ComBat code path reproduces round-11 E1b harmonised features when given controls + calendar year"))
    rng = np.random.default_rng(0); B0 = BP.copy(); BP[te] += rng.normal(0, 3, BP[te].shape).astype(BP.dtype)
    pert = {m: spectral_features(tr, m)[tr] for m in ("H-olsall", "H-zall", "H-combat")}; BP[:] = B0
    ok.append(c17.pf(all(np.allclose(pert[m], spectral_features(tr, m)[tr]) for m in pert), "leakage: perturbing test patients leaves training features unchanged (H-olsall, H-zall, H-combat)"))
    C2 = C.copy(); C2[np.isin(pid, te)] *= 3.0
    ok.append(c17.pf(np.allclose(p3_features_ref(C2, pid, tr, YG, tr)[tr], p3_features_ref(C, pid, tr, YG, tr)[tr]), "leakage: perturbing test covariances leaves P3-all training features unchanged"))
    return all(ok)


if __name__ == "__main__":
    a = c17.args(); c17.header("round20_harmonisation"); t0 = time.time()
    C, pid = p3.load_cache(a.jobs)
    if not checks(C, pid):
        raise SystemExit("STOP: checks failed")
    if a.dry_run:
        r, k, tr, te = J[0]
        print(f"dry-run fold r{r}k{k}: " + " | ".join(f"{m} {roc_auc_score(Y[te], spectral_predict(tr, te, m)):.4f}" for m in ("I2", "H-olsall", "H-zall", "H-combat"))
              + f" | P3-all {roc_auc_score(Y[te], p3_all_predict(C, pid, tr, te)):.4f} | {time.time() - t0:.0f} s"); raise SystemExit
    P = {}
    for m in ("I2", "C1", "H-olsall", "H-zall", "H-combat"):
        P[m] = c17.to_matrix(J, Parallel(n_jobs=a.jobs)(delayed(spectral_predict)(tr, te, m) for _, _, tr, te in J)); print(f"{m:9s} AUC {c17.pooled_auc(P[m]):.5f}", flush=True)
    P["P3-all"] = c17.to_matrix(J, Parallel(n_jobs=a.jobs, max_nbytes="1M")(delayed(p3_all_predict)(C, pid, tr, te) for _, _, tr, te in J))
    P["P3"] = np.load(f"{c17.OUT}/p3.npz")["P3"]; print(f"P3-all    AUC {c17.pooled_auc(P['P3-all']):.5f}", flush=True)
    ok = [c17.pf(abs(c17.pooled_auc(P[m]) - v) < 1e-5, f"{m} == {v}") for m, v in [("I2", 0.71210), ("C1", 0.66627), ("P3", 0.68318)]]
    ok.append(c17.pf(np.allclose(P["I2"], np.load(f"{c17.OUT}/p1.npz")["I2"]), "I2 predictions identical to round 17"))
    if not all(ok):
        raise SystemExit("STOP: reproduction failed")
    P["F1"] = (P["I2"] + P["P3"]) / 2; P["F1-all"] = (P["H-olsall"] + P["P3-all"]) / 2
    P["Metadata floor"] = c17.to_matrix(J, Parallel(n_jobs=a.jobs)(delayed(r9.lr_acq)(tr, te) for _, _, tr, te in J))
    import os; os.makedirs(f"{ek.D}/round20", exist_ok=True); np.savez(f"{ek.D}/round20/harmonisation_preds.npz", **P)
    FA = {m: c17.fold_aucs(J, P[m]) for m in P}; pts = {m: r14.rep_metrics(P[m], np.arange(N)) for m in P}
    CI = dict(zip(P, Parallel(n_jobs=a.jobs)(delayed(r14.boot)(P[m], 20100 + i) for i, m in enumerate(P))))
    prim = [("I2", "H-olsall"), ("I2", "H-zall"), ("I2", "H-combat"), ("P3", "P3-all")]
    pr = [ek.corrected_ttest(FA[x], FA[y_]) for x, y_ in prim]; hh = f18.holm([p for _, p in pr])
    sec = [(m, "C1") for m in ("H-olsall", "H-zall", "H-combat")] + [("F1", "F1-all")]
    rows = []
    for m in P:
        row = dict(method=m)
        for k_ in ["auc", "acc", "sens", "spec", "prec", "f1", "bac", "brier", "calib_slope"]:
            row[k_], row[k_ + "_lo"], row[k_ + "_hi"] = pts[m][k_], CI[m].loc[0.025, k_], CI[m].loc[0.975, k_]
        rows.append(row)
    T = pd.DataFrame(rows)
    tests = [dict(test=f"{x} - {y_}", dAUC=d, p=p, p_holm=h, folds_better=int((FA[x] > FA[y_]).sum()), family="primary (Holm over 4)") for (x, y_), (d, p), h in zip(prim, pr, hh)]
    for x, y_ in sec:
        d, p = ek.corrected_ttest(FA[x], FA[y_]); tests.append(dict(test=f"{x} - {y_}", dAUC=d, p=p, p_holm=np.nan, folds_better=int((FA[x] > FA[y_]).sum()), family="secondary"))
    TT = pd.DataFrame(tests); T.to_csv(f"{c17.R}/round20_harmonisation_results.csv", index=False); TT.to_csv(f"{c17.R}/round20_harmonisation_tests.csv", index=False)
    np.savez(f"{c17.R}/round20_harmonisation_fold_aucs.npz", **FA)
    f = lambda m, k_: f"{100 * pts[m][k_]:.1f} [{100 * CI[m].loc[0.025, k_]:.1f}-{100 * CI[m].loc[0.975, k_]:.1f}]"
    print("\n== strict P3, 200 patients, threshold 0.5, mean over 10 repeats [95% CI] ==")
    for m in P:
        print(f"{m:15s} AUC {f(m, 'auc')} | acc {f(m, 'acc')} | sens {f(m, 'sens')} | spec {f(m, 'spec')} | prec {f(m, 'prec')} | F1 {f(m, 'f1')} | BAC {f(m, 'bac')} | "
              f"Brier {pts[m]['brier']:.3f} | slope {pts[m]['calib_slope']:.2f}")
    print("\n== tests (corrected t-test 160/40) ==")
    print(TT.round(4).to_string(index=False))
    W = r10.within(P, np.arange(N), c17.STRATA, prim + [(m, "Metadata floor") for m in ("I2", "H-olsall", "H-zall", "H-combat", "P3", "P3-all")], 20200)
    W.to_csv(f"{c17.R}/round20_harmonisation_within.csv", index=False)
    print("\n== within-stratum AUC (round10 estimator; Holm over this table) ==")
    print(W.assign(auc_a=[r_[f"auc_{r_['test'].split(' - ')[0]}"] for _, r_ in W.iterrows()], auc_b=[r_[f"auc_{r_['test'].split(' - ')[1]}"] for _, r_ in W.iterrows()])
          [["strata", "test", "auc_a", "auc_b", "diff", "lo", "hi", "p", "diff_repeat_mean", "p_holm"]].round(4).to_string(index=False))
    print(f"done in {(time.time() - t0) / 60:.1f} min")
