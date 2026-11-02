"""Round 19 (reports/round19_prediction.md, commit e0f242a): calibrated sum-rule fusion (CF2, primary) and stacking (SF2) of I2 + P3 + NormMamba-F,
both fitted ONLY on inner out-of-fold scores of the 160 training patients of each outer fold, applied to the frozen outer-test scores.
Same-view control CF-C1 (calibrated I2 + C1). CPU only.
Usage: python round19_fusion.py --inner   (inner OOF for I2, C1, P3 -> data/round19/cpu_inner.npz)
       python round19_fusion.py --eval    (needs data/round19/nm_inner_r*_k*.json from round19_nm_inner.py)"""
import round17_common as c17
import os, sys, json, argparse, numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
import evalkit as ek, round4 as r4, round9 as r9, round10 as r10, round14 as r14, round17_p3 as p3, round18_fusion as f18

J, Y, N, SUBJ = c17.JOBS, c17.Y, c17.N, c17.SUBJ
OUT = f"{ek.D}/round19"; os.makedirs(OUT, exist_ok=True)
lg = lambda q: np.log(np.clip(q, 1e-6, 1 - 1e-6) / (1 - np.clip(q, 1e-6, 1 - 1e-6)))


def inner_folds(r, k, tr):
    return list(StratifiedKFold(5, shuffle=True, random_state=100 * r + k).split(tr, Y[tr]))       # identical to round19_nm_inner.py


def cpu_inner(C, pid, r, k, tr):
    o = {m: np.full(N, np.nan) for m in ("I2", "C1", "P3")}
    for ia, ib in inner_folds(r, k, tr):
        a, b = tr[ia], tr[ib]
        o["I2"][b] = r4.fit_predict("I2", a, b, 0); o["C1"][b] = r4.fit_predict("C1", a, b, 0)
        o["P3"][b] = p3.p3_predict(C, pid, a, b, r4.YEAR_G, "era")[0]
    return o


def platt(s_in, y_in, s_te):
    m = LogisticRegression(penalty=None).fit(lg(s_in)[:, None], y_in); return m.predict_proba(lg(s_te)[:, None])[:, 1], m.coef_[0, 0]


def fuse_fold(inner, test, tr, te, members):
    """Calibrated sum rule: Platt per member on inner OOF of tr, equal-weight mean of calibrated test posteriors."""
    cal, slopes = [], []
    for m in members:
        p, b = platt(inner[m][tr], Y[tr], test[m]); cal.append(p); slopes.append(b)
    return np.mean(cal, 0), slopes


def stack_fold(inner, test, tr, members):
    Xi = np.column_stack([lg(inner[m][tr]) for m in members]); Xt = np.column_stack([lg(test[m]) for m in members])
    m = ek.nested_lr().fit(Xi, Y[tr]); return m.predict_proba(Xt)[:, 1], m[-1].coef_[0]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); g = ap.add_mutually_exclusive_group(required=True); g.add_argument("--inner", action="store_true"); g.add_argument("--eval", action="store_true")
    a = ap.parse_args(); c17.header("round19_fusion")
    if a.inner:
        C, pid = p3.load_cache(12)
        out = Parallel(n_jobs=12, max_nbytes="1M")(delayed(cpu_inner)(C, pid, r, k, tr) for r, k, tr, _ in J)
        O = {m: np.vstack([o[m] for o in out]) for m in ("I2", "C1", "P3")}
        ok = all(np.isnan(O[m][f, te]).all() and not np.isnan(O[m][f, tr]).any() for m in O for f, (_, _, tr, te) in enumerate(J))
        c17.pf(ok, "inner OOF: every training patient has exactly one inner score per outer fold; test patients have none")
        np.savez(f"{OUT}/cpu_inner.npz", **O); sys.exit()
    # ------------------------------------------------------------------ eval
    M = f18.load_members(); M["C1"] = np.load(f"{c17.OUT}/p1.npz")["C1"]
    O = dict(np.load(f"{OUT}/cpu_inner.npz")); O["NM"] = np.full((50, N), np.nan)
    for f, (r, k, tr, te) in enumerate(J):
        d = json.load(open(f"{OUT}/nm_inner_r{r}_k{k}.json"))["oof"]; assert len(d) == len(tr) and not set(d) & set(SUBJ[te])
        O["NM"][f, tr] = [d[s] for s in SUBJ[tr]]
    ok = [c17.pf(abs(c17.pooled_auc(M[m]) - v) < 1e-5, f"outer test scores {m} reproduce {v}") for m, v in [("I2", 0.71210), ("P3", 0.68318), ("NM", 0.70509)]]
    ok.append(c17.pf(abs(c17.pooled_auc(f18.fuse(M, ["I2", "P3", "NM"])) - 0.72709) < 1e-5, "F2 (round 18) reproduces 0.72709"))
    ok.append(c17.pf(all(not np.isnan(O[m][f, tr]).any() and np.isnan(O[m][f, te]).all() for m in O for f, (_, _, tr, te) in enumerate(J)), "inner OOF complete for I2, C1, P3, NM"))
    if not all(ok):
        sys.exit("STOP")
    P = {"I2": M["I2"], "F2 (round 18)": f18.fuse(M, ["I2", "P3", "NM"])}; P.update({m: np.full((10, N), np.nan) for m in ("CF2", "SF2", "CF-C1")})
    slopes, weights, cal_auc_ok = [], [], True
    for f, (r, k, tr, te) in enumerate(J):
        inner = {m: O[m][f] for m in O}; test = {m: M[m][r, te] for m in ("I2", "C1", "P3", "NM")}
        P["CF2"][r, te], s = fuse_fold(inner, test, tr, te, ["I2", "P3", "NM"]); slopes.append(s)
        P["CF-C1"][r, te], _ = fuse_fold(inner, test, tr, te, ["I2", "C1"])
        P["SF2"][r, te], w = stack_fold(inner, test, tr, ["I2", "P3", "NM"]); weights.append(w)
        for m in ("I2", "P3", "NM"):                                                       # Platt is monotone -> member AUC unchanged
            pc, _ = platt(inner[m][tr], Y[tr], test[m]); cal_auc_ok &= abs(ek.roc_auc_score(Y[te], pc) - ek.roc_auc_score(Y[te], test[m])) < 1e-12
    slopes, weights = np.array(slopes), np.array(weights)
    c17.pf(bool((slopes > 0).all()) and cal_auc_ok, f"all Platt slopes > 0 (min {slopes.min():.3f}) and calibrated member AUC == raw member AUC in every fold")
    r0, _, tr0, te0 = J[0]; i0 = {m: O[m][0] for m in O}
    man = np.mean([LogisticRegression(penalty=None).fit(lg(i0[m][tr0])[:, None], Y[tr0]).predict_proba(lg(M[m][r0, te0])[:, None])[:, 1] for m in ("I2", "P3", "NM")], 0)
    c17.pf(np.allclose(man, P["CF2"][r0, te0]), "CF2 on fold r0k0 equals a hand-written Platt-then-mean")
    print("inner-OOF AUC (pooled over the 160 training patients, mean over 50 outer folds): "
          + ", ".join(f"{m} {np.mean([ek.roc_auc_score(Y[tr], O[m][f, tr]) for f, (_, _, tr, _) in enumerate(J)]):.4f}" for m in ("I2", "C1", "P3", "NM")))
    print(f"Platt slopes on logit (mean over folds): I2 {slopes[:, 0].mean():.3f}, P3 {slopes[:, 1].mean():.3f}, NM {slopes[:, 2].mean():.3f}")
    print(f"stacking standardised weights (mean ± sd over folds): I2 {weights[:, 0].mean():.3f} ± {weights[:, 0].std():.3f}, P3 {weights[:, 1].mean():.3f} ± {weights[:, 1].std():.3f}, "
          f"NM {weights[:, 2].mean():.3f} ± {weights[:, 2].std():.3f}; all positive in {(weights > 0).all(1).sum()}/50 folds")
    P["Metadata floor"] = c17.to_matrix(J, Parallel(n_jobs=12)(delayed(r9.lr_acq)(tr, te) for _, _, tr, te in J))
    FA = {m: c17.fold_aucs(J, P[m]) for m in P}
    pts = {m: r14.rep_metrics(P[m], np.arange(N)) for m in P}
    CI = dict(zip(P, Parallel(n_jobs=12)(delayed(r14.boot)(P[m], 19100 + i) for i, m in enumerate(P))))
    prim = [("CF2", "I2"), ("SF2", "I2")]; pr = [ek.corrected_ttest(FA[x], FA[y_]) for x, y_ in prim]; hh = f18.holm([p for _, p in pr])
    sec = [("CF2", "F2 (round 18)"), ("SF2", "CF2"), ("CF-C1", "I2")]
    rows = []
    for m in P:
        row = dict(method=m)
        for k_ in ["auc", "acc", "sens", "spec", "prec", "f1", "bac", "brier", "calib_slope"]:
            row[k_], row[k_ + "_lo"], row[k_ + "_hi"] = pts[m][k_], CI[m].loc[0.025, k_], CI[m].loc[0.975, k_]
        rows.append(row)
    T = pd.DataFrame(rows)
    for c_ in ["comparator", "dAUC", "p", "p_holm", "family"]:
        T[c_] = pd.Series([None] * len(T), dtype=object)
    for (x, y_), (d, p), h in zip(prim, pr, hh):
        T.loc[T.method == x, ["comparator", "dAUC", "p", "p_holm", "family"]] = [y_, d, p, h, "primary (Holm over 2)"]
    for x, y_ in sec:
        d, p = ek.corrected_ttest(FA[x], FA[y_]); T = pd.concat([T, pd.DataFrame([dict(method=f"{x} vs {y_}", comparator=y_, dAUC=d, p=p, family="secondary (descriptive)")])])
    T.to_csv(f"{c17.R}/round19_results.csv", index=False); np.savez(f"{c17.R}/round19_fold_aucs.npz", **FA)
    pd.DataFrame(np.column_stack([slopes, weights]), columns=["platt_I2", "platt_P3", "platt_NM", "w_I2", "w_P3", "w_NM"]).to_csv(f"{c17.R}/round19_calibration_weights.csv", index=False)
    f = lambda m, k_: f"{100 * pts[m][k_]:.1f} [{100 * CI[m].loc[0.025, k_]:.1f}-{100 * CI[m].loc[0.975, k_]:.1f}]"
    print("\n== strict P3, 200 patients, threshold 0.5, mean over 10 repeats [95% CI] ==")
    for m in P:
        print(f"{m:15s} AUC {f(m, 'auc')} | acc {f(m, 'acc')} | sens {f(m, 'sens')} | spec {f(m, 'spec')} | prec {f(m, 'prec')} | F1 {f(m, 'f1')} | BAC {f(m, 'bac')} | "
              f"Brier {pts[m]['brier']:.3f} | slope {pts[m]['calib_slope']:.2f}")
    print("\n== primary: corrected t-test 160/40, Holm over 2 ==")
    for (x, y_), (d, p), h in zip(prim, pr, hh):
        print(f"   {x} - {y_}: dAUC {d:+.4f}, p {p:.4f}, p_Holm {h:.4f}, folds better {(FA[x] > FA[y_]).sum()}/50")
    print("\n== secondary (descriptive) ==")
    for x, y_ in sec:
        d, p = ek.corrected_ttest(FA[x], FA[y_]); print(f"   {x} - {y_}: dAUC {d:+.4f}, p {p:.4f}, folds better {(FA[x] > FA[y_]).sum()}/50")
    W = r10.within(P, np.arange(N), c17.STRATA, [(m, "I2") for m in ("CF2", "SF2")] + [(m, "Metadata floor") for m in ("CF2", "SF2")], 19200)
    W.to_csv(f"{c17.R}/round19_within.csv", index=False)
    print("\n== within-stratum AUC (round10 estimator; Holm over this table) ==")
    print(W.assign(auc_a=[r_[f"auc_{r_['test'].split(' - ')[0]}"] for _, r_ in W.iterrows()], auc_b=[r_[f"auc_{r_['test'].split(' - ')[1]}"] for _, r_ in W.iterrows()])
          [["strata", "test", "auc_a", "auc_b", "diff", "lo", "hi", "p", "diff_repeat_mean", "p_holm"]].round(4).to_string(index=False))
