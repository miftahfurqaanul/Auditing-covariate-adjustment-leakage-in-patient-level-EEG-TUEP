"""Round 22 evaluation (reports/round22_prediction.md, commit 95ad5a4). CPU only.
--combined : part C. Combined test of the normative layer over three views (per-fold mean of I2-C1, NM-MambaRaw, P3-RiemannRaw; one
             Nadeau-Bengio test) and F2 vs F2-raw (same fusion without any normative layer). Holm over 2.
--final    : parts A and B from data/round22/<condition>/ (written by round22_nm.py)."""
import round17_common as c17
import sys, json, argparse, numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score
import evalkit as ek, round4 as r4, round9 as r9, round10 as r10, round14 as r14, round18_fusion as f18, round22_nm as r22

J, J169, Y, N, SUBJ = c17.JOBS, c17.J169, c17.Y, c17.N, c17.SUBJ


def nm_matrix(folder, variant, J_):
    P = np.full((10, N), np.nan)
    for r, k, _, te in J_:
        d = json.load(open(f"{folder}/r{r}_k{k}.json"))[variant]["test"]; P[r, te] = [d[SUBJ[i]] for i in te]
    return P


def metrics_table(P, seed):
    pts = {m: r14.rep_metrics(P[m], np.where(~np.isnan(P[m][0]))[0]) for m in P}
    CI = dict(zip(P, Parallel(n_jobs=12)(delayed(r14.boot)(P[m], seed + i) for i, m in enumerate(P)))); rows = []
    for m in P:
        row = dict(method=m)
        for k_ in ["auc", "acc", "sens", "spec", "prec", "f1", "bac", "brier", "calib_slope"]:
            row[k_], row[k_ + "_lo"], row[k_ + "_hi"] = pts[m][k_], CI[m].loc[0.025, k_], CI[m].loc[0.975, k_]
        rows.append(row)
    return pd.DataFrame(rows)


def show(T):
    f = lambda r_, k_: f"{100 * r_[k_]:.1f} [{100 * r_[k_ + '_lo']:.1f}-{100 * r_[k_ + '_hi']:.1f}]"
    for _, r_ in T.iterrows():
        print(f"{r_['method']:16s} AUC {f(r_, 'auc')} | acc {f(r_, 'acc')} | sens {f(r_, 'sens')} | spec {f(r_, 'spec')} | prec {f(r_, 'prec')} | F1 {f(r_, 'f1')} | "
              f"BAC {f(r_, 'bac')} | Brier {r_['brier']:.3f} | slope {r_['calib_slope']:.2f}")


def within_show(W):
    print(W.assign(auc_a=[r_[f"auc_{r_['test'].split(' - ')[0]}"] for _, r_ in W.iterrows()], auc_b=[r_[f"auc_{r_['test'].split(' - ')[1]}"] for _, r_ in W.iterrows()])
          [["strata", "test", "auc_a", "auc_b", "diff", "lo", "hi", "p", "diff_repeat_mean", "p_holm"]].round(4).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); g = ap.add_mutually_exclusive_group(required=True); g.add_argument("--combined", action="store_true"); g.add_argument("--final", action="store_true")
    a = ap.parse_args(); c17.header("round22_eval")
    # identity of the re-implemented folds/shuffles in round22_nm.py (it cannot import round17_common/round10)
    ok = [c17.pf(all(np.array_equal(x[2], y[2]) and np.array_equal(x[3], y[3]) for x, y in zip(r22.J169, J169)) and all(np.array_equal(x[3], y[3]) for x, y in zip(r22.J200, J)),
                 "round22_nm folds == round17_common.JOBS and J169"),
          c17.pf(all(np.array_equal(r22.shuffled_year(i), c17.shuffled_year(i)) and np.array_equal(r22.year_design(r22.shuffled_year(i)), c17.year_design(c17.shuffled_year(i))) for i in range(3)),
                 "round22_nm shuffles and designs == round17_common (same permutations as round 17)")]
    M = {"I2": np.load(f"{c17.OUT}/p1.npz")["I2"], "C1": np.load(f"{c17.OUT}/p1.npz")["C1"], "P3": np.load(f"{c17.OUT}/p3.npz")["P3"],
         "RR": np.load(f"{c17.OUT}/p3.npz")["Riemann_raw"], "NM": nm_matrix(f"{ek.D}/round16", "F", J), "NMraw": nm_matrix(f"{ek.D}/round16", "raw", J)}
    for m, v in [("I2", 0.71210), ("C1", 0.66627), ("P3", 0.68318), ("RR", 0.61159), ("NM", 0.70509), ("NMraw", 0.64596)]:
        ok.append(c17.pf(abs(c17.pooled_auc(M[m]) - v) < 1e-5, f"{m} reproduces {v}"))
    if not all(ok):
        sys.exit("STOP")
    FA = {m: c17.fold_aucs(J, M[m]) for m in M}
    z16, z17 = np.load(f"{c17.R}/round16_fold_aucs.npz"), np.load(f"{c17.R}/round17_fold_aucs.npz")
    c17.pf(np.allclose(FA["NM"], z16["NormMamba-F"]) and np.allclose(FA["NMraw"], z16["Mamba-raw"]) and np.allclose(FA["I2"], z17["I2"]) and np.allclose(FA["C1"], z17["C1"])
           and np.allclose(FA["P3"], z17["P3"]) and np.allclose(FA["RR"], z17["Riemann-tanpa-normatif"]), "fold AUCs recomputed from predictions == saved fold AUCs (rounds 16, 17)")
    if a.combined:
        diffs = {"spectral (I2 - C1)": FA["I2"] - FA["C1"], "temporal (NM - Mamba-raw)": FA["NM"] - FA["NMraw"], "spatial (P3 - Riemann)": FA["P3"] - FA["RR"]}
        dmat = np.column_stack(list(diffs.values())); dbar = dmat.mean(1)
        print("\n== C1. per-view effect of the normative layer (fold-level dAUC; corrected t-test 160/40, descriptive) ==")
        for k_, v in diffs.items():
            print(f"   {k_:28s} mean {v.mean():+.4f} | p {ek.corrected_ttest(v, np.zeros(50))[1]:.4f} | folds > 0: {(v > 0).sum()}/50")
        print(f"   correlation between per-fold view effects: {np.round(np.corrcoef(dmat.T)[np.triu_indices(3, 1)], 3).tolist()}")
        d1, p1 = ek.corrected_ttest(dbar, np.zeros(50))
        P = {"F2": f18.fuse(M, ["I2", "P3", "NM"]), "F2-raw": f18.fuse(M, ["C1", "RR", "NMraw"])}
        P["Metadata floor"] = c17.to_matrix(J, Parallel(n_jobs=12)(delayed(r9.lr_acq)(tr, te) for _, _, tr, te in J))
        c17.pf(abs(c17.pooled_auc(P["F2"]) - 0.72709) < 1e-5, "F2 reproduces 0.72709")
        FA2 = {m: c17.fold_aucs(J, P[m]) for m in P}; d2, p2 = ek.corrected_ttest(FA2["F2"], FA2["F2-raw"]); hh = f18.holm([p1, p2])
        print("\n== primary (Holm over 2) ==")
        print(f"   C1 combined: mean over views per fold, dbar {d1:+.4f}, p {p1:.4f}, p_Holm {hh[0]:.4f}, folds with dbar > 0: {(dbar > 0).sum()}/50")
        print(f"   C2 F2 - F2-raw: dAUC {d2:+.4f}, p {p2:.4f}, p_Holm {hh[1]:.4f}, folds better {(FA2['F2'] > FA2['F2-raw']).sum()}/50")
        T = metrics_table(P, 22100); print("\n== strict P3, 200 patients, threshold 0.5 [95% CI] =="); show(T)
        W = r10.within(P, np.arange(N), c17.STRATA, [("F2", "F2-raw"), ("F2-raw", "Metadata floor"), ("F2", "Metadata floor")], 22200)
        print("\n== within-stratum AUC (round10 estimator; Holm over this table) =="); within_show(W)
        T.to_csv(f"{c17.R}/round22_combined_results.csv", index=False); W.to_csv(f"{c17.R}/round22_combined_within.csv", index=False)
        pd.DataFrame([dict(test="C1 combined normative-layer effect (mean of 3 views per fold)", dAUC=d1, p=p1, p_holm=hh[0], folds_positive=int((dbar > 0).sum())),
                      dict(test="C2 F2 - F2-raw", dAUC=d2, p=p2, p_holm=hh[1], folds_positive=int((FA2["F2"] > FA2["F2-raw"]).sum()))]).to_csv(f"{c17.R}/round22_combined_tests.csv", index=False)
        np.savez(f"{c17.R}/round22_combined_fold_aucs.npz", dbar=dbar, **{k_.split(" ")[0]: v for k_, v in diffs.items()}, F2=FA2["F2"], F2_raw=FA2["F2-raw"])
        sys.exit()
    # ------------------------------------------------------------------ final: parts A and B
    G = f"{ek.D}/round22"
    det = json.load(open(f"{G}/det/r0_k0.json"))["F"]; ref = json.load(open(f"{ek.D}/round16/r0_k0.json"))["F"]
    dd = np.array([det["test"][s] - ref["test"][s] for s in ref["test"]])
    print(f"\n== GPU determinism (fold r0k0 re-run vs round 16) ==\n   chosen {det['chosen']} vs {ref['chosen']}; max |prediction diff| {np.abs(dd).max():.2e}; "
          f"test AUC {det['test_auc']:.4f} vs {ref['test_auc']:.4f}")
    # A
    P169 = {"I2": np.load(f"{c17.OUT}/p1.npz")["I2_169"], "P3": np.load(f"{c17.OUT}/p3.npz")["P3_169"], "NM": nm_matrix(f"{G}/j169", "F", J169)}
    c17.pf(abs(c17.pooled_auc(P169["I2"]) - 0.6966) < 5e-5 and abs(c17.pooled_auc(P169["P3"]) - 0.6533) < 5e-5, "I2-169 and P3-169 reproduce round 17 (0.6966, 0.6533)")
    c17.pf(all(np.isnan(P169[m][:, ~c17.KEEP169]).all() for m in P169), "no <=2009 patient has a prediction in the 169-patient analysis")
    P169["F2"] = (P169["I2"] + P169["P3"] + P169["NM"]) / 3
    P169["Metadata floor"] = c17.to_matrix(J169, Parallel(n_jobs=12)(delayed(r9.lr_acq)(tr, te) for _, _, tr, te in J169))
    FA169 = {m: c17.fold_aucs(J169, P169[m]) for m in P169}
    tA = [(x, "I2") for x in ("F2", "NM")]; pA = [ek.corrected_ttest(FA169[x], FA169[y_], 135, 34) for x, y_ in tA]; hA = f18.holm([p for _, p in pA])
    print("\n== A. 169 patients without <=2009 (retrained), threshold 0.5 [95% CI] =="); TA = metrics_table(P169, 22300); show(TA)
    print("   tests (corrected t-test 135/34, Holm over 2):")
    for (x, y_), (d, p), h in zip(tA, pA, hA):
        print(f"   {x} - {y_}: dAUC {d:+.4f}, p {p:.4f}, p_Holm {h:.4f}, folds better {(FA169[x] > FA169[y_]).sum()}/50")
    idx = np.where(c17.KEEP169)[0]; WA = r10.within(P169, idx, {k_: v for k_, v in c17.STRATA.items()}, [("F2", "I2"), ("F2", "Metadata floor"), ("I2", "Metadata floor")], 22400)
    print("   within strata (descriptive):"); within_show(WA)
    # B
    S = {f"shuf{i}": nm_matrix(f"{G}/shuf{i}", "F", J) for i in range(3)}; FS = {k_: c17.fold_aucs(J, v) for k_, v in S.items()}; Fmean = np.mean(list(FS.values()), 0)
    tB = [(f"NM - NM-shuf{i}", FA["NM"], FS[f"shuf{i}"]) for i in range(3)] + [("NM - mean(NM-shuf)", FA["NM"], Fmean)]
    pB = [ek.corrected_ttest(x, y_) for _, x, y_ in tB]; hB = f18.holm([p for _, p in pB])
    print("\n== B. NormMamba-F with permuted year groups (200 patients) ==")
    for k_, v in S.items():
        print(f"   {k_}: pooled AUC {c17.pooled_auc(v):.4f}")
    print(f"   true-year NM pooled AUC {c17.pooled_auc(M['NM']):.4f}; mean shuffled {np.mean([c17.pooled_auc(v) for v in S.values()]):.4f}")
    for (n_, x, y_), (d, p), h in zip(tB, pB, hB):
        print(f"   {n_}: dAUC {d:+.4f}, p {p:.4f}, p_Holm {h:.4f}, folds better {(x > y_).sum()}/50")
    TA.to_csv(f"{c17.R}/round22_A_results.csv", index=False); WA.to_csv(f"{c17.R}/round22_A_within.csv", index=False)
    pd.DataFrame([dict(test=f"{x} - {y_}", dAUC=d, p=p, p_holm=h, part="A") for (x, y_), (d, p), h in zip(tA, pA, hA)]
                 + [dict(test=n_, dAUC=d, p=p, p_holm=h, part="B") for (n_, _, _), (d, p), h in zip(tB, pB, hB)]).to_csv(f"{c17.R}/round22_AB_tests.csv", index=False)
    np.savez(f"{c17.R}/round22_AB_fold_aucs.npz", **{f"A_{k_}": v for k_, v in FA169.items()}, **{f"B_{k_}": v for k_, v in FS.items()}, B_NM=FA["NM"],
             det_max_abs_diff=np.abs(dd).max())
