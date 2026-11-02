"""Round 17 aggregation (reports/round17_P1-P5_prediction.md Bagian 0 butir 10 and Bagian 9; choices in round17_implementation_notes.md).
Reads data/round17/p{1..5}.npz. Metrics with patient-bootstrap CIs (round14 helpers), Nadeau-Bengio corrected t-test (160/40) with Holm over
the 5 primary tests (P1-I2, P2-I2, P3-I2, P4-I2, P5-LaBraM-mentah), descriptive secondary tests, within-stratum AUC (round10 estimator),
negative-control summary. Pretrained track (P5) is reported separately. CPU only."""
import round17_common as c17
import numpy as np, pandas as pd
from joblib import Parallel, delayed
import evalkit as ek, round9 as r9, round10 as r10, round14 as r14

J, Y, N = c17.JOBS, c17.Y, c17.N
L = {i: np.load(f"{c17.OUT}/p{i}.npz") for i in range(1, 6)}
P = {"I2-repro": L[1]["I2"], "C1-repro": L[1]["C1"], "P1": L[1]["P1"], "P2": L[2]["P2"], "P2ekor-saja": L[2]["P2ekor"], "P2-tanpa-beta-gamma": L[2]["P2_noBG"],
     "P3": L[3]["P3"], "P3-global": L[3]["P3_global"], "Riemann-tanpa-normatif-repro": L[3]["Riemann_raw"],
     "P4": L[4]["P4"], "P4-ens": L[4]["P4_ens"], "P4-tanpa-total": L[4]["P4_nototal"]}
P["Metadata floor"] = c17.to_matrix(J, Parallel(n_jobs=12)(delayed(r9.lr_acq)(tr, te) for _, _, tr, te in J))
PT = {"P5-adapter": L[5]["P5"], "P5-global": L[5]["P5_global"], "P5-whiten": L[5]["P5_whiten"], "LaBraM-median": L[5]["LaBraM_median"],
      "LaBraM-mentah-repro": L[5]["LaBraM_raw"]}
TRACK = {**{m: "from scratch" for m in P}, **{m: "pretrained" for m in PT}}; TRACK["Metadata floor"] = "non-EEG"
ALLP = {**P, **PT}
# primary family (Holm over 5) and descriptive secondary comparisons
PRIMARY = [("P1", "I2-repro"), ("P2", "I2-repro"), ("P3", "I2-repro"), ("P4", "I2-repro"), ("P5-adapter", "LaBraM-mentah-repro")]
SECOND = [("P2ekor-saja", "I2-repro"), ("P2-tanpa-beta-gamma", "I2-repro"), ("P3", "Riemann-tanpa-normatif-repro"), ("P3-global", "Riemann-tanpa-normatif-repro"),
          ("P3", "P3-global"), ("P4-ens", "I2-repro"), ("P4-tanpa-total", "P4"), ("P5-global", "LaBraM-mentah-repro"), ("P5-whiten", "LaBraM-mentah-repro"),
          ("P5-adapter", "P5-global"), ("I2-repro", "C1-repro")]


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); h = np.empty(len(p)); run = 0
    for k, i in enumerate(o):
        run = max(run, min(1, p[i] * (len(p) - k))); h[i] = run
    return h


if __name__ == "__main__":
    c17.header("round17_eval")
    ok = [c17.pf(np.array_equal(L[1]["I2"], L[i]["I2"], equal_nan=True), f"I2 predictions identical in p1.npz and p{i}.npz") for i in (2, 4)]
    rep = {"C1-repro": 0.66627, "I2-repro": 0.71210}
    for m, v in rep.items():
        ok.append(c17.pf(abs(c17.pooled_auc(ALLP[m]) - v) < 1e-5, f"{m} == {v} ({c17.pooled_auc(ALLP[m]):.5f})"))
    ok.append(c17.pf(abs(c17.pooled_auc(ALLP["Riemann-tanpa-normatif-repro"]) - 0.612) < 0.01, f"Riemann-tanpa-normatif ~ 0.612 ({c17.pooled_auc(ALLP['Riemann-tanpa-normatif-repro']):.5f})"))
    ok.append(c17.pf(abs(c17.pooled_auc(ALLP["LaBraM-mentah-repro"]) - 0.640) < 0.01, f"LaBraM-mentah ~ 0.640 ({c17.pooled_auc(ALLP['LaBraM-mentah-repro']):.5f})"))
    ok.append(c17.pf(abs(c17.pooled_auc(P["Metadata floor"]) - 0.6479) < 5e-4, f"metadata floor == 0.6479 ({c17.pooled_auc(P['Metadata floor']):.5f})"))
    if not all(ok):
        raise SystemExit("STOP: reproduction failed")
    FA = {m: c17.fold_aucs(J, ALLP[m]) for m in ALLP}
    pts = {m: r14.rep_metrics(ALLP[m], np.arange(N)) for m in ALLP}
    CI = dict(zip(ALLP, Parallel(n_jobs=12)(delayed(r14.boot)(ALLP[m], 17300 + i) for i, m in enumerate(ALLP))))
    tests = {}
    pr = [ek.corrected_ttest(FA[a], FA[b]) for a, b in PRIMARY]; hh = holm([p for _, p in pr])
    for (a, b), (d, p), h in zip(PRIMARY, pr, hh):
        tests[a] = dict(comparator=b, dAUC=d, p=p, p_holm=h, family="primary (Holm over 5)")
    sec = {}
    for a, b in SECOND:
        d, p = ek.corrected_ttest(FA[a], FA[b]); sec[(a, b)] = (d, p)
        if a not in tests:
            tests[a] = dict(comparator=b, dAUC=d, p=p, p_holm=np.nan, family="secondary (descriptive)")
    rows = []
    for m in ALLP:
        row = dict(method=m, track=TRACK[m])
        for k_ in ["auc", "acc", "sens", "spec", "prec", "f1", "bac", "brier", "calib_slope"]:
            row[k_], row[k_ + "_lo"], row[k_ + "_hi"] = pts[m][k_], CI[m].loc[0.025, k_], CI[m].loc[0.975, k_]
        row.update(tests.get(m, dict(comparator="", dAUC=np.nan, p=np.nan, p_holm=np.nan, family="")))
        rows.append(row)
    T = pd.DataFrame(rows); T.to_csv(f"{c17.R}/round17_results.csv", index=False)
    FK = {k.replace("-repro", ""): v for k, v in FA.items()}; FK["P5"] = FA["P5-adapter"]; np.savez(f"{c17.R}/round17_fold_aucs.npz", **FK)
    f = lambda m, k_: f"{100 * T.set_index('method').loc[m, k_]:.1f} [{100 * T.set_index('method').loc[m, k_ + '_lo']:.1f}-{100 * T.set_index('method').loc[m, k_ + '_hi']:.1f}]"
    for track in ["from scratch", "non-EEG", "pretrained"]:
        print(f"\n== {track.upper()} track: strict P3, 200 patients, threshold 0.5, mean over 10 repeats [95% CI] ==")
        for m in [m for m in ALLP if TRACK[m] == track]:
            print(f"{m:30s} AUC {f(m, 'auc')} | acc {f(m, 'acc')} | sens {f(m, 'sens')} | spec {f(m, 'spec')} | prec {f(m, 'prec')} | F1 {f(m, 'f1')} | BAC {f(m, 'bac')} | "
                  f"Brier {T.set_index('method').loc[m, 'brier']:.3f} | slope {T.set_index('method').loc[m, 'calib_slope']:.2f}")
    print("\n== primary tests: corrected t-test 160/40, Holm over 5 ==")
    for (a, b), (d, p), h in zip(PRIMARY, pr, hh):
        print(f"   {a} - {b}: dAUC {d:+.4f}, p {p:.4f}, p_Holm {h:.4f}")
    print("\n== secondary (descriptive, no correction) ==")
    for (a, b), (d, p) in sec.items():
        print(f"   {a} - {b}: dAUC {d:+.4f}, p {p:.4f}")
    wt = [(a, b) for a, b in PRIMARY] + [(a, "Metadata floor") for a, _ in PRIMARY]
    W = r10.within(ALLP, np.arange(N), c17.STRATA, wt, 17400)
    W.to_csv(f"{c17.R}/round17_within.csv", index=False)
    print("\n== within-stratum AUC (round10 estimator; Holm over this table, secondary) ==")
    print(W.drop(columns=[c for c in W.columns if c.startswith("auc_")]).assign(auc_a=[r[f"auc_{r['test'].split(' - ')[0]}"] for _, r in W.iterrows()],
                                                                               auc_b=[r[f"auc_{r['test'].split(' - ')[1]}"] for _, r in W.iterrows()]).round(4).to_string(index=False))
    neg = []
    for i, (m, key, ref) in enumerate([("P1", "shuffle_auc", "P1"), ("P2", "shuffle_auc", "P2"), ("P3", "shuffle_auc", "P3"), ("P4", "perm_auc", "P4"), ("P5", "shuffle_auc", "P5")]):
        s = L[i + 1][key]; obs = c17.pooled_auc(L[i + 1][ref])
        neg.append(dict(experiment=m, control="label permutation (training labels) 20x" if key == "perm_auc" else "year groups shuffled 20x", observed_auc=obs,
                        null_mean=s.mean(), null_sd=s.std(ddof=1), null_p95=np.percentile(s, 95), null_max=s.max(), observed_gt_p95=bool(obs > np.percentile(s, 95)),
                        rank_p=(1 + (s >= obs).sum()) / (1 + len(s))))
    for i, (m, ref) in enumerate([("P1", "I2"), ("P2", "I2"), ("P3", "I2"), ("P4", "I2"), ("P5", "LaBraM_raw")]):
        a, b = L[i + 1][f"{m}_169"], L[i + 1][f"{ref}_169"]; d, p = ek.corrected_ttest(c17.fold_aucs(c17.J169, a), c17.fold_aucs(c17.J169, b), 135, 34)
        neg.append(dict(experiment=m, control=f"without <=2009 (169 patients), vs {ref}", observed_auc=c17.pooled_auc(a), comparator_auc=c17.pooled_auc(b), dAUC=d, p=p))
    NG = pd.DataFrame(neg); NG.to_csv(f"{c17.R}/round17_negative_controls.csv", index=False)
    print("\n== negative controls =="); print(NG.round(4).to_string(index=False))
