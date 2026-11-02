"""Round 18 (reports/round18_prediction.md, commit 22cea25): multi-view normative fusion by the equal-weight sum rule (Kittler et al. 1998).
Members are FROZEN out-of-fold predictions already produced on the canonical folds (no retraining):
I2 (spectral, round17 p1.npz), P3 (spatial, Riemannian re-centring per era, round17 p3.npz), NM (temporal, NormMamba-F, round16 JSON),
LA (LaBraM frozen + era adapter, pretrained, round17 p5.npz), RR (Riemann without normative layer, ablation).
Primary: probability mean; secondary: within-fold rank mean (AUC only). Holm over 3 (F1-I2, F2-I2, F3-I2). CPU only."""
import round17_common as c17
import json, numpy as np, pandas as pd
from joblib import Parallel, delayed
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
import evalkit as ek, round9 as r9, round10 as r10, round14 as r14

J, Y, N, SUBJ = c17.JOBS, c17.Y, c17.N, c17.SUBJ
REPORTED = {"I2": 0.71210, "P3": 0.68318, "NM": 0.70509, "LA": 0.64534, "RR": 0.61159}
MODELS = {"F1": ["I2", "P3"], "F2": ["I2", "P3", "NM"], "F3": ["I2", "LA"], "F-NM": ["I2", "NM"], "F1-RR": ["I2", "RR"]}
TRACK = {"F1": "from scratch", "F2": "from scratch", "F3": "hybrid (from scratch + pretrained)", "F-NM": "from scratch", "F1-RR": "from scratch"}


def load_members():
    M = {"I2": np.load(f"{c17.OUT}/p1.npz")["I2"], "P3": np.load(f"{c17.OUT}/p3.npz")["P3"], "RR": np.load(f"{c17.OUT}/p3.npz")["Riemann_raw"],
         "LA": np.load(f"{c17.OUT}/p5.npz")["P5"]}
    nm = np.full((10, N), np.nan)
    for r, k, _, te in J:
        d = json.load(open(f"{ek.D}/round16/r{r}_k{k}.json"))["F"]["test"]; nm[r, te] = [d[s] for s in SUBJ[te]]
    M["NM"] = nm; return M


def fuse(M, members):
    return np.mean([M[m] for m in members], 0)


def rank_fuse(M, members):
    out = np.full((10, N), np.nan)
    for r, _, _, te in J:
        out[r, te] = np.mean([(rankdata(M[m][r, te]) - 1) / (len(te) - 1) for m in members], 0)
    return out


def partner_shuffle(M, members, seed):
    rng = np.random.default_rng(seed); S = {m: M[m].copy() for m in members}
    for r, _, _, te in J:
        for m in members[1:]:
            S[m][r, te] = M[m][r, te][rng.permutation(len(te))]
    return c17.pooled_auc(fuse(S, members))


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); h = np.empty(len(p)); run = 0
    for k, i in enumerate(o):
        run = max(run, min(1, p[i] * (len(p) - k))); h[i] = run
    return h


if __name__ == "__main__":
    c17.header("round18_fusion")
    M = load_members(); ok = []
    for m, v in REPORTED.items():
        a = c17.pooled_auc(M[m]); ok.append(c17.pf(abs(a - v) < 1e-5 and not np.isnan(M[m]).any(), f"member {m}: AUC {a:.5f} == reported {v} and no NaN"))
    ok.append(c17.pf(all((~np.isnan(M[m])).sum(1).tolist() == [N] * 10 for m in M), "every patient has exactly one score per repeat for every member"))
    r0, _, _, te0 = J[0]; man = np.array([(M["I2"][r0, i] + M["P3"][r0, i] + M["NM"][r0, i]) / 3 for i in te0])
    ok.append(c17.pf(np.allclose(fuse(M, MODELS["F2"])[r0, te0], man), "F2 on fold r0k0 equals a hand-written arithmetic mean"))
    ok.append(c17.pf(np.array_equal(fuse(M, ["I2", "I2"]), M["I2"]), "identity: fusing I2 with itself returns I2"))
    if not all(ok):
        raise SystemExit("STOP: checks failed")
    P = {"I2": M["I2"], "C1": np.load(f"{c17.OUT}/p1.npz")["C1"]}
    P.update({k: fuse(M, v) for k, v in MODELS.items()}); P.update({"P3": M["P3"], "NormMamba-F": M["NM"], "LaBraM-adapter": M["LA"]})
    P["Metadata floor"] = c17.to_matrix(J, Parallel(n_jobs=12)(delayed(r9.lr_acq)(tr, te) for _, _, tr, te in J))
    RK = {f"{k}-rank": rank_fuse(M, v) for k, v in MODELS.items() if k in ("F1", "F2", "F3")}
    FA = {m: c17.fold_aucs(J, P[m]) for m in P}; FA.update({m: c17.fold_aucs(J, RK[m]) for m in RK})
    pts = {m: r14.rep_metrics(P[m], np.arange(N)) for m in P}
    CI = dict(zip(P, Parallel(n_jobs=12)(delayed(r14.boot)(P[m], 18100 + i) for i, m in enumerate(P))))
    prim = [("F1", "I2"), ("F2", "I2"), ("F3", "I2")]; pr = [ek.corrected_ttest(FA[a], FA[b]) for a, b in prim]; hh = holm([p for _, p in pr])
    sec = [("F2", "F1"), ("F-NM", "I2"), ("F1", "F1-RR"), ("F1-rank", "I2"), ("F2-rank", "I2"), ("F3-rank", "I2")]
    rows = []
    for m in P:
        row = dict(method=m, track=TRACK.get(m, {"LaBraM-adapter": "pretrained", "Metadata floor": "non-EEG"}.get(m, "from scratch")),
                   members="+".join(MODELS.get(m, [m])))
        for k_ in ["auc", "acc", "sens", "spec", "prec", "f1", "bac", "brier", "calib_slope"]:
            row[k_], row[k_ + "_lo"], row[k_ + "_hi"] = pts[m][k_], CI[m].loc[0.025, k_], CI[m].loc[0.975, k_]
        rows.append(row)
    for m in RK:
        rows.append(dict(method=m, track=TRACK[m.replace("-rank", "")], members="+".join(MODELS[m.replace("-rank", "")]) + " (rank mean)", auc=c17.pooled_auc(RK[m])))
    T = pd.DataFrame(rows)
    for c_ in ["comparator", "dAUC", "p", "p_holm", "family"]:
        T[c_] = pd.Series([None] * len(T), dtype=object)
    for (a, b), (d, p), h in zip(prim, pr, hh):
        T.loc[T.method == a, ["comparator", "dAUC", "p", "p_holm", "family"]] = [b, d, p, h, "primary (Holm over 3)"]
    for a, b in sec:
        d, p = ek.corrected_ttest(FA[a], FA[b]); T = pd.concat([T, pd.DataFrame([dict(method=f"{a} vs {b}", comparator=b, dAUC=d, p=p, family="secondary (descriptive)")])])
    T.to_csv(f"{c17.R}/round18_results.csv", index=False); np.savez(f"{c17.R}/round18_fold_aucs.npz", **FA)
    f = lambda m, k_: f"{100 * pts[m][k_]:.1f} [{100 * CI[m].loc[0.025, k_]:.1f}-{100 * CI[m].loc[0.975, k_]:.1f}]"
    print("\n== strict P3, 200 patients, threshold 0.5, mean over 10 repeats [95% CI] ==")
    for m in P:
        print(f"{m:15s} AUC {f(m, 'auc')} | acc {f(m, 'acc')} | sens {f(m, 'sens')} | spec {f(m, 'spec')} | prec {f(m, 'prec')} | F1 {f(m, 'f1')} | BAC {f(m, 'bac')} | "
              f"Brier {pts[m]['brier']:.3f} | slope {pts[m]['calib_slope']:.2f}")
    print("rank-mean variants (AUC only): " + " | ".join(f"{m} {c17.pooled_auc(RK[m]):.4f}" for m in RK))
    print("\n== primary: corrected t-test 160/40, Holm over 3 ==")
    for (a, b), (d, p), h in zip(prim, pr, hh):
        print(f"   {a} - {b}: dAUC {d:+.4f}, p {p:.4f}, p_Holm {h:.4f}, folds better {(FA[a] > FA[b]).sum()}/50")
    print("\n== secondary (descriptive) ==")
    for a, b in sec:
        d, p = ek.corrected_ttest(FA[a], FA[b]); print(f"   {a} - {b}: dAUC {d:+.4f}, p {p:.4f}, folds better {(FA[a] > FA[b]).sum()}/50")
    W = r10.within(P, np.arange(N), c17.STRATA, [(m, "I2") for m in ("F1", "F2", "F3")] + [(m, "Metadata floor") for m in ("F1", "F2", "F3")], 18200)
    W.to_csv(f"{c17.R}/round18_within.csv", index=False)
    print("\n== within-stratum AUC (round10 estimator; Holm over this table) ==")
    print(W.assign(auc_a=[r[f"auc_{r['test'].split(' - ')[0]}"] for _, r in W.iterrows()], auc_b=[r[f"auc_{r['test'].split(' - ')[1]}"] for _, r in W.iterrows()])
          [["strata", "test", "auc_a", "auc_b", "diff", "lo", "hi", "p", "diff_repeat_mean", "p_holm"]].round(4).to_string(index=False))
    neg = []
    for m in ("F1", "F2"):
        s = np.array(Parallel(n_jobs=12)(delayed(partner_shuffle)(M, MODELS[m], 18000 + i) for i in range(20))); obs = c17.pooled_auc(P[m])
        neg.append(dict(model=m, control="partner scores permuted within fold 20x", observed_auc=obs, null_mean=s.mean(), null_sd=s.std(ddof=1),
                        null_p95=np.percentile(s, 95), null_max=s.max(), observed_gt_p95=bool(obs > np.percentile(s, 95)), rank_p=(1 + (s >= obs).sum()) / 21))
    keep = c17.KEEP169
    for m in ("I2", "F1", "F2", "F3"):
        neg.append(dict(model=m, control="same predictions evaluated on the 169 patients without <=2009 (no retraining)",
                        observed_auc=float(np.mean([roc_auc_score(Y[keep], p[keep]) for p in P[m]]))))
    NG = pd.DataFrame(neg); NG.to_csv(f"{c17.R}/round18_negative_controls.csv", index=False)
    print("\n== negative controls =="); print(NG.round(4).to_string(index=False))
