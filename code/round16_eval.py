"""Round 16 evaluation (reports/round16_normmamba_prediction.md, commit e2b8b14). Reads data/round16/r*_k*.json.
Metrics as reported by TUEP studies (AUC, accuracy, sensitivity, specificity, precision, F1, BAC) with patient-bootstrap CIs;
corrected t-tests (160/40) with Holm over {F vs I2, F vs C1, F vs raw}; within-year-stratum AUC (round10 estimator). CPU only."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ["CUDA_VISIBLE_DEVICES"] = ""
import json, numpy as np, pandas as pd, warnings
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score
import evalkit as ek, round4 as r4, round9 as r9, round10 as r10, round14 as r14
warnings.filterwarnings("ignore"); R = "/home/miftah_f/RTX-5090/tuep_fresh/reports"; D = "/home/miftah_f/RTX-5090/tuep_fresh/data/round16"
Y, N, SUBJ = r4.Y, len(r4.Y), r4.SUBJ; POS = {s: i for i, s in enumerate(SUBJ)}
JOBS = r14.r7.J200


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); h = np.empty(len(p)); run = 0
    for k, i in enumerate(o):
        run = max(run, min(1, p[i] * (len(p) - k))); h[i] = run
    return h


if __name__ == "__main__":
    res = {(r, k): json.load(open(f"{D}/r{r}_k{k}.json")) for r, k, _, _ in JOBS}
    assert len(res) == 50
    P, FA = {}, {}
    for v in ["F", "S", "raw"]:
        preds = [np.array([res[(r, k)][v]["test"][s] for s in SUBJ[te]]) for r, k, _, te in JOBS]
        P[f"NormMamba-{v}" if v != "raw" else "Mamba-raw"] = r14.to_matrix(JOBS, preds)
    for name, m in [("I2", "I2"), ("C1", "C1")]:
        out = Parallel(n_jobs=12)(delayed(r4.fit_predict)(m, tr, te, 0) for _, _, tr, te in JOBS); P[name] = r14.to_matrix(JOBS, out)
    with np.load(f"{ek.D}/p3_patient_table.npz", allow_pickle=True) as T:
        acq = T["acq"]
    P["Metadata floor"] = r14.to_matrix(JOBS, Parallel(n_jobs=12)(delayed(lambda tr, te: ek.nested_lr().fit(acq[tr], Y[tr]).predict_proba(acq[te])[:, 1])(tr, te) for _, _, tr, te in JOBS))
    for m in P:
        FA[m] = np.array([roc_auc_score(Y[te], P[m][r][te]) for r, _, _, te in JOBS])
    pts = {m: r14.rep_metrics(P[m], np.arange(N)) for m in P}
    print(("PASS" if abs(pts["I2"].auc - 0.71210) < 1e-5 and abs(pts["C1"].auc - 0.66627) < 1e-5 else "FAIL") + f" I2 {pts['I2'].auc:.5f}, C1 {pts['C1'].auc:.5f} reproduced")
    CI = dict(zip(P, Parallel(n_jobs=6)(delayed(r14.boot)(P[m], 16000 + i) for i, m in enumerate(P))))
    rows = []
    for m in P:
        row = {"method": m}
        for k_ in ["auc", "acc", "sens", "spec", "prec", "f1", "bac", "brier"]:
            row[k_], row[k_ + "_lo"], row[k_ + "_hi"] = pts[m][k_], CI[m].loc[0.025, k_], CI[m].loc[0.975, k_]
        rows.append(row)
    T = pd.DataFrame(rows).set_index("method"); T.to_csv(f"{R}/round16_results.csv")
    fmt = lambda m, k_: f"{100 * T.loc[m, k_]:.1f} [{100 * T.loc[m, k_ + '_lo']:.1f}–{100 * T.loc[m, k_ + '_hi']:.1f}]"
    print("\n== strict P3, 200 patients, threshold 0.5, mean over 10 repeats [95% CI] ==")
    for m in P:
        print(f"{m:15s} AUC {fmt(m, 'auc')} | acc {fmt(m, 'acc')} | sens {fmt(m, 'sens')} | spec {fmt(m, 'spec')} | prec {fmt(m, 'prec')} | F1 {fmt(m, 'f1')} | BAC {fmt(m, 'bac')}")
    tests = [("NormMamba-F", "I2"), ("NormMamba-F", "C1"), ("NormMamba-F", "Mamba-raw")]
    tt = [ek.corrected_ttest(FA[a], FA[b]) for a, b in tests]; h = holm([p for _, p in tt])
    print("\n== corrected t-test 160/40, Holm over 3 ==")
    for (a, b), (d, p), hh in zip(tests, tt, h):
        print(f"   {a} - {b}: dAUC {d:+.4f}, p {p:.4f}, p_Holm {hh:.4f}")
    for a, b in [("NormMamba-S", "I2"), ("NormMamba-F", "NormMamba-S")]:
        d, p = ek.corrected_ttest(FA[a], FA[b]); print(f"   (descriptive) {a} - {b}: dAUC {d:+.4f}, p {p:.4f}")
    W = r10.within({m: P[m] for m in ["NormMamba-F", "I2", "Metadata floor"]}, np.arange(N), {"year": r4.YEAR_G}, [("NormMamba-F", "Metadata floor"), ("NormMamba-F", "I2")], 16100)
    print("\n== within-year-stratum AUC (round10 estimator) =="); print(W.round(4).to_string(index=False))
    ch = pd.Series([tuple(res[(r, k)]["F"]["chosen"]) for r, k, _, _ in JOBS]).value_counts()
    print("\nchosen (d, L) for NormMamba-F over 50 folds:", ch.to_dict(), "| median epochs:", np.median([e for v in res.values() for e in v["F"]["epochs"]]))
    pd.DataFrame([dict(test=f"{a} - {b}", dAUC=d, p=p, p_holm=hh) for (a, b), (d, p), hh in zip(tests, tt, h)]).to_csv(f"{R}/round16_tests.csv", index=False)
    W.to_csv(f"{R}/round16_within.csv", index=False); np.savez(f"{R}/round16_fold_aucs.npz", **FA)
