"""Round 3 parts B1 (band ablation), B2 (awake/clean windows) and E (C1 coefficients) on the strict P3 protocol
(200 patients, canonical folds, nested LR trained on all training patients as in the primary C1 analysis).
reports/round3_prediction.md, commit 0d442ba. CPU only."""
import os, numpy as np, pandas as pd, warnings
os.environ.setdefault("OMP_NUM_THREADS", "1")
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score
import evalkit as ek
warnings.filterwarnings("ignore"); D, R = ek.D, "/home/miftah_f/RTX-5090/tuep_fresh/reports"
T = np.load(f"{D}/p3_patient_table.npz", allow_pickle=True); subjects, y, BP, AS = T["subjects"], T["y"], T["bandpower"], T["agesex"]
CH = ["FP1", "FP2", "F7", "F3", "FZ", "F4", "F8", "T3", "C3", "CZ", "C4", "T4", "T5", "P3", "PZ", "P4", "T6", "O1", "O2"]
BANDS = ["delta", "theta", "alpha", "beta", "gamma", "logtot"]; col = lambda b: list(range(19 * BANDS.index(b), 19 * BANDS.index(b) + 19))
F = ek.folds(); pos = {s: i for i, s in enumerate(subjects)}
JOBS = [(r, k, np.setdiff1d(np.arange(len(y)), te), te) for r in range(10) for k in range(5)
        for te in [np.array([pos[s] for s in F[(F.repeat == r) & (F.fold == k)].subject])]]


def p3(X):
    out = Parallel(n_jobs=10)(delayed(lambda tr, te: ek.nested_lr().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1])(tr, te) for _, _, tr, te in JOBS)
    auc, fa = [], []
    for r in range(10):
        p = np.zeros(len(y))
        for (rr, _, _, te), o in zip(JOBS, out):
            if rr == r:
                p[te] = o
        auc.append(roc_auc_score(y, p))
    fa = [roc_auc_score(y[te], o) for (_, _, _, te), o in zip(JOBS, out)]
    return np.mean(auc), np.std(auc), fa


def window_table():
    """Same construction as baseline_clean.py (window -> recording -> first session -> patient)."""
    z = np.load(f"{D}/features_bandpower_windows.npz", allow_pickle=True); Xw, win_h5 = z["X"], z["h5_path"]
    rec = pd.read_csv(f"{D}/features_bandpower_recordings.csv"); rec = rec[rec.n_clean > 0].reset_index(drop=True)
    hdr = pd.read_csv(f"{D}/edf_headers.csv"); rec = rec.merge(hdr[["file"]], on="file")
    p = {h: i for i, h in enumerate(rec.h5_path)}; wr = np.array([p.get(h, -1) for h in win_h5]); Xw, wr = Xw[wr >= 0], wr[wr >= 0]
    first = (rec.groupby("subject").session_no.transform("min") == rec.session_no).values
    return Xw, wr, first, rec


if __name__ == "__main__":
    rows = []
    # ---------------- B1 band ablation
    sets = {"C1 full (all bands + age/sex)": col("delta") + col("theta") + col("alpha") + col("beta") + col("gamma") + col("logtot"),
            "without relative gamma": col("delta") + col("theta") + col("alpha") + col("beta") + col("logtot"),
            "without relative beta and gamma": col("delta") + col("theta") + col("alpha") + col("logtot"),
            "relative delta/theta/alpha only": col("delta") + col("theta") + col("alpha"),
            "log total power only": col("logtot"),
            "relative bands only": col("delta") + col("theta") + col("alpha") + col("beta") + col("gamma")}
    res = {k: p3(np.hstack([BP[:, c], AS])) for k, c in sets.items()}
    ref = res["C1 full (all bands + age/sex)"]
    print(("PASS" if abs(ref[0] - 0.66627) < 1e-5 else "FAIL") + f" C1 full reproduces the primary analysis (AUC {ref[0]:.5f} vs 0.66627)")
    print("== B1 band ablation (P3, 200 patients, + age/sex) ==")
    for k, (a, sd, fa) in res.items():
        d, p = ek.corrected_ttest(fa, ref[2]) if k != "C1 full (all bands + age/sex)" else (0, 1)
        print(f"{k:36s} AUC {a:.3f} ± {sd:.3f}   dAUC vs full {d:+.3f} (corrected p = {p:.3f})"); rows.append(dict(part="B1", set=k, AUC=a, AUC_sd=sd, dAUC=d, p=p))
    # ---------------- B2 awake/clean windows
    Xw, wr, first, rec = window_table()
    rebuilt = np.vstack([np.median(Xw[np.isin(wr, np.where(first & (rec.subject.values == s))[0])], axis=0) for s in subjects])
    assert np.allclose(rebuilt, BP), "window table does not reproduce the patient band-power table"
    print("\nPASS rebuilding patient features from all windows reproduces the P3 band-power table exactly")
    alpha_occ = Xw[:, [col("alpha")[CH.index("O1")], col("alpha")[CH.index("O2")]]].mean(1); gam = Xw[:, col("gamma")].mean(1)
    med = lambda v: pd.Series(v).groupby(wr).transform("median").values
    awake = (alpha_occ >= med(alpha_occ)) & (gam <= med(gam))
    sel_first = first[wr]
    print(f"awake/clean windows: {100 * awake[sel_first].mean():.1f}% of first-session windows")
    for name, mask in [("all windows", np.ones(len(wr), bool)), ("awake/clean windows", awake), ("other windows", ~awake)]:
        feats = []
        for s in subjects:
            m = mask & np.isin(wr, np.where(first & (rec.subject.values == s))[0])
            feats.append(np.median(Xw[m], axis=0) if m.any() else np.full(Xw.shape[1], np.nan))
        Xs = np.vstack(feats); assert np.isfinite(Xs).all(), f"patient without {name}"
        a, sd, fa = p3(np.hstack([Xs, AS])); d, p = ek.corrected_ttest(fa, ref[2])
        print(f"B2 {name:22s} AUC {a:.3f} ± {sd:.3f}   dAUC vs C1 full {d:+.3f} (corrected p = {p:.3f})"); rows.append(dict(part="B2", set=name, AUC=a, AUC_sd=sd, dAUC=d, p=p))
    pd.DataFrame(rows).to_csv(f"{R}/round3_confound.csv", index=False)
    # ---------------- E coefficients (standardised), averaged over the 50 training folds
    X = np.hstack([BP, AS])
    coefs = np.vstack(Parallel(n_jobs=10)(delayed(lambda tr: ek.nested_lr().fit(X[tr], y[tr])[-1].coef_[0])(tr) for _, _, tr, _ in JOBS)).mean(0)
    C = pd.DataFrame(coefs[:114].reshape(6, 19), index=BANDS, columns=CH)
    print("\n== E: mean standardised coefficient of C1 (positive = more epilepsy), 50 training folds ==")
    print(C.round(3).to_string()); print("age, male:", np.round(coefs[114:], 3))
    print("\nby band: sum |coef|", C.abs().sum(1).round(3).to_dict()); print("by band: signed mean", C.mean(1).round(3).to_dict())
    C.to_csv(f"{R}/round3_c1_coefficients.csv")
