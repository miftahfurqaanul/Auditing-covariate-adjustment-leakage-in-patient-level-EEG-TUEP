"""Decision threshold + model choice made inside the training patients only (reports/threshold_prediction.md).
P4: official EEG-Bench split, one test evaluation, bootstrap CI and 1000x permutation of the whole procedure.
P3: same procedure on the canonical 10x5 folds."""
import os, sys, numpy as np, pandas as pd, warnings
os.environ.setdefault("OMP_NUM_THREADS", "1")
from joblib import Parallel, delayed
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, recall_score
import evalkit as ek, cbramod_eval as ce
warnings.filterwarnings("ignore")
D, R = ek.D, "/home/miftah_f/RTX-5090/tuep_fresh/reports"
N_PERM = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 1000
GRID = np.arange(1, 100)

def fit_lr(X, tr, te, y, groups):
    cv = list(GroupKFold(5).split(X[tr], y[tr], groups[tr])) if groups is not None else 5
    m = ek.nested_lr(); m.set_params(logisticregressioncv__cv=cv)
    return m.fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]

def fit_gbm(X, tr, te, y, groups):
    m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, class_weight="balanced", random_state=0)
    return m.fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]

def predict(cand, S, tr, te, y, groups, seed):
    if cand == "K1":
        return fit_lr(S["bp"], tr, te, y, groups)
    if cand == "K2":
        return fit_lr(S["bp_as"], tr, te, y, groups)
    if cand == "K3":
        return fit_gbm(S["bp"], tr, te, y, groups)
    L = S["cbr"]; best, ba = 0, -1                               # K4: CBraMod layer chosen on training units only
    cv = inner_cv(tr, y, groups, seed)
    for l in range(L.shape[1]):
        p = np.zeros(len(tr))
        for itr, ite in cv:
            p[ite] = fit_lr(np.hstack([S["bp"], L[:, l]]), tr[itr], tr[ite], y, groups)
        a = roc_auc_score(y[tr], p)
        if a > ba:
            best, ba = l, a
    return fit_lr(np.hstack([S["bp"], L[:, best]]), tr, te, y, groups)

def inner_cv(tr, y, groups, seed):
    if groups is not None:
        return list(GroupKFold(5).split(tr, y[tr], groups[tr]))
    return list(StratifiedKFold(5, shuffle=True, random_state=seed).split(tr, y[tr]))

def best_threshold(y, p):
    cands = np.percentile(p, GRID)
    scores = [balanced_accuracy_score(y, p > t) for t in cands]
    i = int(np.argmax(scores)); return cands[i], scores[i]

def procedure(S, y, tr, te, groups, seed, cands=("K1", "K2", "K3", "K4")):
    """Inner OOF per candidate -> threshold + inner BAC -> choose -> refit on all training -> test."""
    cv = inner_cv(tr, y, groups, seed); info = {}
    for c in cands:
        p = np.zeros(len(tr))
        for itr, ite in cv:
            p[ite] = predict(c, S, tr[itr], tr[ite], y, groups, seed + 1)
        info[c] = best_threshold(y[tr], p)
    chosen = max(info, key=lambda c: info[c][1])
    test = {c: predict(c, S, tr, te, y, groups, seed + 2) for c in cands}
    return chosen, info, test

def tables():
    subjects, y, P3, meta, lab_r, cbr_r, brec, bfeat = ce.tables()
    T = np.load(f"{D}/p3_patient_table.npz", allow_pickle=True)
    P3S = {"bp": T["bandpower"], "bp_as": np.hstack([T["bandpower"], T["agesex"]]), "cbr": P3["CBraMod"][:, :, :200]}
    hdr = pd.read_csv(f"{D}/edf_headers.csv")
    hdr["age"] = hdr.patient.str.extract(r"Age:(-?\d+)")[0].astype(float)
    hdr["male"] = (hdr.patient.str.extract(r"^\w+\s+([MF])\s")[0] == "M").astype(float)
    m4 = meta.merge(hdr[["file", "age", "male", "duration_s"]], on="file", how="left")
    z = np.load(f"{D}/features_bandpower_windows.npz", allow_pickle=True)
    pr_ = {h: i for i, h in enumerate(meta.h5_path)}; wr = np.array([pr_.get(h, -1) for h in z["h5_path"]]); Xw, wr = z["X"][wr >= 0], wr[wr >= 0]
    rbp = np.vstack([np.median(Xw[wr == i], 0) for i in range(len(meta))])
    P4S = {"bp": rbp, "bp_as": np.hstack([rbp, m4[["age", "male"]].values.astype(float)]), "cbr": cbr_r}
    return subjects, y, P3S, m4, P4S

if __name__ == "__main__":
    subjects, y, P3S, m4, P4S = tables()
    # ---------------------------------------------------------------- P4 official split, one test evaluation
    test_subj = ek.eegbench_test_subjects(); keep = (m4.duration_s >= 60).values
    subj_r, y_r = m4.subject.values, m4.label.values
    tr4 = np.where(keep & ~np.isin(subj_r, list(test_subj)))[0]; te4 = np.where(keep & np.isin(subj_r, list(test_subj)))[0]
    chosen, info, test = procedure(P4S, y_r, tr4, te4, subj_r, 0)
    yt, st = y_r[te4], subj_r[te4]; rng = np.random.default_rng(0); us = np.unique(st); rows_of = {s: np.where(st == s)[0] for s in us}
    def boot(pred):
        b = []
        for _ in range(2000):
            i = np.concatenate([rows_of[s] for s in rng.choice(us, len(us))])
            if len(np.unique(yt[i])) == 2:
                b.append(balanced_accuracy_score(yt[i], pred[i]))
        return np.percentile(b, [2.5, 97.5])
    print(f"== P4 EEG-Bench split: {len(tr4)} training files / 150 patients, {len(te4)} test files / 50 patients ==")
    rows = []
    for c in ["K1", "K2", "K3", "K4"]:
        p, (thr, inner_bac) = test[c], info[c]
        for tag, t in [("thr 0.5", 0.5), ("thr inner", thr)]:
            pred = p > t; lo, hi = boot(pred)
            r = dict(cand=c, threshold=tag, t=t, inner_BAC=inner_bac, AUC=roc_auc_score(yt, p), BAC=balanced_accuracy_score(yt, pred),
                     SENS=recall_score(yt, pred), SPEC=recall_score(1 - yt, 1 - pred), lo=lo, hi=hi, chosen=(c == chosen and tag == "thr inner"))
            rows.append(r)
            print(f"{c} {tag:9s} t={t:.3f}  inner BAC {inner_bac:.3f} | test AUC {r['AUC']:.3f}  BAC {r['BAC']:.3f} (95% CI {lo:.3f}-{hi:.3f})  SENS {r['SENS']:.3f}  SPEC {r['SPEC']:.3f}{'   <== chosen inside training' if r['chosen'] else ''}")
    sel = next(r for r in rows if r["chosen"])
    print(f"\nSELECTED PROCEDURE: {chosen}, test BAC {sel['BAC']:.3f} (95% CI {sel['lo']:.3f}-{sel['hi']:.3f}); target BENDR 0.740")
    # permutation: shuffle training labels at patient level, rerun the whole procedure, score on the real test labels
    pts_tr = np.unique(subj_r[tr4]); lab_pt = pd.Series(y_r[tr4], index=subj_r[tr4]).groupby(level=0).first().loc[pts_tr].values
    def one(i):
        g = np.random.default_rng(10_000 + i); perm = dict(zip(pts_tr, g.permutation(lab_pt)))
        yp = y_r.copy(); yp[tr4] = [perm[s] for s in subj_r[tr4]]
        ch, inf, tst = procedure(P4S, yp, tr4, te4, subj_r, i + 1, cands=("K1", "K2", "K3"))   # K4 omitted for cost; see report
        return balanced_accuracy_score(yt, tst[ch] > inf[ch][0])
    null = np.array(Parallel(n_jobs=20)(delayed(one)(i) for i in range(N_PERM)))
    pval = (1 + (null >= sel["BAC"]).sum()) / (1 + N_PERM)
    print(f"permutation ({N_PERM}x, training labels shuffled per patient, procedure without K4): null BAC mean {null.mean():.3f}, 95th pct {np.percentile(null, 95):.3f}, p = {pval:.4f}")
    pd.DataFrame(rows).assign(p_perm_selected=pval).to_csv(f"{R}/threshold_p4.csv", index=False)
    np.save(f"{R}/threshold_p4_null.npy", null)

    # ---------------------------------------------------------------- P3 canonical folds, BAC with inner threshold
    F = ek.folds(); pos = {s: i for i, s in enumerate(subjects)}; jobs = []
    for r in range(10):
        Fr = F[F.repeat == r]
        for k in range(5):
            te = np.array([pos[s] for s in Fr[Fr.fold == k].subject]); jobs.append((r, np.setdiff1d(np.arange(len(y)), te), te))
    res = Parallel(n_jobs=20)(delayed(procedure)(P3S, y, tr, te, None, 100 * r + i) for i, (r, tr, te) in enumerate(jobs))
    print("\n== P3 strict protocol, canonical 10x5 folds ==")
    out = {}
    for name in ["K1", "K2", "K3", "K4", "SELECTED"]:
        bac05, bacin, auc = [], [], []
        for r in range(10):
            p = np.zeros(len(y)); pr = np.zeros(len(y), bool)
            for (rr, tr, te), (ch, inf, tst) in zip(jobs, res):
                if rr == r:
                    c = ch if name == "SELECTED" else name
                    p[te] = tst[c]; pr[te] = tst[c] > inf[c][0]
            auc.append(roc_auc_score(y, p)); bac05.append(balanced_accuracy_score(y, p > 0.5)); bacin.append(balanced_accuracy_score(y, pr))
        out[name] = (np.mean(auc), np.mean(bac05), np.mean(bacin), np.std(bacin))
        print(f"{name:9s} AUC {out[name][0]:.3f} | BAC thr 0.5 {out[name][1]:.3f} | BAC inner thr {out[name][2]:.3f} ± {out[name][3]:.3f}")
    print("chosen per fold:", pd.Series([c for c, _, _ in res]).value_counts().to_dict())
    pd.DataFrame(out, index=["AUC", "BAC_0.5", "BAC_inner", "BAC_inner_sd"]).T.to_csv(f"{R}/threshold_p3.csv")
