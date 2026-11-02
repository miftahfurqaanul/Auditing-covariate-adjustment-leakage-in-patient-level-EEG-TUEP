"""Fusion of the main method (band power) with the strongest frozen pretrained model, and stacking
(reports/fusion_prediction.md). All choices are made inside the training patients."""
import os, sys, numpy as np, pandas as pd, warnings
os.environ.setdefault("OMP_NUM_THREADS", "1")
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, recall_score
import evalkit as ek, cbramod_eval as ce, threshold_select as ts
warnings.filterwarnings("ignore")
D, R = ek.D, "/home/miftah_f/RTX-5090/tuep_fresh/reports"
MEMBERS = ["M1", "M2", "M3", "M4"]

def member_pred(m, S, tr, te, y, groups, layers):
    if m == "M1":
        return ce.fp(S["C1"], tr, te, y, groups=groups)
    if m == "M2":
        return ce.fp(S["LaBraM"][:, layers["M2"]], tr, te, y, groups=groups)
    if m == "M3":
        return ce.fp(S["CBraMod"][:, layers["M3"]], tr, te, y, groups=groups)
    return ce.fp(S["BENDR"], tr, te, y, pca=128, groups=groups)

def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4); return np.log(p / (1 - p))

def procedure(S, y, tr, te, groups, seed):
    layers = {"M2": ce.pick_layer(S["LaBraM"], tr, y, seed, groups), "M3": ce.pick_layer(S["CBraMod"], tr, y, seed + 1, groups)}
    cv = (list(GroupKFold(5).split(tr, y[tr], groups[tr])) if groups is not None
          else list(StratifiedKFold(5, shuffle=True, random_state=seed + 2).split(tr, y[tr])))
    oof = {m: np.zeros(len(tr)) for m in MEMBERS}
    for itr, ite in cv:
        for m in MEMBERS:
            oof[m][ite] = member_pred(m, S, tr[itr], tr[ite], y, groups, layers)
    inner_auc = {m: roc_auc_score(y[tr], oof[m]) for m in MEMBERS}
    best_pre = max(["M2", "M3", "M4"], key=lambda m: inner_auc[m])
    stack = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000).fit(np.column_stack([logit(oof[m]) for m in MEMBERS]), y[tr])
    test = {m: member_pred(m, S, tr, te, y, groups, layers) for m in MEMBERS}
    out = dict(test)
    out["F-avg"] = np.mean([test[m] for m in MEMBERS], 0)
    out["F-best2"] = (test["M1"] + test[best_pre]) / 2
    out["F-stack"] = stack.predict_proba(np.column_stack([logit(test[m]) for m in MEMBERS]))[:, 1]
    # inner (training) versions for threshold selection
    inner = dict(oof); inner["F-avg"] = np.mean([oof[m] for m in MEMBERS], 0); inner["F-best2"] = (oof["M1"] + oof[best_pre]) / 2
    inner["F-stack"] = stack.predict_proba(np.column_stack([logit(oof[m]) for m in MEMBERS]))[:, 1]   # in-sample for the stacker: threshold only
    thr = {k: ts.best_threshold(y[tr], v)[0] for k, v in inner.items()}
    return out, thr, best_pre, layers, stack.coef_[0], np.corrcoef(np.vstack([oof[m] for m in MEMBERS]))

if __name__ == "__main__":
    subjects, y, P3, meta, lab_r, cbr_r, brec, bfeat = ce.tables()
    F = ek.folds(); pos = {s: i for i, s in enumerate(subjects)}; jobs = []
    for r in range(10):
        Fr = F[F.repeat == r]
        for k in range(5):
            te = np.array([pos[s] for s in Fr[Fr.fold == k].subject]); jobs.append((r, np.setdiff1d(np.arange(len(y)), te), te))
    res = Parallel(n_jobs=20)(delayed(procedure)(P3, y, tr, te, None, 100 * r + i) for i, (r, tr, te) in enumerate(jobs))
    keys = MEMBERS + ["F-avg", "F-best2", "F-stack"]; fa = {k: [] for k in keys}; rep = {k: [] for k in keys}
    for r in range(10):
        P = {k: np.zeros(len(y)) for k in keys}
        for (rr, tr, te), (out, *_ ) in zip(jobs, res):
            if rr == r:
                for k in keys:
                    P[k][te] = out[k]; fa[k].append(roc_auc_score(y[te], out[k]))
        for k in keys:
            rep[k].append(ek.metrics(y, P[k]))
    print("== P3 strict protocol, canonical 10x5 folds ==")
    rows = []
    for k in keys:
        a = np.array([m["AUC"] for m in rep[k]]); b = np.array([m["BAC"] for m in rep[k]])
        d, p = ek.corrected_ttest(fa[k], fa["M1"]) if k != "M1" else (0.0, 1.0)
        rows.append(dict(config=k, AUC=a.mean(), AUC_sd=a.std(), BAC=b.mean(), dAUC_vs_M1=d, p_vs_M1=p))
        print(f"{k:8s} AUC {a.mean():.3f} ± {a.std():.3f}  BAC {b.mean():.3f}   vs M1: dAUC {d:+.3f} p={p:.3f}")
    print("strongest pretrained chosen inside training:", pd.Series([x[2] for x in res]).value_counts().to_dict())
    print("mean stacking weights (M1..M4):", np.round(np.mean([x[4] for x in res], 0), 3))
    print("mean OOF prediction correlation (M1..M4):\n", np.round(np.mean([x[5] for x in res], 0), 2))
    pd.DataFrame(rows).to_csv(f"{R}/fusion_p3.csv", index=False)

    # ---------------------------------------------------------------- P4, one shot
    _, _, _, m4, P4S = ts.tables()
    ras = m4[["age", "male"]].values.astype(float)
    keep = np.isin(np.arange(len(meta)), brec); bmap = {r: j for j, r in enumerate(brec)}
    ben_r = np.vstack([bfeat[bmap[i]] if i in bmap else np.zeros(bfeat.shape[1], np.float32) for i in range(len(meta))])
    S4 = {"C1": np.hstack([P4S["bp"], ras]), "BENDR": np.hstack([ben_r, ras]),
          "LaBraM": np.concatenate([lab_r, np.repeat(ras[:, None, :], 12, 1)], 2), "CBraMod": np.concatenate([cbr_r, np.repeat(ras[:, None, :], 12, 1)], 2)}
    test_subj = ek.eegbench_test_subjects(); subj_r, y_r = m4.subject.values, m4.label.values
    tr4 = np.where(keep & ~np.isin(subj_r, list(test_subj)))[0]; te4 = np.where(keep & np.isin(subj_r, list(test_subj)))[0]
    out, thr, best_pre, layers, w, corr = procedure(S4, y_r, tr4, te4, subj_r, 0)
    yt, st = y_r[te4], subj_r[te4]; rng = np.random.default_rng(0); us = np.unique(st); rows_of = {s: np.where(st == s)[0] for s in us}
    print(f"\n== P4 EEG-Bench split (strongest pretrained inside training: {best_pre}; layers LaBraM L{layers['M2'] + 1}, CBraMod L{layers['M3'] + 1}) ==")
    print("stacking weights (M1..M4):", np.round(w, 3), "| OOF correlation:\n", np.round(corr, 2))
    rows4 = []
    for k in keys:
        pred = out[k] > thr[k]; b = []
        for _ in range(2000):
            i = np.concatenate([rows_of[s] for s in rng.choice(us, len(us))])
            if len(np.unique(yt[i])) == 2:
                b.append(balanced_accuracy_score(yt[i], pred[i]))
        lo, hi = np.percentile(b, [2.5, 97.5])
        rows4.append(dict(config=k, AUC=roc_auc_score(yt, out[k]), thr=thr[k], BAC=balanced_accuracy_score(yt, pred), lo=lo, hi=hi,
                          SENS=recall_score(yt, pred), SPEC=recall_score(1 - yt, 1 - pred)))
        r_ = rows4[-1]
        print(f"{k:8s} AUC {r_['AUC']:.3f}  BAC {r_['BAC']:.3f} (95% CI {lo:.3f}-{hi:.3f})  thr {thr[k]:.3f}  SENS {r_['SENS']:.3f} SPEC {r_['SPEC']:.3f}")
    print("target: BENDR 0.740 (EEG-Bench README)")
    pd.DataFrame(rows4).to_csv(f"{R}/fusion_p4.csv", index=False)
