"""Shared evaluation code: every method is scored with these functions on the canonical folds.

Primary protocol (P3): first session of each patient, one prediction per patient,
10 x 5-fold patient-level CV from data/folds_patient.csv.
Benchmark protocol (P4): EEG-Bench subject split (last 25 + 25 subjects alphabetically = test),
all sessions, one prediction per recording file.
"""
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, recall_score

D = "/home/miftah_f/RTX-5090/tuep_fresh/data"
RAW = "/home/miftah_f/RTX-5090/EEG-TUEP-v2.0.1"
CS = np.logspace(-3, 1, 9)

def nested_lr(groups=None, n_inner=5):
    """Standardize + L2 logistic regression; C chosen by inner CV on the training data only."""
    return make_pipeline(StandardScaler(), LogisticRegressionCV(
        Cs=CS, cv=n_inner, scoring="roc_auc", class_weight="balanced", max_iter=5000))

def metrics(y, p, thr=0.5):
    yhat = (p > thr).astype(int)
    return dict(AUC=roc_auc_score(y, p), BAC=balanced_accuracy_score(y, yhat),
                SENS=recall_score(y, yhat), SPEC=recall_score(1 - y, 1 - yhat))

def folds():
    return pd.read_csv(f"{D}/folds_patient.csv")

def cv_patient(X, subjects, y, fit_predict, n_repeats=10, return_folds=False):
    """X rows aligned with `subjects`. fit_predict(Xtr, ytr, Xte) -> probabilities.
    Returns (per-repeat metrics DataFrame, mean out-of-fold probability per patient[, fold-level AUCs])."""
    F = folds(); pos = {s: i for i, s in enumerate(subjects)}
    rows, P, fold_auc = [], np.zeros(len(y)), []
    for r in range(n_repeats):
        Fr = F[F.repeat == r]; p = np.full(len(y), np.nan)
        for k in sorted(Fr.fold.unique()):
            te = np.array([pos[s] for s in Fr[Fr.fold == k].subject])
            tr = np.setdiff1d(np.arange(len(y)), te)
            p[te] = fit_predict(X[tr], y[tr], X[te])
            fold_auc.append(roc_auc_score(y[te], p[te]))
        rows.append(dict(repeat=r, **metrics(y, p))); P += p / n_repeats
    if return_folds:
        return pd.DataFrame(rows), P, np.array(fold_auc)
    return pd.DataFrame(rows), P

def corrected_ttest(a, b, n_train=160, n_test=40):
    """Nadeau-Bengio corrected resampled t-test on paired fold-level scores (repeated k-fold)."""
    from scipy import stats
    d = np.asarray(a) - np.asarray(b); J = len(d)
    var = d.var(ddof=1) * (1 / J + n_test / n_train)
    t = d.mean() / np.sqrt(var)
    return d.mean(), 2 * stats.t.sf(abs(t), J - 1)

def summarize(name, m):
    s = {c: f"{m[c].mean():.3f} ± {m[c].std():.3f}" for c in ["AUC", "BAC", "SENS", "SPEC"]}
    print(f"{name:34s} AUC {s['AUC']}  BAC {s['BAC']}  SENS {s['SENS']}  SPEC {s['SPEC']}", flush=True)
    return dict(name=name, **{c: m[c].mean() for c in ["AUC", "BAC", "SENS", "SPEC"]},
                **{c + "_sd": m[c].std() for c in ["AUC", "BAC"]})

def eegbench_test_subjects():
    ep = sorted(l.split()[0] for l in open(f"{RAW}/00_subject_ids_epilepsy.list"))
    ne = sorted(l.split()[0] for l in open(f"{RAW}/01_subject_ids_no_epilepsy.list"))
    return set(ep[75:]) | set(ne[75:])

def p4_eval(X, subjects, y, train_mask, test_mask, fit_predict, n_boot=2000, seed=0):
    """Train on EEG-Bench train subjects, score per file on test subjects; bootstrap CI over patients."""
    p = fit_predict(X[train_mask], y[train_mask], X[test_mask], groups=subjects[train_mask])
    yt, st = y[test_mask], subjects[test_mask]
    m = metrics(yt, p); rng = np.random.default_rng(seed); us = np.unique(st); boots = []
    rows_of = {s: np.where(st == s)[0] for s in us}
    for _ in range(n_boot):
        idx = np.concatenate([rows_of[s] for s in rng.choice(us, len(us))])
        if len(np.unique(yt[idx])) == 2:
            boots.append(balanced_accuracy_score(yt[idx], p[idx] > 0.5))
    m["BAC_lo"], m["BAC_hi"] = np.percentile(boots, [2.5, 97.5])
    return m, p
