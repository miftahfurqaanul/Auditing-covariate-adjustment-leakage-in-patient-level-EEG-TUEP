"""Evaluate the EEG-Bench replication variants A/B/C (see replicate_eegbench.py) on P4 and P3."""
import numpy as np, pandas as pd, warnings
from sklearn.model_selection import GroupKFold
import evalkit as ek
warnings.filterwarnings("ignore")
D, R = ek.D, "/home/miftah_f/RTX-5090/tuep_fresh/reports"
meta = pd.read_csv(f"{D}/features_bandpower_recordings.csv"); meta = meta[meta.n_clean > 0].reset_index(drop=True)
subj_r, y_r = meta.subject.values, meta.label.values
first = (meta.groupby("subject").session_no.transform("min") == meta.session_no).values
test_subj = ek.eegbench_test_subjects()
T = np.load(f"{D}/p3_patient_table.npz", allow_pickle=True); subjects, y = T["subjects"], T["y"]

def lr_fp(Xtr, ytr, Xte, groups=None):
    # same model (L2 logistic regression, nested C). For d >> n (BENDR flattened, 8704-d) both lbfgs and
    # dual liblinear needed > 10 min per nested fit, so the features are first reduced by PCA fitted on
    # the training fold only (unsupervised, 128 components) inside the same pipeline.
    m = ek.nested_lr()
    if Xtr.shape[1] > 2000:
        from sklearn.decomposition import PCA
        m.steps.insert(1, ("pca", PCA(n_components=min(128, len(Xtr) - 1), random_state=0)))
    if groups is not None:
        m.set_params(logisticregressioncv__cv=list(GroupKFold(5).split(Xtr, ytr, groups)))
    return m.fit(Xtr, ytr).predict_proba(Xte)[:, 1]

rows, fold_aucs = [], {}
for v, desc in [("A", "EEG-Bench recipe (notch 50 Hz)"), ("B", "same, notch 60 Hz"), ("C", "our clean data (0.5-45 Hz)")]:
    z = np.load(f"{D}/eegbench_rep_{v}.npz"); rec = z["rec"]
    for enc in ["bendr", "labram"]:
        X = z[enc].astype(np.float32); s, yy = subj_r[rec], y_r[rec]
        trm, tem = ~np.isin(s, list(test_subj)), np.isin(s, list(test_subj))
        m4, _ = ek.p4_eval(X, s, yy, trm, tem, lr_fp)
        Xp = np.vstack([X[first[rec] & (s == p)].mean(0) for p in subjects])
        m3, _, fa = ek.cv_patient(Xp, subjects, y, lr_fp, return_folds=True)
        name = f"{v} {enc}"; fold_aucs[name] = fa
        rows.append(dict(variant=v, desc=desc, encoder=enc, P4_AUC=m4["AUC"], P4_BAC=m4["BAC"], P4_BAC_lo=m4["BAC_lo"], P4_BAC_hi=m4["BAC_hi"],
                         P3_AUC=m3.AUC.mean(), P3_AUC_sd=m3.AUC.std(), P3_BAC=m3.BAC.mean()))
        print(f"{v} {enc:7s} | P4 AUC {m4['AUC']:.3f} BAC {m4['BAC']:.3f} ({m4['BAC_lo']:.3f}-{m4['BAC_hi']:.3f}) | "
              f"P3 AUC {m3.AUC.mean():.3f}±{m3.AUC.std():.3f} BAC {m3.BAC.mean():.3f}   [{desc}]", flush=True)
print("\npaired P3 comparisons (corrected resampled t-test, 50 folds):")
for enc in ["bendr", "labram"]:
    for a, b in [("A", "B"), ("A", "C"), ("B", "C")]:
        d, p = ek.corrected_ttest(fold_aucs[f"{a} {enc}"], fold_aucs[f"{b} {enc}"])
        print(f"  {enc:7s} {a} - {b}: dAUC {d:+.3f}  p={p:.3f}")
print("reference (EEG-Bench README, BAC): BENDR 0.740, LaBraM 0.565")
pd.DataFrame(rows).to_csv(f"{R}/eegbench_variants.csv", index=False)
