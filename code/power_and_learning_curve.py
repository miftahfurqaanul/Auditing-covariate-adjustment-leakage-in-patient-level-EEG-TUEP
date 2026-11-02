"""Two diagnostics for the strict protocol (P3):
1. minimal detectable AUC difference between two methods on the canonical 10x5 folds
   (corrected resampled t-test, alpha 0.05, power 0.8), from observed paired fold-level differences;
2. learning curve: AUC when the training folds are subsampled to 25/50/75/100 % of patients.
"""
import numpy as np, pandas as pd, warnings
from scipy import stats
import evalkit as ek
warnings.filterwarnings("ignore")
D = ek.D
T = np.load(f"{D}/p3_patient_table.npz", allow_pickle=True)
subjects, y, bp, agesex = T["subjects"], T["y"], T["bandpower"], T["agesex"]
meta = pd.read_csv(f"{D}/features_bandpower_recordings.csv"); meta = meta[meta.n_clean > 0].reset_index(drop=True)
first = (meta.groupby("subject").session_no.transform("min") == meta.session_no).values
e = np.load(f"{D}/emb_labram_windows.npz", allow_pickle=True); E, rec = e["E"].astype(np.float32), e["rec"]
lab = np.vstack([E[np.isin(rec, np.where(first & (meta.subject.values == s))[0])].mean(0) for s in subjects])
X_bp, X_lab = np.hstack([bp, agesex]), np.hstack([lab, agesex])

def fp(Xtr, ytr, Xte):
    return ek.nested_lr().fit(Xtr, ytr).predict_proba(Xte)[:, 1]

_, _, fa_bp = ek.cv_patient(X_bp, subjects, y, fp, return_folds=True)
_, _, fa_lab = ek.cv_patient(X_lab, subjects, y, fp, return_folds=True)
d = fa_lab - fa_bp; J = len(d)
se = np.sqrt(d.var(ddof=1) * (1 / J + 40 / 160))
mde = (stats.t.ppf(0.975, J - 1) + stats.norm.ppf(0.8)) * se
print(f"paired fold-level dAUC (LaBraM - bandpower): mean {d.mean():+.3f}, corrected SE {se:.3f}")
print(f"minimal detectable AUC difference (alpha .05, power .8): {mde:.3f}")

print("\nlearning curve (training patients subsampled inside each training fold; 10x5 folds):")
F = ek.folds(); pos = {s: i for i, s in enumerate(subjects)}; rng = np.random.default_rng(0)
for name, X in [("bandpower+age+sex", X_bp), ("LaBraM+age+sex", X_lab)]:
    for frac in [0.25, 0.5, 0.75, 1.0]:
        aucs = []
        for r in range(10):
            Fr = F[F.repeat == r]; p = np.full(len(y), np.nan)
            for k in range(5):
                te = np.array([pos[s] for s in Fr[Fr.fold == k].subject]); tr = np.setdiff1d(np.arange(len(y)), te)
                if frac < 1:
                    tr = np.concatenate([rng.choice(tr[y[tr] == c], int(round(frac * (y[tr] == c).sum())), replace=False) for c in (0, 1)])
                p[te] = fp(X[tr], y[tr], X[te])
            aucs.append(ek.roc_auc_score(y, p))
        print(f"  {name:18s} train patients ~{int(160 * frac):3d}: AUC {np.mean(aucs):.3f} ± {np.std(aucs):.3f}", flush=True)
