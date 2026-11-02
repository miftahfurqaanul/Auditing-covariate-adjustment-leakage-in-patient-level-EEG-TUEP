"""Round 4 addendum 1 (reports/round4_prediction.md, commit a06d778): negative control for I2 with permuted year groups,
and I2 + Platt recalibration (I2R) with the normative model refitted inside every inner split. Exploratory. CPU only."""
import os, numpy as np, pandas as pd, warnings
os.environ.setdefault("OMP_NUM_THREADS", "1")
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import evalkit as ek, round4 as r4
warnings.filterwarnings("ignore"); R = "/home/miftah_f/RTX-5090/tuep_fresh/reports"; N_PERM = 100
Y, BP, AGESEX, AGE, MALE = r4.Y, r4.BP, r4.AGESEX, r4.AGE, r4.MALE
F = ek.folds(); pos = {s: i for i, s in enumerate(r4.SUBJ)}
JOBS = [(r, np.setdiff1d(np.arange(len(Y)), te), te) for r in range(10) for k in range(5)
        for te in [np.array([pos[s] for s in F[(F.repeat == r) & (F.fold == k)].subject])]]


def z_from(design, fit_idx):
    c = fit_idx[Y[fit_idx] == 0]; B = np.linalg.lstsq(design[c], BP[c], rcond=None)[0]
    sd = (BP[c] - design[c] @ B).std(0, ddof=np.linalg.matrix_rank(design[c])); return (BP - design @ B) / sd


def i2_auc(groups):
    design = np.column_stack([np.ones(len(Y)), AGE, MALE, r4.dummies(groups)]); aucs = []
    for r in range(10):
        p = np.zeros(len(Y))
        for rr, tr, te in JOBS:
            if rr == r:
                X = np.hstack([z_from(design, tr), AGESEX]); p[te] = ek.nested_lr().fit(X[tr], Y[tr]).predict_proba(X[te])[:, 1]
        aucs.append(roc_auc_score(Y, p))
    return np.mean(aucs)


def i2r_fold(tr, te, seed):
    design = r4.DESIGN["I2"]; X = np.hstack([z_from(design, tr), AGESEX]); p = ek.nested_lr().fit(X[tr], Y[tr]).predict_proba(X[te])[:, 1]
    oof = np.zeros(len(tr))
    for a, b in StratifiedKFold(5, shuffle=True, random_state=seed).split(tr, Y[tr]):
        Xi = np.hstack([z_from(design, tr[a]), AGESEX]); oof[b] = ek.nested_lr().fit(Xi[tr[a]], Y[tr[a]]).predict_proba(Xi[tr[b]])[:, 1]
    lg = lambda q: np.log(np.clip(q, 1e-6, 1 - 1e-6) / (1 - np.clip(q, 1e-6, 1 - 1e-6)))
    return LogisticRegression(penalty=None).fit(lg(oof)[:, None], Y[tr]).predict_proba(lg(p)[:, None])[:, 1]


if __name__ == "__main__":
    real = i2_auc(r4.YEAR_G); print(("PASS" if abs(real - 0.71206) < 5e-4 else "CHECK") + f" I2 re-run with real year groups: AUC {real:.4f} (round 4: 0.7121)", flush=True)
    rng = np.random.default_rng(4242); perms = [rng.permutation(r4.YEAR_G) for _ in range(N_PERM)]
    null = np.array(Parallel(n_jobs=12)(delayed(i2_auc)(g) for g in perms)); np.save(f"{R}/round4_null_permuted_year.npy", null)
    print(f"negative control (year groups permuted {N_PERM}x): AUC mean {null.mean():.4f}, sd {null.std():.4f}, 95th pct {np.percentile(null, 95):.4f}, max {null.max():.4f}")
    print(f"real I2 {real:.4f} exceeds {100 * (null < real).mean():.0f}% of permuted runs; p = {(1 + (null >= real).sum()) / (1 + N_PERM):.3f}")
    out = Parallel(n_jobs=12)(delayed(i2r_fold)(tr, te, 7 + i) for i, (r, tr, te) in enumerate(JOBS))
    per = []
    for r in range(10):
        p = np.zeros(len(Y))
        for (rr, _, te), o in zip(JOBS, out):
            if rr == r:
                p[te] = o
        ci, cs = r4.calib(Y, p); per.append(dict(auc=roc_auc_score(Y, p), **r4.binm(Y, p > 0.5), brier=np.mean((p - Y) ** 2), calib_int=ci, calib_slope=cs))
    D = pd.DataFrame(per).mean(); print("\nI2R (I2 + recalibration): " + "  ".join(f"{k} {100 * v:.2f}" if k not in ("brier", "calib_int", "calib_slope") else f"{k} {v:.3f}" for k, v in D.items()))
