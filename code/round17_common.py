"""Round 17 shared code (reports/round17_P1-P5_prediction.md, commit e8abbe9; choices in reports/round17_implementation_notes.md, commit 624858e).
Canonical fold jobs (copied verbatim from the pre-registration, Bagian 1), year-shuffle designs, strata, prediction matrices. CPU only."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys, time, argparse, platform, numpy as np, pandas as pd, sklearn, scipy
from sklearn.metrics import roc_auc_score
import evalkit as ek, round4 as r4, round10 as r10

OUT = f"{ek.D}/round17"; R = "/home/miftah_f/RTX-5090/tuep_fresh/reports"; os.makedirs(OUT, exist_ok=True)
SUBJ, Y, BP, AS, N = r4.SUBJ, r4.Y, r4.BP, r4.AGESEX, len(r4.Y)

# ---- fold construction, copied verbatim from round17_P1-P5_prediction.md Bagian 1
import evalkit as ek, round4 as r4
F = ek.folds(); POS = {s:i for i,s in enumerate(r4.SUBJ)}
JOBS = [(r,k,tr,te) for r in range(10) for k in range(5)
        for te in [__import__('numpy').array([POS[s] for s in F[(F.repeat==r)&(F.fold==k)].subject])]
        for tr in [__import__('numpy').setdiff1d(__import__('numpy').arange(len(r4.Y)), te)]]
# ----
assert all(np.array_equal(a[2], b[2]) and np.array_equal(a[3], b[3]) and a[:2] == b[:2] for a, b in zip(JOBS, r10.J200)) and len(JOBS) == 50
KEEP169 = r4.YEAR_G != "<=2009"; J169 = r10.jobs(KEEP169)
MONT = np.where(r4.S.montage.values == "02_tcp_le", "LE", "AR")                  # AR_A (1 patient) merged into AR, as in round 13
STRATA = {"year": r4.YEAR_G, "device": r4.DEV_G, "year x montage": np.char.add(np.char.add(r4.YEAR_G, "|"), MONT)}
N_SHUF = 20


def year_design(yg):
    """I2 design [1, age, sex, year-group dummies] for any assignment of year groups (r4.dummies: reference = most frequent group)."""
    return np.column_stack([np.ones(N), r4.AGE, r4.MALE, r4.dummies(yg)])


assert np.array_equal(year_design(r4.YEAR_G), r4.DESIGN["I2"])


def shuffled_year(i):
    """Year groups permuted over the 200 patients (group sizes fixed), seed 17000 + i."""
    return np.random.default_rng(17000 + i).permutation(r4.YEAR_G)


def to_matrix(J, preds):
    P = np.full((10, N), np.nan)
    for (r, _, _, te), p in zip(J, preds):
        P[r, te] = p
    return P


def pooled_auc(P):
    """Mean over repeats of the AUC over all evaluated patients (the round 4-16 point estimate)."""
    return float(np.mean([roc_auc_score(Y[~np.isnan(p)], p[~np.isnan(p)]) for p in P]))


def fold_aucs(J, P):
    return np.array([roc_auc_score(Y[te], P[r, te]) for r, _, _, te in J])


def args():
    a = argparse.ArgumentParser(); a.add_argument("--dry-run", action="store_true"); a.add_argument("--run", action="store_true")
    a.add_argument("--jobs", type=int, default=12); return a.parse_args()


def header(name):
    import joblib
    print(f"{name} | {time.strftime('%Y-%m-%d %H:%M:%S')} | device cpu (CUDA_VISIBLE_DEVICES='{os.environ.get('CUDA_VISIBLE_DEVICES')}') | python {platform.python_version()} "
          f"numpy {np.__version__} scipy {scipy.__version__} sklearn {sklearn.__version__} pandas {pd.__version__} joblib {joblib.__version__}", flush=True)


def pf(ok, msg):
    print(("PASS " if ok else "FAIL ") + msg, flush=True); return ok
