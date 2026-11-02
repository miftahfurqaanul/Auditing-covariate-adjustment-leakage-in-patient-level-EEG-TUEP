"""Post hoc checks of the matched-subset result (reports/matched_prediction.md, adendum 1, commit 0a682c3). Exploratory. CPU only."""
import os, numpy as np, pandas as pd, warnings
os.environ.setdefault("OMP_NUM_THREADS", "1")
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score
import evalkit as ek, matched_design as md
warnings.filterwarnings("ignore")
D, R = md.D, "/home/miftah_f/RTX-5090/tuep_fresh/reports"; N_DRAW = 500
T = np.load(f"{D}/p3_patient_table.npz", allow_pickle=True); subj, yall, BP = T["subjects"], T["y"], T["bandpower"]
S = md.subject_table(); P = pd.read_csv(f"{D}/matched_pairs.csv")


def auc_paired(X, y, pair):
    """Same procedure as matched_eval: 10 repeats x 5 folds, pairs kept together, nested L2 logistic regression."""
    out = []; npair = pair.max() + 1
    for r in range(10):
        g = np.array_split(np.random.default_rng(r).permutation(npair), 5); p = np.zeros(len(y))
        for k in g:
            te = np.isin(pair, k); p[te] = ek.nested_lr().fit(X[~te], y[~te]).predict_proba(X[te])[:, 1]
        out.append(roc_auc_score(y, p))
    return np.mean(out)


def draw(pool_idx, i):
    """43 epilepsy + 43 non-epilepsy drawn at random from pool_idx, paired at random (no age/year matching)."""
    g = np.random.default_rng(50_000 + i); e = g.choice(pool_idx[yall[pool_idx] == 1], 43, replace=False); n = g.choice(pool_idx[yall[pool_idx] == 0], 43, replace=False)
    rows = np.concatenate([e, n]); pair = np.concatenate([np.arange(43), np.arange(43)])
    return auc_paired(BP[rows], yall[rows], pair)


if __name__ == "__main__":
    OBS = 0.784
    # 1. small-sample chance: random 43+43 draws from all 200 and from the 139 homogeneous candidates
    homog = np.where(((S.fs == 250) & (S.montage == "01_tcp_ar")).values)[0]
    for name, pool in [("all 200 patients", np.arange(len(yall))), ("139 homogeneous (250 Hz, 01_tcp_ar)", homog)]:
        a = np.array(Parallel(n_jobs=20)(delayed(draw)(pool, i) for i in range(N_DRAW)))
        np.save(f"{R}/matched_posthoc_draws_{len(pool)}.npy", a)
        print(f"1. random 43+43 from {name}: AUC mean {a.mean():.3f}, sd {a.std():.3f}, 5-95% {np.percentile(a, 5):.3f}-{np.percentile(a, 95):.3f}; "
              f"share of draws >= {OBS}: {(a >= OBS).mean():.3f}", flush=True)
    # 2. which features carry it, on the matched set
    pos = {s: i for i, s in enumerate(subj)}
    members = np.concatenate([P.epilepsy.values, P.non_epilepsy.values]); rows = np.array([pos[s] for s in members])
    pair = np.concatenate([P.pair.values, P.pair.values]); y = yall[rows]; Xm = BP[rows]
    for name, cols in [("all 114 band-power features", slice(0, 114)), ("relative band power only (95)", slice(0, 95)), ("log total power only (19)", slice(95, 114))]:
        print(f"2. matched set, {name}: AUC {auc_paired(Xm[:, cols], y, pair):.3f}", flush=True)
    # 3. direction of effect and residual covariates
    lt = Xm[:, 95:].mean(1)
    print(f"3. mean log total power (19-ch average): epilepsy {lt[y == 1].mean():.3f}, non {lt[y == 0].mean():.3f}; "
          f"within-pair difference epilepsy-minus-non: mean {np.mean(lt[:43] - lt[43:]):.3f}, positive in {(lt[:43] > lt[43:]).mean():.0%} of pairs")
    bands = ["delta", "theta", "alpha", "beta", "gamma"]
    for b, nm in enumerate(bands):
        v = Xm[:, 19 * b:19 * (b + 1)].mean(1); d = v[:43] - v[43:]
        print(f"   relative {nm:5s}: within-pair difference epilepsy-minus-non mean {d.mean():+.3f}, positive in {(d > 0).mean():.0%} of pairs")
    p = np.zeros(len(y))
    for r in range(10):
        for k in np.array_split(np.random.default_rng(r).permutation(43), 5):
            te = np.isin(pair, k); p[te] += ek.nested_lr().fit(Xm[~te], y[~te]).predict_proba(Xm[te])[:, 1] / 10
    cov = S.loc[members]
    for c in ["n_files", "dur_min", "male"]:
        print(f"   corr(out-of-fold score, {c}) within class: epilepsy {np.corrcoef(p[y == 1], cov[c].values[y == 1])[0, 1]:+.2f}, "
              f"non {np.corrcoef(p[y == 0], cov[c].values[y == 0])[0, 1]:+.2f}")
