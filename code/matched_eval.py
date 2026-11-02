"""Matched-subset analysis pre-registered in reports/matched_prediction.md (commit cbd48bd). CPU only."""
import os, sys, numpy as np, pandas as pd, warnings
os.environ.setdefault("OMP_NUM_THREADS", "1")
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
import evalkit as ek, matched_design as md
warnings.filterwarnings("ignore")
D, R = md.D, "/home/miftah_f/RTX-5090/tuep_fresh/reports"
N_PERM = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 1000

T = np.load(f"{D}/p3_patient_table.npz", allow_pickle=True); pos = {s: i for i, s in enumerate(T["subjects"])}
S = md.subject_table(); P = pd.read_csv(f"{D}/matched_pairs.csv")
members = np.concatenate([P.epilepsy.values, P.non_epilepsy.values]); pair = np.concatenate([P.pair.values, P.pair.values])
rows = np.array([pos[s] for s in members]); y = T["y"][rows]
assert (y[: len(P)] == 1).all() and (y[len(P):] == 0).all()
X = {"M-EEG": T["bandpower"][rows], "M-EEG+AS": np.hstack([T["bandpower"][rows], T["agesex"][rows]]),
     "M-meta": S.loc[members, ["age", "male", "year", "dur_min", "n_files"]].values.astype(float)}
assert np.allclose(S.loc[members, "age"].values, T["agesex"][rows, 0])          # same age source in both tables


def splits(r):
    g = np.array_split(np.random.default_rng(r).permutation(len(P)), 5)
    return [(np.where(~np.isin(pair, k))[0], np.where(np.isin(pair, k))[0]) for k in g]


for r in range(10):                                                             # double check: whole pairs, balanced classes
    for tr, te in splits(r):
        assert len(set(pair[tr]) & set(pair[te])) == 0 and y[te].sum() * 2 == len(te)


def procedure(Xm, yy):
    auc, bac = [], []
    for r in range(10):
        p = np.zeros(len(yy))
        for tr, te in splits(r):
            p[te] = ek.nested_lr().fit(Xm[tr], yy[tr]).predict_proba(Xm[te])[:, 1]
        auc.append(roc_auc_score(yy, p)); bac.append(balanced_accuracy_score(yy, p > 0.5))
    return np.mean(auc), np.std(auc), np.mean(bac), np.std(bac)


if __name__ == "__main__":
    print(f"matched set: {len(P)} pairs, {len(y)} patients; folds keep pairs together (checked)")
    obs = {k: procedure(v, y) for k, v in X.items()}
    for k, (a, asd, b, bsd) in obs.items():
        print(f"{k:9s} AUC {a:.3f} ± {asd:.3f} (sd over 10 repeats)   BAC {b:.3f} ± {bsd:.3f}")

    def null(i):
        flip = np.random.default_rng(10_000 + i).random(len(P)) < 0.5
        yp = y.copy(); f = np.isin(pair, np.where(flip)[0]); yp[f] = 1 - yp[f]   # swap labels inside flipped pairs
        return procedure(X["M-EEG"], yp)[0]
    nul = np.array(Parallel(n_jobs=20)(delayed(null)(i) for i in range(N_PERM)))
    p = (1 + (nul >= obs["M-EEG"][0]).sum()) / (1 + N_PERM)
    print(f"permutation M-EEG ({N_PERM}x, labels swapped within pairs, full 10x5 procedure refit): null AUC mean {nul.mean():.3f}, "
          f"95th pct {np.percentile(nul, 95):.3f}, p = {p:.4f}")
    np.save(f"{R}/matched_null.npy", nul)
    pd.DataFrame({k: dict(AUC=v[0], AUC_sd=v[1], BAC=v[2], BAC_sd=v[3]) for k, v in obs.items()}).T.assign(p_perm_MEEG=p).to_csv(f"{R}/matched_results.csv")
