"""Evaluate the Rohira et al. 2023 reproduction (reports/rohira_repro_prediction.md, commit a6e6bbf). CPU only.
W = windows shuffled 67/33 (paper protocol, 5 seeds); P = patients split 67/33 per class (10 seeds)."""
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import rohira_features as rf
OUT = rf.OUT; R = "/home/miftah_f/RTX-5090/tuep_fresh/reports"
SETS = {"A": ["FP1", "FP2", "C3", "C4", "T3", "T4", "O1", "O2"], "B": ["F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2"]}


def m(y, p):
    q = p > 0.5; sens = (q & (y == 1)).sum() / (y == 1).sum(); spec = (~q & (y == 0)).sum() / (y == 0).sum()
    return dict(acc=100 * (q == y).mean(), sens=100 * sens, spec=100 * spec, bac=100 * (sens + spec) / 2, auc=100 * roc_auc_score(y, p))


def fit_predict(X, y, tr, te, seed):
    return RandomForestClassifier(n_estimators=100, max_depth=1000, n_jobs=12, random_state=seed).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]


if __name__ == "__main__":
    z = np.load(f"{OUT}/features.npz"); idx = pd.read_csv(f"{OUT}/index.csv")
    subj = idx.subj.values[z["file_idx"]]; y = idx.label.values[z["file_idx"]]
    print(f"{len(y)} windows ({y.sum()} epilepsy / {(1 - y).sum()} non), {len(np.unique(subj))} patients; majority-class accuracy {100 * max(y.mean(), 1 - y.mean()):.2f}%")
    rows = []
    for name, chans in SETS.items():
        X = z["X"][:, [rf.CH.index(c) for c in chans], :].reshape(len(y), -1)
        for seed in range(5):
            tr, te = train_test_split(np.arange(len(y)), test_size=0.33, shuffle=True, random_state=seed)
            p = fit_predict(X, y, tr, te, seed)
            rows.append(dict(set=name, proto="W", seed=seed, shared_patients=len(set(subj[tr]) & set(subj[te])),
                             majority=100 * max(y[te].mean(), 1 - y[te].mean()), **m(y[te], p)))
            print(rows[-1], flush=True)
        for seed in range(10):
            g = np.random.default_rng(seed); test_p = []
            for c in [0, 1]:
                ps = np.unique(subj[y == c]); test_p += list(g.permutation(ps)[: int(round(0.33 * len(ps)))])
            te = np.where(np.isin(subj, test_p))[0]; tr = np.where(~np.isin(subj, test_p))[0]
            p = fit_predict(X, y, tr, te, seed)
            pp = pd.DataFrame(dict(s=subj[te], y=y[te], p=p)).groupby("s").agg(y=("y", "first"), p=("p", "mean"))
            pl = m(pp.y.values, pp.p.values)
            rows.append(dict(set=name, proto="P", seed=seed, shared_patients=len(set(subj[tr]) & set(subj[te])),
                             majority=100 * max(y[te].mean(), 1 - y[te].mean()), **m(y[te], p), **{f"pat_{k}": v for k, v in pl.items()}))
            print(rows[-1], flush=True)
    D = pd.DataFrame(rows); D.to_csv(f"{R}/rohira_results.csv", index=False)
    cols = ["acc", "sens", "spec", "bac", "auc", "majority", "shared_patients", "pat_bac", "pat_auc"]
    print("\n== mean (sd) over seeds ==")
    for (s, pr), g in D.groupby(["set", "proto"]):
        print(f"set {s} {pr}: " + "  ".join(f"{c} {g[c].mean():.2f} ({g[c].std():.2f})" for c in cols if c in g and g[c].notna().any()))
    print("paper (Rohira 2023, PSD + RF, shuffled 67/33): accuracy 95.73")
