"""Adendum 2 of reports/matched_prediction.md (commit 6570360): AUC across 200 equally valid random matchings per rule. Exploratory."""
import os, numpy as np, pandas as pd, warnings
os.environ.setdefault("OMP_NUM_THREADS", "1")
from joblib import Parallel, delayed
import matched_design as md, matched_posthoc as mp
warnings.filterwarnings("ignore")
R = "/home/miftah_f/RTX-5090/tuep_fresh/reports"; N = 200; OBS = 0.784
S = mp.S; pos = {s: i for i, s in enumerate(mp.subj)}
C = S[(S.fs == 250) & (S.montage == "01_tcp_ar")]


def random_matching(rule, i):
    g = np.random.default_rng(70_000 + i); E = C[C.y == 1].iloc[g.permutation((C.y == 1).sum())]; free = set(C[C.y == 0].index); pairs = []
    for s, e in E.iterrows():
        ok = sorted(n for n in free if (rule == "age" or C.at[n, "year"] == e.year) and (rule == "year" or abs(C.at[n, "age"] - e.age) <= md.CALIPER))
        if ok:
            n = ok[g.integers(len(ok))]; free.remove(n); pairs.append((s, n))
    rows = np.array([pos[s] for s, _ in pairs] + [pos[n] for _, n in pairs]); k = len(pairs)
    return k, mp.auc_paired(mp.BP[rows], mp.yall[rows], np.concatenate([np.arange(k), np.arange(k)]))


if __name__ == "__main__":
    for rule, name in [("year+age", "(C) same year + |age diff| <= 5  [design rule]"), ("year", "(A) same year only"), ("age", "(B) |age diff| <= 5 only")]:
        res = np.array(Parallel(n_jobs=20)(delayed(random_matching)(rule, i) for i in range(N)))
        np.save(f"{R}/matched_variability_{rule}.npy", res); k, a = res[:, 0], res[:, 1]
        print(f"{name}: pairs {k.mean():.1f} (range {int(k.min())}-{int(k.max())}); AUC mean {a.mean():.3f}, sd {a.std():.3f}, "
              f"5-95% {np.percentile(a, 5):.3f}-{np.percentile(a, 95):.3f}; share >= {OBS}: {(a >= OBS).mean():.3f}", flush=True)
