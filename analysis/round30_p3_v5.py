#!/usr/bin/env python3
"""Round 30, P3-V5 (addendum 2 of the pre-registration): one global reference (all training patients) + covariate regression on the PCA components (all training patients).
Real data and leakage test (permutation of the window sets). CPU only.
Usage: CUDA_VISIBLE_DEVICES="" python round30_p3_v5.py --dry-run | --run [n_replicates]"""
import os, sys, json, time, hashlib
os.environ["CUDA_VISIBLE_DEVICES"] = ""; os.environ.setdefault("OMP_NUM_THREADS", "1")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, "/home/miftah_f/RTX-5090/tuep_fresh/code")
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
import round30_p3 as M3
p3, c17, r9, ek, Y, AS, N, J, ERA, YD, permute_windows = M3.p3, M3.c17, M3.r9, M3.ek, M3.Y, M3.AS, M3.N, M3.J, M3.ERA, M3.YD, M3.permute_windows
import round4 as r4
OUT = M3.OUT; D = r4.DESIGN["I2"]; NJ = 8


def features_v5(C, pid, tr):
    tw = np.isin(pid, tr); out = np.empty((len(pid), 5 * 171))
    for b in range(5):
        Cb = C[:, b]; W, _ = p3.inv_sqrt(Cb[tw].mean(0)); out[:, b * 171:(b + 1) * 171] = p3.tangent_windows(W @ Cb @ W)
    cnt = np.bincount(pid, minlength=N)
    return np.add.reduceat(out, np.r_[0, np.cumsum(cnt)[:-1]], axis=0) / cnt[:, None]


def predict_v5(C, pid, tr, te, with_as):
    X = features_v5(C, pid, tr); pca = PCA(p3.NPC, random_state=0).fit(X[tr]); Zp = pca.transform(X)
    B = np.linalg.lstsq(D[tr], Zp[tr], rcond=None)[0]; F = Zp - D @ B
    if with_as:
        F = np.hstack([F, AS])
    return ek.nested_lr().fit(F[tr], Y[tr]).predict_proba(F[te])[:, 1]


def matrix(C, pid, with_as):
    out = Parallel(n_jobs=NJ, max_nbytes="1M")(delayed(predict_v5)(C, pid, tr, te, with_as) for _, _, tr, te in J); return c17.to_matrix(J, out)


def null_replicate(C, pid, pi):
    C2, pid2 = permute_windows(C, pid, pi); Mx = matrix(C2, pid2, False); s = Mx.mean(0)
    return dict(V5=c17.pooled_auc(Mx), V5_wera=r9.strat_auc(Y, s, ERA), V5_wyd=r9.strat_auc(Y, s, YD))


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "--dry-run"; t0 = time.time(); C, pid = p3.load_cache(NJ)
    json.dump(dict(prereg_sha256=sha(os.path.join(HERE, "PRAREGISTRASI_putaran30.md")), code_sha256=sha(__file__)), open(f"{OUT}/meta_p3_v5.json", "w"), indent=1)
    if mode == "--dry-run":
        Ma = matrix(C, pid, True); print("P3-V5 (AS) AUC gabungan", round(c17.pooled_auc(Ma), 5), f"{time.time() - t0:.0f} s", flush=True)
        t1 = time.time(); print("H1 one replicate:", {k: round(x, 4) for k, x in null_replicate(C, pid, np.random.default_rng(30700).permutation(N)).items()}, f"{time.time() - t1:.0f} s"); sys.exit()
    nrep = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    real = {"P3_V5_AS": matrix(C, pid, True), "P3_V5_noAS": matrix(C, pid, False)}; np.savez(f"{OUT}/p3_v5_prediksi_data_asli.npz", **real)
    for k, Mx in real.items():
        print(f"{k}: pooled AUC {c17.pooled_auc(Mx):.4f}; per fold {c17.fold_aucs(J, Mx).mean():.4f}", flush=True)
    rows = []
    for name, seed0 in (("H1_all", 30700), ("H2a_era", 30900)):
        for r in range(nrep):
            rng = np.random.default_rng(seed0 + r)
            if name == "H1_all":
                pi = rng.permutation(N)
            else:
                pi = np.arange(N)
                for e in np.unique(ERA):
                    g = np.where(ERA == e)[0]; pi[g] = g[rng.permutation(len(g))]
            rows.append(dict(scenario=name, rep=r, **null_replicate(C, pid, pi)))
            if r % 10 == 9:
                print(f"  {name} replicate {r + 1}/{nrep} ({(time.time() - t0) / 60:.1f}minutes)", flush=True); pd.DataFrame(rows).to_csv(f"{OUT}/p3_v5_kebocoran_replikasi.csv", index=False)
    R = pd.DataFrame(rows); R.to_csv(f"{OUT}/p3_v5_kebocoran_replikasi.csv", index=False)
    S = R.groupby("scenario").agg(**{f"{c}_{f}": (c, fn) for c in ("V5", "V5_wera") for f, fn in (("mean", "mean"), ("sd", "std"), ("p5", lambda x: np.percentile(x, 5)), ("p95", lambda x: np.percentile(x, 95)))}).reset_index()
    S.to_csv(f"{OUT}/p3_v5_kebocoran_ringkasan.csv", index=False); print(S.round(4).T.to_string()); print(f"done {(time.time() - t0) / 60:.1f}minutes")
