#!/usr/bin/env python3
"""Round 30, P3 part (CPU): per-era realignment with the reference from ALL training patients (V4) vs. the reference from training controls (V0),
real data and leakage test (the sets of covariance windows are exchanged between patients). See PRAREGISTRASI_putaran30.md.
Usage: CUDA_VISIBLE_DEVICES="" python round30_p3.py --dry-run | --run [n_replicates]"""
import os, sys, json, time, hashlib
os.environ["CUDA_VISIBLE_DEVICES"] = ""; os.environ.setdefault("OMP_NUM_THREADS", "1")
HERE = os.path.dirname(os.path.abspath(__file__)); CODE = "/home/miftah_f/RTX-5090/tuep_fresh/code"; sys.path.insert(0, CODE)
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
import evalkit as ek, round4 as r4, round9 as r9, round17_common as c17, round17_p3 as p3

OUT = os.path.join(HERE, "hasil30"); os.makedirs(OUT, exist_ok=True)
Y, AS, N, J = c17.Y, c17.AS, c17.N, c17.JOBS
ERA = np.asarray(r4.YEAR_G); NJ = 8
YD = np.char.add(np.char.add(ERA, "|"), np.asarray(r4.DEV_G))


def features_v4(C, pid, tr, yg):
    """reference per era = Euclidean mean of the window covariances of ALL training patients in that era (no labels)."""
    tw = np.isin(pid, tr); era = yg[pid]; out = np.empty((len(pid), 5 * 171))
    for b in range(5):
        Cb = C[:, b]; Mg = Cb[tw].mean(0); Rc = np.empty_like(Cb)
        for g in np.unique(era):
            sel = era == g; ref = tw & sel
            M = Cb[ref].mean(0) if ref.sum() >= p3.MIN_WIN else Mg
            W, _ = p3.inv_sqrt(M); Rc[sel] = W @ Cb[sel] @ W
        out[:, b * 171:(b + 1) * 171] = p3.tangent_windows(Rc)
    cnt = np.bincount(pid, minlength=N)
    return np.add.reduceat(out, np.r_[0, np.cumsum(cnt)[:-1]], axis=0) / cnt[:, None]


def predict(C, pid, tr, te, variant, with_as):
    X = p3.p3_features(C, pid, tr, ERA, "era")[0] if variant == "V0" else features_v4(C, pid, tr, ERA)
    pca = PCA(p3.NPC, random_state=0).fit(X[tr]); F = pca.transform(X)
    if with_as:
        F = np.hstack([F, AS])
    return ek.nested_lr().fit(F[tr], Y[tr]).predict_proba(F[te])[:, 1]


def matrix(C, pid, variant, with_as):
    out = Parallel(n_jobs=NJ, max_nbytes="1M")(delayed(predict)(C, pid, tr, te, variant, with_as) for _, _, tr, te in J)
    return c17.to_matrix(J, out)


def permute_windows(C, pid, pi):
    cnt = np.bincount(pid, minlength=N); start = np.r_[0, np.cumsum(cnt)[:-1]]
    idx = np.concatenate([np.arange(start[pi[i]], start[pi[i]] + cnt[pi[i]]) for i in range(N)]); pid2 = np.concatenate([np.full(cnt[pi[i]], i) for i in range(N)])
    return C[idx], pid2


def null_replicate(C, pid, pi):
    C2, pid2 = permute_windows(C, pid, pi); row = {}
    for v in ("V0", "V4"):
        M = matrix(C2, pid2, v, False); s = M.mean(0); row[v] = c17.pooled_auc(M); row[v + "_wera"] = r9.strat_auc(Y, s, ERA); row[v + "_wyd"] = r9.strat_auc(Y, s, YD)
    return row


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "--dry-run"; t0 = time.time()
    C, pid = p3.load_cache(NJ); cnt = np.bincount(pid, minlength=N); assert (np.diff(pid) >= 0).all()
    json.dump(dict(prereg_sha256=sha(os.path.join(HERE, "PRAREGISTRASI_putaran30.md")), code_sha256=sha(__file__)), open(f"{OUT}/meta_p3.json", "w"), indent=1)
    # properti permutasi jendela
    pi = np.random.default_rng(1).permutation(N); C2, pid2 = permute_windows(C, pid, pi)
    assert len(C2) == len(C) and (np.diff(pid2) >= 0).all() and np.allclose(C2[pid2 == 5], C[pid == pi[5]]); print("window permutation properties passed", flush=True)
    if mode == "--dry-run":
        M0 = matrix(C, pid, "V0", True); print("P3 V0 (AS) AUC gabungan", round(c17.pooled_auc(M0), 5), "(must be 0.68318)", f"{time.time() - t0:.0f} s", flush=True)
        assert abs(c17.pooled_auc(M0) - 0.68318) < 1e-4
        M4 = matrix(C, pid, "V4", True); print("P3 V4 (AS) AUC gabungan", round(c17.pooled_auc(M4), 5), f"{time.time() - t0:.0f} s"); t1 = time.time()
        print("H1 one replicate:", {k: round(x, 4) for k, x in null_replicate(C, pid, np.random.default_rng(30100).permutation(N)).items()}, f"{time.time() - t1:.0f} s"); sys.exit()
    nrep = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    # data asli
    real = {}
    for v in ("V0", "V4"):
        for wa in (True, False):
            real[f"P3_{v}_{'AS' if wa else 'noAS'}"] = matrix(C, pid, v, wa)
    np.savez(f"{OUT}/p3_prediksi_data_asli.npz", **real)
    for k, M in real.items():
        fa = c17.fold_aucs(J, M); print(f"{k}: pooled AUC {c17.pooled_auc(M):.4f}; per fold {fa.mean():.4f}", flush=True)
    assert abs(c17.pooled_auc(real["P3_V0_AS"]) - 0.68318) < 1e-4
    rows = []
    for name, seed0 in (("H1_all", 30100), ("H2a_era", 30300)):
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
                print(f"  {name} replicate {r + 1}/{nrep} ({(time.time() - t0) / 60:.1f}minutes)", flush=True)
                pd.DataFrame(rows).to_csv(f"{OUT}/p3_kebocoran_replikasi.csv", index=False)
    R = pd.DataFrame(rows); R.to_csv(f"{OUT}/p3_kebocoran_replikasi.csv", index=False)
    S = R.groupby("scenario").agg(**{f"{c}_{f}": (c, fn) for c in ("V0", "V4", "V0_wera", "V4_wera") for f, fn in (("mean", "mean"), ("sd", "std"), ("p5", lambda x: np.percentile(x, 5)), ("p95", lambda x: np.percentile(x, 95)))}).reset_index()
    S.to_csv(f"{OUT}/p3_kebocoran_ringkasan.csv", index=False); print(S.round(4).T.to_string()); print(f"done {(time.time() - t0) / 60:.1f}minutes")
