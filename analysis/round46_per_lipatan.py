#!/usr/bin/env python3
"""Round 46 (T5): rule R1 on the per-fold AUC for C1 and V0--V4 on features without information H1 (100 replicates of round 29; see PRAREGISTRASI_putaran46.md; SHA-256 recorded before the run).
CPU only (CUDA_VISIBLE_DEVICES=""). Reuses run_matrix and the seeds of round29_sensitivitas.py (H1_all: seed 29100 + r); there is no new training beyond the round 29 pipeline.
Usage: python round46_per_lipatan.py --timing | --run [n_replicates]"""
import os, sys, json, time, hashlib
os.environ["CUDA_VISIBLE_DEVICES"] = ""; os.environ.setdefault("OMP_NUM_THREADS", "1")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import round28_analisis as A
import round29_sensitivitas as S
c17, J, BP, N = A.c17, A.J, A.BP, A.N
OUT = os.path.join(HERE, "hasil46"); os.makedirs(OUT, exist_ok=True)
VARIANTS = S.VARIANTS
POOL = pd.read_csv(os.path.join(HERE, "hasil29/A_kebocoran_replikasi.csv")); POOL = POOL[POOL.scenario == "H1_all"].set_index("rep")


def replicate(r):
    Xs = A.shuffle_cols(BP, [np.arange(N)], np.random.default_rng(29100 + r)); M = S.run_matrix(Xs, VARIANTS, False)
    fa = {v: c17.fold_aucs(J, M[v]) for v in VARIANTS}; pooled = {v: c17.pooled_auc(M[v]) for v in VARIANTS}
    for v in VARIANTS:  # reproduction of the pooled AUC of round 29 (not the per-fold value)
        assert abs(pooled[v] - POOL.loc[r, v]) < 1e-9, (r, v, pooled[v], POOL.loc[r, v])
        assert len(fa[v]) == 50
    return fa


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "--timing"; t0 = time.time()
    if mode == "--timing":
        replicate(0); print(f"1 replicate: {time.time() - t0:.1f} s; pooled AUC of round 29 reproduced (6 variants); per-fold values not printed", flush=True); sys.exit()
    nrep = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    FA = np.full((nrep, len(VARIANTS), 50), np.nan)
    for r in range(nrep):
        fa = replicate(r)
        for i, v in enumerate(VARIANTS): FA[r, i] = fa[v]
        if (r + 1) % 10 == 0: print(f"  replicate {r + 1}/{nrep} ({(time.time() - t0) / 60:.1f}minutes)", flush=True)
    np.savez(f"{OUT}/H1_auc_per_lipatan.npz", fold_auc=FA, variants=np.array(VARIANTS), seed0=29100)
    rows = [dict(rep=r, variant=v, fold_mean=float(FA[r, i].mean()), pooled=float(POOL.loc[r, v])) for r in range(nrep) for i, v in enumerate(VARIANTS)]
    pd.DataFrame(rows).to_csv(f"{OUT}/H1_per_replikasi.csv", index=False)  # raw output is saved before summarizing


def keputusan(m, se):
    return "tidak ada bukti" if (m <= 0.520 and m + 1.645 * se <= 0.540) else ("kebocoran" if m > 0.55 else "tidak pasti")
