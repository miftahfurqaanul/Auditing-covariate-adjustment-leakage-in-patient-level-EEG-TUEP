#!/usr/bin/env python3
"""Round 40 (GPU, NM stage): paired null of NormMamba N0 vs V4 (H1 global, H2a within era) with per-fold and pooled AUC, 50 replicates. Uses the round38_nm_paired functions unchanged (same seeds).
See PRAREGISTRASI_putaran40.md + addendum1. One GPU process at a time. Usage: python round40_nm_perlipatan.py [n=50]"""
import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd, torch
from sklearn.metrics import roc_auc_score
import round38_nm_paired as N38
OUT = os.path.join(HERE, "hasil40"); os.makedirs(OUT, exist_ok=True)
Y, N, JOBS = N38.Y, N38.N, N38.JOBS

if __name__ == "__main__":
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50; path = f"{OUT}/nm_perlipatan_replikasi.csv"; t0 = time.time()
    done = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame(columns=["scenario", "rep", "variant", "auc"])
    for rep in range(n):
        for scen in ("H1", "H2a"):
            todo = [v for v in ("N0", "V4") if not ((done.scenario == scen) & (done.rep == rep) & (done.variant == v)).any()]
            if not todo:
                continue
            X2, pid2 = N38.shuffled(scen, (30500 if scen == "H1" else 30900) + rep)
            for v in todo:
                P = np.full(N, np.nan); fa = {}
                for i, (r, k, tr, te) in enumerate(JOBS[:5]):
                    P[te] = N38.null_fold(X2, pid2, r, k, tr, te, v); fa[f"fold_auc_{i}"] = float(roc_auc_score(Y[te], P[te]))
                done = pd.concat([done, pd.DataFrame([dict(scenario=scen, rep=rep, variant=v, auc=float(roc_auc_score(Y, P)), **fa)])]); done.to_csv(path, index=False)
            print(f"{scen} rep {rep} [{time.time() - t0:.0f}s]", flush=True)
    print(f"done {time.time() - t0:.0f}s", flush=True)
