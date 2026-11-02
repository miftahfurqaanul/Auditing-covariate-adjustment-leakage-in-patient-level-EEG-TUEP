#!/usr/bin/env python3
"""Round 40 (CPU, ROH stage): Rohira null on real EEG, per-fold AUC (50 canonical folds) and pooled (mean of 10 repetitions), 30 replicates H1/H2a, seed 33000+rep.
Cells: RF-none, RF-logpat (log features; copy of round36c_paired.null_cell), LR-none, LR-pat (raw features; round36_diagnosis_rohira.null_cell). See PRAREGISTRASI_putaran40.md + addendum1.
Usage: CUDA_VISIBLE_DEVICES="" python round40_rohira_perlipatan.py [n=30]"""
import os, sys, time
os.environ["CUDA_VISIBLE_DEVICES"] = ""
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score
import round36_diagnosis_rohira as R6, round36c_paired as P36
R3, A, J, N, XR, PIDX, Y = R6.R3, R6.A, R6.J, R6.N, R6.XR, R6.PIDX, R6.Y
OUT = os.path.join(HERE, "hasil40"); os.makedirs(OUT, exist_ok=True)


def cell_matrix(cell, scen, seed):
    perm = R3.perm_owner(scen, seed); blocks = [np.where(PIDX == q)[0] for q in range(N)]
    if cell.startswith("RF-"):
        Xs = P36.XL; kind = "none" if cell == "RF-none" else "pat"
    else:
        Xs = XR; kind = cell[3:]
    Xn = np.vstack([Xs[blocks[perm[p]]] for p in range(N)]); pn = np.concatenate([np.full(len(blocks[perm[p]]), p) for p in range(N)]); XB, _ = R6.patient_means(Xn, pn)
    if cell.startswith("RF-"):
        res = Parallel(n_jobs=int(os.environ.get("NJOBS", 12)))(delayed(P36.rf_fold_as)(Xn, pn, XB, kind, False, J[f][0], J[f][1], J[f][2], J[f][3]) for f in range(len(J)))
    else:
        res = Parallel(n_jobs=int(os.environ.get("NJOBS", 12)))(delayed(R6.lr_fold)(Xn, pn, XB, kind, J[f][0], J[f][1], J[f][2], J[f][3]) for f in range(len(J)))
    return A.c17.to_matrix(J, res)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30; path = f"{OUT}/rohira_perlipatan_replikasi.csv"; t0 = time.time()
    done = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame(columns=["cell", "scenario", "rep", "auc"])
    for rep in range(n):
        for scen in ("H1", "H2a"):
            for cell in ("RF-none", "RF-logpat", "LR-none", "LR-pat"):
                if ((done.cell == cell) & (done.scenario == scen) & (done.rep == rep)).any():
                    continue
                M = cell_matrix(cell, scen, 33000 + rep); fa = [roc_auc_score(Y[te], M[r, te]) for r, k, tr, te in J]
                done = pd.concat([done, pd.DataFrame([dict(cell=cell, scenario=scen, rep=rep, auc=R3.pooled(M), fold_mean=float(np.mean(fa)), fold_sd=float(np.std(fa, ddof=1)))])]); done.to_csv(path, index=False)
        print(f"rep {rep} [{time.time() - t0:.0f}s]", flush=True)
    print(f"done {time.time() - t0:.0f}s", flush=True)
