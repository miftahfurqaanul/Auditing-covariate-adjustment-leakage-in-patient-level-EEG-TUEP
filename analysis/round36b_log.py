#!/usr/bin/env python3
"""Round 36b (CPU): log cell (Addendum 1 of PRAREGISTRASI_putaran36.md): RF and LR with adjustment on log band-power features; null tests H1/H2a; kurtosis and leverage diagnostics.
Usage: CUDA_VISIBLE_DEVICES="" python round36b_log.py --diag | --null [n=10]"""
import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
from scipy import stats
from joblib import Parallel, delayed
import round36_diagnosis_rohira as R6
R3, A, J, N, XR, PIDX = R6.R3, R6.A, R6.J, R6.N, R6.XR, R6.PIDX
OUT = R6.OUT; XL = np.log(np.maximum(XR.astype(np.float64), 1e-12))


def null_cell_log(cell, scen, seed):
    perm = R3.perm_owner(scen, seed); blocks = [np.where(PIDX == q)[0] for q in range(N)]
    Xn = np.vstack([XL[blocks[perm[p]]] for p in range(N)]); pn = np.concatenate([np.full(len(blocks[perm[p]]), p) for p in range(N)]); XB, _ = R6.patient_means(Xn, pn)
    fn, kind = (R6.rf_fold, cell[3:]) if cell.startswith("RF-") else (R6.lr_fold, cell[3:])
    return R3.pooled(A.c17.to_matrix(J, Parallel(n_jobs=12)(delayed(fn)(Xn, pn, XB, kind, J[f][0], J[f][1], J[f][2], J[f][3]) for f in range(len(J)))))


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "--diag"; t0 = time.time()
    if mode == "--diag":
        XB_r, cnt = R6.patient_means(XR, PIDX); XB_l, _ = R6.patient_means(XL, PIDX)
        rows = [dict(skala=nm, niveau=nv, kurtosis_median=float(np.median(stats.kurtosis(M, axis=0))), kurtosis_maks=float(np.max(stats.kurtosis(M, axis=0))), skew_median=float(np.median(stats.skew(M, axis=0))), skew_maks=float(np.max(np.abs(stats.skew(M, axis=0))))) for nm, nv, M in (("mentah", "jendela", XR.astype(np.float64)), ("log", "jendela", XL), ("mentah", "rata-rata pasien", XB_r), ("log", "rata-rata pasien", XB_l))]
        print(pd.DataFrame(rows).round(2).to_string(index=False)); pd.DataFrame(rows).to_csv(f"{OUT}/diagnosis_kurtosis.csv", index=False)
        r, k, tr, te = J[0]; lev = {}
        for nm, M in (("mentah", XB_r), ("log", XB_l)):
            Dm = R6.D[tr]; H = Dm @ np.linalg.pinv(Dm.T @ Dm) @ Dm.T; lev[nm] = float(np.max(np.diag(H)))
        print("max patient leverage (design; same for both scales):", lev)
    else:
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 10; path = f"{OUT}/diagnosis_log_null_replikasi.csv"; done = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame(columns=["cell", "scenario", "rep", "auc"])
        for cell in ("RF-logpat", "RF-logwin", "LR-logpat"):
            kind = cell.replace("log", "")
            for scen in ("H1", "H2a"):
                for r in range(n):
                    if ((done.cell == cell) & (done.scenario == scen) & (done.rep == r)).any():
                        continue
                    a = null_cell_log(kind, scen, 33000 + r); done = pd.concat([done, pd.DataFrame([dict(cell=cell, scenario=scen, rep=r, auc=a)])]); done.to_csv(path, index=False)
                g = done[(done.cell == cell) & (done.scenario == scen)].auc; se = g.std(ddof=1) / np.sqrt(len(g)); print(f"{cell} {scen}: n {len(g)} mean {g.mean():.4f} sd {g.std(ddof=1):.4f} mean+1.645SE {g.mean() + 1.645 * se:.4f} [{time.time() - t0:.0f}s]", flush=True)
    print(f"done {time.time() - t0:.0f}s")
