#!/usr/bin/env python3
"""Round 36 (CPU): diagnosis of the leakage mechanism of window-level adjustment (Rohira as the test instrument). See PRAREGISTRASI_putaran36.md.
Cells: RF on windows with adjustment win / pat / win_x / pat_x, and logistic regression on patient means with win / pat / none; null tests H1 and H2a.
Usage: CUDA_VISIBLE_DEVICES="" python round36_diagnosis_rohira.py --check | --null [n=10]"""
import os, sys, time, json, hashlib
os.environ["CUDA_VISIBLE_DEVICES"] = ""; os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ["PYTHONWARNINGS"] = "ignore"
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, "/home/miftah_f/RTX-5090/tuep_fresh/code")
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
import round33_pembanding_v4 as R3
import evalkit as ek
A, J, Y, N, SUBJ, D, XR, PIDX, YLAB = R3.A, R3.J, R3.Y, R3.N, R3.SUBJ, R3.D, R3.XR, R3.PIDX, R3.YLAB
OUT = os.path.join(HERE, "hasil36"); os.makedirs(OUT, exist_ok=True)


def patient_means(Xw, pidx):
    S = np.zeros((N, Xw.shape[1])); np.add.at(S, pidx, Xw.astype(np.float64)); cnt = np.bincount(pidx, minlength=N); return S / np.maximum(cnt, 1)[:, None], cnt


def fitB(Xw, pidx, XB, pool, kind):
    if kind.startswith("win"):
        rows = np.isin(pidx, pool); return np.linalg.lstsq(D[pidx][rows], Xw[rows].astype(np.float64), rcond=None)[0]
    return np.linalg.lstsq(D[pool], XB[pool], rcond=None)[0]


def offsets(Xw, pidx, XB, pool, kind, seed):
    """Offset per patient (N x d): D_p @ B. 'x' = cross-fit: training patients use B from the other training folds, patients outside the pool use B from the whole pool."""
    Ball = fitB(Xw, pidx, XB, pool, kind); Off = D @ Ball
    if kind.endswith("_x"):
        for g in np.array_split(np.random.default_rng(seed).permutation(pool), 5):
            Bg = fitB(Xw, pidx, XB, np.setdiff1d(pool, g), kind); Off[g] = D[g] @ Bg
    return Off


def rf_fold(Xw, pidx, XB, kind, r, k, tr, te):
    tr_p, _ = R3.split_val(list(SUBJ[tr]), YLAB, 100 * r + k); Z = Xw.astype(np.float64) - offsets(Xw, pidx, XB, tr, kind, 100 * r + k)[pidx] if kind != "none" else Xw
    fit = np.isin(SUBJ[pidx], tr_p); m = RandomForestClassifier(n_estimators=100, max_depth=1000, class_weight="balanced", n_jobs=1, random_state=1000 * r + k)
    m.fit(Z[fit], Y[pidx][fit]); sel = np.isin(pidx, te)
    return pd.Series(m.predict_proba(Z[sel])[:, 1]).groupby(pidx[sel]).mean().loc[te].values


def lr_fold(Xw, pidx, XB, kind, r, k, tr, te):
    Z = XB - offsets(Xw, pidx, XB, tr, kind, 100 * r + k) if kind != "none" else XB
    return ek.nested_lr().fit(Z[tr], Y[tr]).predict_proba(Z[te])[:, 1]


def null_cell(cell, scen, seed):
    perm = R3.perm_owner(scen, seed); blocks = [np.where(PIDX == q)[0] for q in range(N)]
    Xn = np.vstack([XR[blocks[perm[p]]] for p in range(N)]); pn = np.concatenate([np.full(len(blocks[perm[p]]), p) for p in range(N)]); XB, _ = patient_means(Xn, pn)
    fn, kind = (rf_fold, cell[3:]) if cell.startswith("RF-") else (lr_fold, cell[3:])
    return R3.pooled(A.c17.to_matrix(J, Parallel(n_jobs=12)(delayed(fn)(Xn, pn, XB, kind, J[f][0], J[f][1], J[f][2], J[f][3]) for f in range(len(J)))))


def checks():
    XB, cnt = patient_means(XR, PIDX); r, k, tr, te = J[0]; n = 0
    for kind in ("win", "pat", "win_x", "pat_x"):
        Off = offsets(XR, PIDX, XB, tr, kind, 1); Off2 = offsets(XR, PIDX, XB, tr, kind, 1); assert np.array_equal(Off, Off2); n += 1
        X2 = XR.copy(); m = np.isin(PIDX, te); X2[m] = X2[m] * 5 + 3; XB2, _ = patient_means(X2, PIDX); assert np.allclose(offsets(X2, PIDX, XB2, tr, kind, 1)[tr], Off[tr]); n += 1     # test data do not change the offset of training patients
    rows = np.isin(PIDX, tr); Bw = fitB(XR, PIDX, XB, tr, "win"); Rw = XR.astype(np.float64)[rows] - D[PIDX][rows] @ Bw; assert np.abs(D[PIDX][rows].T @ Rw).max() / rows.sum() < 1e-6; n += 1            # window-weighted orthogonal
    Bp = fitB(XR, PIDX, XB, tr, "pat"); Rp = XB[tr] - D[tr] @ Bp; assert np.abs(D[tr].T @ Rp).max() / len(tr) < 1e-6; n += 1                                                                    # patient-level orthogonality
    print(f"{n} checks passed; windows per patient: min {cnt.min():.0f}, median {np.median(cnt):.0f}, max {cnt.max():.0f}; max/min ratio {cnt.max() / cnt.min():.0f}", flush=True)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "--check"; t0 = time.time(); checks()
    if mode == "--null":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 10; path = f"{OUT}/diagnosis_null_replikasi.csv"; done = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame(columns=["cell", "scenario", "rep", "auc"])
        cells = ["LR-none", "LR-win", "LR-pat", "RF-pat", "RF-win_x", "RF-pat_x"]
        for cell in cells:
            for scen in ("H1", "H2a"):
                for r in range(n):
                    if ((done.cell == cell) & (done.scenario == scen) & (done.rep == r)).any():
                        continue
                    a = null_cell(cell, scen, 33000 + r); done = pd.concat([done, pd.DataFrame([dict(cell=cell, scenario=scen, rep=r, auc=a)])]); done.to_csv(path, index=False)
                g = done[(done.cell == cell) & (done.scenario == scen)].auc; se = g.std(ddof=1) / np.sqrt(len(g)); print(f"{cell} {scen}: n {len(g)} mean {g.mean():.4f} sd {g.std(ddof=1):.4f} mean+1.645SE {g.mean() + 1.645 * se:.4f} [{time.time() - t0:.0f}s]", flush=True)
        S = done.groupby(["cell", "scenario"]).auc.agg(n="count", mean="mean", sd="std").reset_index(); S["p95_rata2"] = S["mean"] + 1.645 * S.sd / np.sqrt(S.n)
        S["keputusan"] = np.where((S["mean"] <= 0.52) & (S.p95_rata2 <= 0.54), "tidak ada bukti kebocoran", np.where(S["mean"] > 0.55, "kebocoran", "pemanfaatan kecil/tidak pasti")); S.to_csv(f"{OUT}/diagnosis_null_ringkasan.csv", index=False); print(S.round(4).to_string(index=False))
    print(f"done {time.time() - t0:.0f}s")
