#!/usr/bin/env python3
"""Putaran 56 (CPU): referensi acak berukuran setara kontrol (ulasan M3). Lihat PRAREGISTRASI_putaran56.md (SHA-256 dicatat sebelum run).
Pemakaian: CUDA_VISIBLE_DEVICES="" python round56_referensi_acak.py --check | --run [njobs]   (keluaran ke hasil56/)"""
import os, sys, json, time, hashlib
os.environ["CUDA_VISIBLE_DEVICES"] = ""; os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ["PYTHONWARNINGS"] = "ignore"
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from joblib import Parallel, delayed
import round28_analisis as A
from round29_sensitivitas import fit_norm
ek, c17, J, Y, N, BP, ERA = A.ek, A.c17, A.J, A.Y, A.N, A.BP, A.ERA
D = A.D_I2; OUT = os.path.join(HERE, "hasil56"); os.makedirs(OUT, exist_ok=True)
VAR = ("C1", "V0", "VR", "V4")


def ref_random(tr, f):
    rng = np.random.default_rng(56000 + f); n = int((Y[tr] == 0).sum()); return np.sort(rng.choice(tr, n, replace=False))


def features(X, f, tr, v):
    if v == "C1": return X
    R = {"V0": tr[Y[tr] == 0], "VR": ref_random(tr, f), "V4": tr}[v]
    B, sd = fit_norm(X, D, R); return (X - D @ B) / sd


def fold_job(X, f, tr, te):
    return {v: ek.nested_lr().fit(features(X, f, tr, v)[tr], Y[tr]).predict_proba(features(X, f, tr, v)[te])[:, 1] for v in VAR}


def replicate(r, njobs):
    Xs = A.shuffle_cols(BP, [np.arange(N)], np.random.default_rng(29100 + r))
    out = Parallel(n_jobs=njobs)(delayed(fold_job)(Xs, f, tr, te) for f, (_, _, tr, te) in enumerate(J))
    M = {v: c17.to_matrix(J, [o[v] for o in out]) for v in VAR}
    return dict(rep=r, **{v: c17.pooled_auc(M[v]) for v in VAR}, **{v + "_fold": float(np.mean(c17.fold_aucs(J, M[v]))) for v in VAR})


def diagnostics():
    rows = []
    for f, (_, _, tr, te) in enumerate(J):
        for v, R in (("V0", tr[Y[tr] == 0]), ("VR", ref_random(tr, f))):
            DR = D[R]; H = DR @ np.linalg.pinv(DR.T @ DR) @ DR.T; eras = pd.Series(ERA[R]).value_counts()
            rows.append(dict(fold=f, ref=v, n=len(R), rank=int(np.linalg.matrix_rank(DR)), min_era=int(min(eras.reindex(np.unique(ERA)).fillna(0))),
                             max_leverage=float(np.diag(H).max()), frac_epi=float(Y[R].mean())))
    return pd.DataFrame(rows)


def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()


if __name__ == "__main__":
    m = sys.argv[1] if len(sys.argv) > 1 else "--check"; nj = int(sys.argv[2]) if len(sys.argv) > 2 else 22; t0 = time.time()
    ref = pd.read_csv(os.path.join(HERE, "hasil29", "A_kebocoran_replikasi.csv")); ref = ref[ref.scenario == "H1_all"].set_index("rep")
    for f in (0, 13, 49):
        _, _, tr, te = J[f]; R = ref_random(tr, f); assert set(R) <= set(tr) and len(R) == (Y[tr] == 0).sum() and not set(R) & set(te)
    if m == "--check":
        row = replicate(0, nj)
        for v in ("C1", "V0", "V4"):
            assert abs(row[v] - ref.loc[0, v]) < 1e-9, (v, row[v], ref.loc[0, v])
        print(f"reproduksi replikasi 0 (C1, V0, V4) identik dengan putaran 29; properti referensi acak lulus; {time.time() - t0:.0f} s (AUC VR tidak dicetak)"); sys.exit()
    json.dump(dict(prereg_sha256=sha(os.path.join(HERE, "PRAREGISTRASI_putaran56.md")), code_sha256=sha(__file__), njobs=nj), open(os.path.join(OUT, "meta.json"), "w"), indent=1)
    diagnostics().to_csv(os.path.join(OUT, "diagnostik_desain.csv"), index=False)
    rows = []
    for r in range(100):
        row = replicate(r, nj); rows.append(row)
        for v in ("C1", "V0", "V4"):
            assert abs(row[v] - ref.loc[r, v]) < 1e-9, ("reproduksi gagal", r, v)
        if r % 10 == 9: print(f"  {r + 1}/100 ({(time.time() - t0) / 60:.1f} menit)", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "replikasi.csv"), index=False); print(f"selesai {(time.time() - t0) / 60:.1f} menit")
