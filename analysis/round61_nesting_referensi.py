#!/usr/bin/env python3
"""Putaran 61 (CPU): validasi dalam bersarang ketat untuk C1, V0, VR, V4 (ulasan editorial JBI butir 1). Lihat PRAREGISTRASI_putaran61.md (SHA-256 dicatat sebelum run).
Pemakaian: CUDA_VISIBLE_DEVICES="" python round61_nesting_referensi.py --check | --run [njobs]   (keluaran ke hasil61/; ringkasan: round61_ringkas.py)"""
import os, sys, json, time, hashlib
os.environ["CUDA_VISIBLE_DEVICES"] = ""; os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ["PYTHONWARNINGS"] = "ignore"
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import round28_analisis as A
import evalkit as ek
from round29_sensitivitas import fit_norm
from round56_referensi_acak import ref_random
c17, J, Y, N, BP = A.c17, A.J, A.Y, A.N, A.BP
D = A.D_I2; CS = ek.CS; OUT = os.path.join(HERE, "hasil61"); os.makedirs(OUT, exist_ok=True)
VAR = ("C1", "V0", "VR", "V4")


def reference(v, fit, f, k=None):
    """Pasien pemasang referensi, dipilih hanya dari `fit`. k=None: referensi luar putaran 56; k = indeks lipatan dalam: referensi dalam."""
    if v == "V0": return fit[Y[fit] == 0]
    if v == "V4": return fit
    if v == "VR":
        if k is None: return ref_random(fit, f)
        n = int((Y[fit] == 0).sum()); return np.sort(np.random.default_rng(61000 + 10 * f + k).choice(fit, n, replace=False))
    raise ValueError(v)


def adjust(X, v, R):
    if v == "C1": return X
    B, sd = fit_norm(X, D, R); return (X - D @ B) / sd


def nested(X, v, f, tr, te, strict, audit=None):
    """Pemilihan C oleh StratifiedKFold(5) tanpa acak (seperti LogisticRegressionCV). strict: referensi dan scaler dipasang ulang dari pasien latih-dalam."""
    if not strict:
        F = adjust(X, v, None if v == "C1" else reference(v, tr, f)); G = StandardScaler().fit(F[tr]).transform(F)
    score = np.zeros(len(CS))
    for k, (a, b) in enumerate(StratifiedKFold(5).split(tr, Y[tr])):
        ia, ib = tr[a], tr[b]
        if strict:
            R = None if v == "C1" else reference(v, ia, f, k)
            if audit is not None and R is not None:
                audit.append(dict(v=v, f=f, k=k, leak=bool(set(R) & set(ib)), n=len(R), n_ctrl=int((Y[ia] == 0).sum())))
            F = adjust(X, v, R); G = StandardScaler().fit(F[ia]).transform(F)
        est = LogisticRegression(class_weight="balanced", max_iter=5000, warm_start=True)
        for i, c in enumerate(CS):
            score[i] += roc_auc_score(Y[ib], est.set_params(C=c).fit(G[ia], Y[ia]).decision_function(G[ib]))
    c = CS[int(np.argmax(score))]
    F = adjust(X, v, None if v == "C1" else reference(v, tr, f)); G = StandardScaler().fit(F[tr]).transform(F)
    return LogisticRegression(C=c, class_weight="balanced", max_iter=5000).fit(G[tr], Y[tr]).predict_proba(G[te])[:, 1], float(c)


def fold_job(X, f, tr, te, strict):
    return {v: nested(X, v, f, tr, te, strict) for v in VAR}


def replicate(r, nj, strict):
    Xs = A.shuffle_cols(BP, [np.arange(N)], np.random.default_rng(29100 + r))
    out = Parallel(n_jobs=nj)(delayed(fold_job)(Xs, f, tr, te, strict) for f, (_, _, tr, te) in enumerate(J))
    M = {v: c17.to_matrix(J, [o[v][0] for o in out]) for v in VAR}
    row = dict(rep=r, **{v: c17.pooled_auc(M[v]) for v in VAR}, **{v + "_fold": float(np.mean(c17.fold_aucs(J, M[v]))) for v in VAR})
    return row, {v: np.array([o[v][1] for o in out]) for v in VAR}


def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()


if __name__ == "__main__":
    m = sys.argv[1] if len(sys.argv) > 1 else "--check"; nj = int(sys.argv[2]) if len(sys.argv) > 2 else 24; t0 = time.time()
    ref56 = pd.read_csv(os.path.join(HERE, "hasil56", "replikasi.csv")).set_index("rep")
    if m == "--check":
        row, _ = replicate(0, nj, strict=False)
        for v in VAR:
            d = abs(row[v] - ref56.loc[0, v]); print(f"longgar {v}: selisih AUC gabungan replikasi 0 terhadap putaran 56 = {d:.2e}"); assert d < 0.005, (v, d)
        audit = []; Xs = A.shuffle_cols(BP, [np.arange(N)], np.random.default_rng(29100))
        for f in (0, 13, 49):
            _, _, tr, te = J[f]
            for v in ("V0", "VR", "V4"): nested(Xs, v, f, tr, te, True, audit)
        assert audit and not any(a["leak"] for a in audit), "referensi dalam memuat pasien validasi dalam"
        assert all(a["n"] == a["n_ctrl"] for a in audit if a["v"] in ("V0", "VR")), "ukuran referensi dalam V0/VR tidak sama dengan jumlah kontrol latih-dalam"
        print(f"{len(audit)} referensi dalam diperiksa: tidak memuat pasien validasi dalam; ukuran V0/VR = kontrol latih-dalam; {time.time() - t0:.0f} s (AUC ketat tidak dicetak)"); sys.exit()
    json.dump(dict(prereg_sha256=sha(os.path.join(HERE, "PRAREGISTRASI_putaran61.md")), code_sha256=sha(__file__), njobs=nj), open(os.path.join(OUT, "meta.json"), "w"), indent=1)
    for mode, strict in (("longgar", False), ("ketat", True)):
        rows, Cs = [], {v: [] for v in VAR}
        for r in range(100):
            row, c = replicate(r, nj, strict); rows.append(row)
            for v in VAR: Cs[v].append(c[v])
            if r % 10 == 9:
                pd.DataFrame(rows).to_csv(os.path.join(OUT, f"replikasi_{mode}.csv"), index=False); print(f"  {mode} {r + 1}/100 ({(time.time() - t0) / 60:.1f} menit)", flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(OUT, f"replikasi_{mode}.csv"), index=False)
        np.savez(os.path.join(OUT, f"C_terpilih_{mode}.npz"), **{v: np.array(Cs[v]) for v in VAR})
    open(os.path.join(OUT, "SELESAI"), "w").write(time.ctime())
    print(f"selesai {(time.time() - t0) / 60:.1f} menit")
