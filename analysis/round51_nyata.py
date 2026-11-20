#!/usr/bin/env python3
"""Putaran 51 (E3a, CPU): label nyata, 50 lipatan kanonis, nesting ketat seperti putaran 49A; penyesuaian OLS (v4) diganti penyesuaian EB untuk I2, P3, dan N1.
Kode putaran 49 diimpor apa adanya (nested, J, BP, AS, ACQ, P3). Lihat PRAREGISTRASI_putaran51.md.
Pemakaian: CUDA_VISIBLE_DEVICES="" python round51_nyata.py --check | --run   (hasil51/nyata_prediksi.npz, hasil51/nyata_lambda.csv)"""
import os, sys, time, json, hashlib
os.environ["CUDA_VISIBLE_DEVICES"] = ""; os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ["PYTHONWARNINGS"] = "ignore"
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
import round49_nesting_ketat as R49
import round51_common as K
J, Y, N, BP, D, AS, ACQ, P5, p3 = R49.J, R49.Y, R49.N, R49.BP, R49.D, R49.AS, R49.ACQ, R49.P5, R49.p3
OUT = os.path.join(HERE, "hasil51"); os.makedirs(OUT, exist_ok=True)


def v4eb(X, fit, how="ash"):
    """Seperti R49.v4 (residu dibagi SD residu pemasangan), dengan koefisien ash_coef (utama) atau eb_coef."""
    B = (K.ash_coef if how == "ash" else K.eb_coef)(D[fit], X[fit])[0]; sd = (X[fit] - D[fit] @ B).std(0, ddof=np.linalg.matrix_rank(D[fit])); return (X - D @ B) / sd


def make_tf(model, C=None, pid=None):
    if model == "I2eb":
        return lambda fit: np.hstack([v4eb(BP, fit), AS])
    if model == "N1eb":
        return lambda fit: np.hstack([ACQ, AS, v4eb(BP, fit)])
    if model == "P3eb":
        def f(fit):
            X = P5.features_v5(C, pid, fit); Zp = PCA(p3.NPC, random_state=0).fit(X[fit]).transform(X); B = K.ash_coef(D[fit], Zp[fit])[0]; return np.hstack([Zp - D @ B, AS])
        return f
    raise ValueError(model)


def job(model, f, C=None, pid=None):
    r, k, tr, te = J[f]; assert not set(tr) & set(te); p, c = R49.nested(make_tf(model, C, pid), tr, te, True); return f, p, c


def lambdas():
    """Faktor penyusutan ash per lipatan luar (deskriptif; median dan maksimum lintas fitur per kolom kovariat): BP (I2/N1) dan komponen PCA P3."""
    rows = []; C, pid = p3.load_cache(8)
    for f in range(len(J)):
        tr = J[f][2]; X = P5.features_v5(C, pid, tr); Zp = PCA(p3.NPC, random_state=0).fit(X[tr]).transform(X)
        for nama, F in (("BP", BP), ("P3_PCA", Zp)):
            fac = K.ash_coef(D[tr], F[tr])[1][1:]; rows.append(dict(fold=f, fitur=nama, **{f"median_fac_{i}": float(np.median(v)) for i, v in enumerate(fac)}, **{f"maks_fac_{i}": float(v.max()) for i, v in enumerate(fac)}))
    return pd.DataFrame(rows)


def check():
    n = 0; tr = J[0][2]; a = tr[:100]; F1 = make_tf("I2eb")(a); B = BP.copy(); B[np.setdiff1d(np.arange(N), a)] *= 7.0
    F2 = np.hstack([v4eb(B, a), AS]); assert np.allclose(F1[a], F2[a]); n += 1                          # data di luar himpunan pemasangan tidak mengubah fitur pemasangan
    Bo = K.ols_coef(D[tr], BP[tr]); Be, fac = K.ash_coef(D[tr], BP[tr]); assert np.allclose(Be[1:], fac[1:] * Bo[1:]); n += 1
    lam = np.median(fac, axis=1)
    Z = v4eb(BP, tr); assert np.allclose(Z[tr].mean(0), 0, atol=1e-8); n += 1                            # rata-rata residu terbakukan pasien latih = 0
    print(f"{n} pemeriksaan lulus; median faktor ash lipatan 0 (BP; usia, jenis kelamin, era x4): {np.round(lam[1:], 3)}; persentil 90 usia {np.percentile(fac[1], 90):.3f}", flush=True)


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "--check"; t0 = time.time()
    if mode == "--check":
        check(); sys.exit()
    json.dump(dict(prereg_sha256=sha(os.path.join(HERE, "PRAREGISTRASI_putaran51.md")), code_sha256=sha(__file__), common_sha256=sha(os.path.join(HERE, "round51_common.py"))), open(f"{OUT}/meta_nyata.json", "w"), indent=1)
    out = {}; nj = int(os.environ.get("NJOBS", 6))
    for model in ("I2eb", "N1eb"):
        res = Parallel(n_jobs=nj)(delayed(job)(model, f) for f in range(len(J))); P, cs = R49.to_matrix(res); out[model] = P; out[f"{model}__C"] = cs
        np.savez(f"{OUT}/nyata_prediksi.npz", **out); print(f"{model} selesai [{time.time() - t0:.0f}s]", flush=True)
    C, pid = p3.load_cache(8)
    res = Parallel(n_jobs=4, max_nbytes="1M")(delayed(job)("P3eb", f, C, pid) for f in range(len(J))); P, cs = R49.to_matrix(res); out["P3eb"] = P; out["P3eb__C"] = cs
    np.savez(f"{OUT}/nyata_prediksi.npz", **out); print(f"P3eb selesai [{time.time() - t0:.0f}s]", flush=True)
    lambdas().to_csv(f"{OUT}/nyata_lambda.csv", index=False); open(f"{OUT}/NYATA_SELESAI", "w").write(time.ctime()); print(f"selesai {time.time() - t0:.0f}s", flush=True)
