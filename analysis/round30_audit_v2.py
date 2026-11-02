#!/usr/bin/env python3
"""Round 30: leakage audit v2 (PRAREGISTRASI_putaran30.md section 5). CPU only; read-only with respect to tuep_fresh.
A. perturbation test on the V4 pipeline (I2, P3, NM token) and the reference; B. global label permutation (I2-V4, P3-V4); D. the existing audit_kebocoran.py.
Usage: CUDA_VISIBLE_DEVICES="" python round30_audit_v2.py [n_permutations_I2] [n_permutations_P3]"""
import os, sys, subprocess, json
os.environ["CUDA_VISIBLE_DEVICES"] = ""; os.environ.setdefault("OMP_NUM_THREADS", "1")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, "/home/miftah_f/RTX-5090/tuep_fresh/code")
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score
import round28_analisis as A, round29_sensitivitas as S, round30_p3 as P3m, round30_p3_v5 as P5m, round30_nm as NMm
import normmamba as nmm, round17_p3 as p3

OUT = os.path.join(HERE, "hasil30"); Y0 = A.Y.copy(); J, N = A.J, A.N; res = []


def chk(ok, msg):
    res.append(dict(ok=bool(ok), check=msg)); print(("PASS " if ok else "FAIL ") + msg, flush=True)


# ---------- A. perturbation test
BP0 = A.BP.copy()
for f in (0, 17, 33):
    _, _, tr, te = J[f]
    # I2-V4
    F0, t0 = S.features(BP0, f, tr, te, "V4"); B1 = BP0.copy(); B1[te] = B1[te] * 5 + 3; F1, t1 = S.features(B1, f, tr, te, "V4")
    chk(np.allclose(F0[tr], F1[tr]) and np.array_equal(t0, t1), f"I2-V4 (lipatan {f}): mengubah data pasien uji tidak mengubah fitur pasien latih maupun indeks pelatihan")
    p_a = S.predict(BP0, f, tr, te, "V4", True); A.Y[te] = 1 - A.Y[te]; S.Y[te] = A.Y[te]; p_b = S.predict(BP0, f, tr, te, "V4", True); A.Y[:] = Y0; S.Y[:] = Y0
    chk(np.allclose(p_a, p_b), f"I2-V4 (lipatan {f}): membalik label pasien uji tidak mengubah prediksi uji")
    # P3-V5 (and V4 as a contrast)
    C, pid = P3m.p3.load_cache(8); X0 = P5m.features_v5(C, pid, tr); C2 = C.copy(); C2[np.isin(pid, te)] *= 4.0; X1 = P5m.features_v5(C2, pid, tr)
    chk(np.allclose(X0[tr], X1[tr]), f"P3-V5 (lipatan {f}): mengubah kovarians pasien uji tidak mengubah fitur pasien latih (referensi global hanya dari pasien latih)")
    q_a = P5m.predict_v5(C, pid, tr, te, True); P5m.Y[te] = 1 - P5m.Y[te]; q_b = P5m.predict_v5(C, pid, tr, te, True); P5m.Y[:] = Y0
    chk(np.allclose(q_a, q_b), f"P3-V5 (lipatan {f}): membalik label pasien uji tidak mengubah prediksi uji")
    # NM-V4 token
    t_a, s_a = NMm.tokens(nmm.XSEG, nmm.PID, tr, "V4"); X0s = nmm.XSEG.copy(); seg_te = np.isin(nmm.PID, te); Xp = X0s.copy(); Xp[seg_te] = Xp[seg_te] * 3 + 1
    t_b, s_b = NMm.tokens(Xp, nmm.PID, tr, "V4")
    chk(all(np.array_equal(t_a[i], t_b[i]) for i in tr) and np.allclose(s_a[tr], s_b[tr]), f"NM-V4 (lipatan {f}): mengubah segmen pasien uji tidak mengubah token dan ringkasan pasien latih")
    Yb = nmm.Y.copy(); nmm.Y[te] = 1 - nmm.Y[te]; t_c, s_c = NMm.tokens(nmm.XSEG, nmm.PID, tr, "V4"); nmm.Y[:] = Yb
    chk(all(np.array_equal(t_a[i], t_c[i]) for i in range(N)) and np.allclose(s_a, s_c), f"NM-V4 (lipatan {f}): membalik label pasien uji tidak mengubah token siapa pun")
# the V4 coefficients indeed contain no test information: B from all training patients only
_, _, tr, te = J[5]; fm = np.isin(nmm.PID, tr); chk(not (set(np.unique(nmm.PID[fm])) & set(te)), "NM-V4: segmen yang dipakai memasang koefisien hanya milik pasien latih")
# orthogonality of the V4 residuals on training patients (property of standard covariate regression)
_, _, tr, te = J[9]; F0, _ = S.features(BP0, 9, tr, te, "V4"); resid_corr = np.abs(np.linalg.lstsq(A.D_I2[tr], F0[tr], rcond=None)[0]).max()
chk(resid_corr < 1e-8, f"I2-V4: fitur pasien latih ortogonal terhadap matriks desain (koefisien regresi sisa maks {resid_corr:.1e}); pada V0 hanya kontrol yang ortogonal")
F00, _ = S.features(BP0, 9, tr, te, "V0"); ctrl = tr[A.Y[tr] == 0]; epi = tr[A.Y[tr] == 1]
cc = np.abs(np.linalg.lstsq(A.D_I2[ctrl], F00[ctrl], rcond=None)[0]).max(); ce = np.abs(np.linalg.lstsq(A.D_I2[epi], F00[epi], rcond=None)[0]).max()
chk(cc < 1e-8 and ce > 1e-3, f"I2-V0: kontrol latih ortogonal terhadap desain ({cc:.1e}) tetapi epilepsi latih tidak ({ce:.2f}): asimetri yang menjelaskan mengapa V0 dapat membocorkan kovariat")

# ---------- B. permutasi label global
def perm_auc(kind, seed):
    rng = np.random.default_rng(seed); yp = rng.permutation(Y0); A.Y[:] = yp; S.Y[:] = yp; P3m.Y[:] = yp; P5m.Y[:] = yp
    if kind == "I2V4":
        out = [S.predict(BP0, f, tr, te, "V4", True) for f, (_, _, tr, te) in enumerate(J)]
    else:
        C, pid = P3m.p3.load_cache(1); out = [P5m.predict_v5(C, pid, tr, te, True) for _, _, tr, te in J]
    M = A.c17.to_matrix(J, out); A.Y[:] = Y0; S.Y[:] = Y0; P3m.Y[:] = Y0; P5m.Y[:] = Y0
    return float(np.mean([roc_auc_score(yp, p) for p in M]))
n1 = int(sys.argv[1]) if len(sys.argv) > 1 else 100; n2 = int(sys.argv[2]) if len(sys.argv) > 2 else 100
for kind, n, nj in (("I2V4", n1, 12), ("P3V5", n2, 12)):
    vals = []
    # joblib uses separate processes: each worker changes its own local copy
    vals = Parallel(n_jobs=nj)(delayed(perm_auc)(kind, 30900 + i) for i in range(n))
    vals = np.array(vals); se = vals.std(ddof=1) / np.sqrt(n)
    chk(abs(vals.mean() - 0.5) < max(3 * se, 0.01) and np.percentile(vals, 95) < 0.58, f"permutasi label global {kind}: {n} permutations, mean AUC {vals.mean():.4f} (SE {se:.4f}), persentil 95 {np.percentile(vals, 95):.3f}")
    pd.DataFrame(dict(perm=range(n), auc=vals)).to_csv(f"{OUT}/audit_permutasi_label_{kind}.csv", index=False)

# ---------- D. audit lama
r = subprocess.run([sys.executable, "/home/miftah_f/RTX-5090/tuep_fresh/code/audit_kebocoran.py"], capture_output=True, text=True, cwd="/home/miftah_f/RTX-5090/tuep_fresh/code")
lines = [ln for ln in r.stdout.splitlines() if ln.startswith(("PASS", "FAIL"))]; npass = sum(ln.startswith("PASS") for ln in lines); nfail = sum(ln.startswith("FAIL") for ln in lines)
open(f"{OUT}/audit_kebocoran_lama.log", "w").write(r.stdout + r.stderr[-2000:]); chk(nfail == 0 and npass >= 16, f"the existing audit_kebocoran.py was rerun: {npass} PASS, {nfail} FAIL")
pd.DataFrame(res).to_csv(f"{OUT}/audit_v2_hasil.csv", index=False)
print(f"\n{sum(x['ok'] for x in res)} of {len(res)}checks passed")
