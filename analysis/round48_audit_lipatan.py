#!/usr/bin/env python3
"""Round 48: dynamic leakage audit (perturbation test, part A of round30_audit_v2.py) on the given canonical folds (PRAREGISTRASI_putaran48.md; SHA-256 recorded before the run).
CPU only (CUDA_VISIBLE_DEVICES=""); read-only with respect to tuep_fresh. The check code is identical to round30_audit_v2.py (part A); only the list of folds is extended.
Usage: python round48_audit_lipatan.py --timing | --run [first_fold last_fold]"""
import os, sys, subprocess, json
os.environ["CUDA_VISIBLE_DEVICES"] = ""; os.environ.setdefault("OMP_NUM_THREADS", "1")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, "/home/miftah_f/RTX-5090/tuep_fresh/code")
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score
import round28_analisis as A, round29_sensitivitas as S, round30_p3 as P3m, round30_p3_v5 as P5m, round30_nm as NMm
import normmamba as nmm, round17_p3 as p3

OUT = os.path.join(HERE, "hasil48"); os.makedirs(OUT, exist_ok=True); Y0 = A.Y.copy(); J, N = A.J, A.N; res = []


def chk(ok, msg):
    res.append(dict(ok=bool(ok), check=msg)); print(("PASS " if ok else "FAIL ") + msg, flush=True)


MODE = sys.argv[1] if len(sys.argv) > 1 else "--timing"
if MODE == "--timing": FOLDS = [0]
else: FOLDS = list(range(int(sys.argv[2]), int(sys.argv[3]))) if len(sys.argv) > 3 else list(range(50))
import time; T0 = time.time()

BP0 = A.BP.copy()
for f in FOLDS:
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
print(f"\n{sum(x['ok'] for x in res)} of {len(res)} checks passed; folds {FOLDS[0]}..{FOLDS[-1]} ({len(FOLDS)}); {time.time() - T0:.0f} s")
if MODE != "--timing": pd.DataFrame(res).to_csv(os.path.join(HERE, "hasil48", f"audit_lipatan_{FOLDS[0]}_{FOLDS[-1]}.csv"), index=False)
