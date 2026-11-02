#!/usr/bin/env python3
"""Round 48: independent check (second code). (A) complete output: 300 rows, 6 per fold 0--49, all passed; (B) invariance of the V4 features of training patients to changes in the test data, reimplemented
with plain numpy (own least squares) on folds 3, 22, 41 and compared with S.features; (C) folds 0, 17, 33 match hasil30/audit_v2_hasil.csv. CPU."""
import os, sys, re
os.environ["CUDA_VISIBLE_DEVICES"] = ""
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, "/home/miftah_f/RTX-5090/tuep_fresh/code")
import numpy as np, pandas as pd
import round28_analisis as A, round29_sensitivitas as S
ok = fail = 0; GRP = {"A": [0, 0], "B": [0, 0], "C": [0, 0]}; g = "A"
def chk(name, cond):
    global ok, fail
    ok += bool(cond); fail += (not cond); GRP[g][0 if cond else 1] += 1; print(("PASS  " if cond else "FAIL  ") + f"[{g}] " + name, flush=True)
R = pd.read_csv(os.path.join(HERE, "hasil48/audit_lipatan_0_49.csv")); R["lip"] = R.check.str.extract(r"lipatan (\d+)")[0].astype(int)
chk("300 baris pemeriksaan", len(R) == 300); chk("semua pemeriksaan lulus", bool(R.ok.all()))
chk("6 checks per fold for folds 0--49", sorted(R.lip.unique()) == list(range(50)) and (R.groupby("lip").size() == 6).all())
chk("each path (I2-V4, P3-V5, NM-V4) 100 checks", all(R.check.str.startswith(k).sum() == 100 for k in ("I2-V4", "P3-V5", "NM-V4")))
g = "B"; BP = A.BP.copy(); D = A.D_I2; J = A.J
def z_own(X, tr):
    Bc = np.linalg.lstsq(D[tr], X[tr], rcond=None)[0]; res = X - D @ Bc; sd = res[tr].std(0, ddof=np.linalg.matrix_rank(D[tr])); return res / sd
for f in (3, 22, 41):
    _, _, tr, te = J[f]; X1 = BP.copy(); X1[te] = X1[te] * 5 + 3; z0 = z_own(BP, tr); z1 = z_own(X1, tr)
    chk(f"fold {f}: own V4: training features unchanged when the test data are changed (max difference {np.abs(z0[tr] - z1[tr]).max():.1e})", np.abs(z0[tr] - z1[tr]).max() < 1e-9)
    F0, _ = S.features(BP, f, tr, te, "V4"); chk(f"fold {f}: training features S.features(V4) = own implementation on the first 114 columns", np.allclose(F0[tr][:, :114], z0[tr], atol=1e-7))
g = "C"; V = pd.read_csv(os.path.join(HERE, "hasil30/audit_v2_hasil.csv"))
for f in (0, 17, 33):
    a_ = R[R.lip == f].set_index("check").ok; b_ = V[V.check.str.contains(rf"fold {f}\)")].set_index("check").ok
    chk(f"fold {f}: 6 checks of round 48 equal round 30 (all passed)", len(a_) == 6 and len(b_) == 6 and bool(a_.all()) and bool(b_.all()) and set(a_.index) == set(b_.index))
print(f"\n{ok} PASS, {fail} FAIL"); print({k: tuple(v) for k, v in GRP.items()})
if not fail: print(f"ALL {sum(1 for v in GRP.values() if v[0] and not v[1])} CHECK GROUPS PASSED")
sys.exit(1 if fail else 0)
