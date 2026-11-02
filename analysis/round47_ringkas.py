#!/usr/bin/env python3
"""Round 47: summary of the AUC of F2 without NM (rule from PRAREGISTRASI_putaran47.md). Reads hasil47/auc_per_lipatan.npz; writes hasil47/ringkas_f2_tanpa_nm.csv and ringkas_selisih.csv. CPU."""
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "hasil47"); sys.path.insert(0, HERE); sys.path.insert(0, "/home/miftah_f/RTX-5090/tuep_fresh/code")
import numpy as np, pandas as pd
from scipy import stats
import round29_sensitivitas as S
Z = np.load(f"{OUT}/auc_per_lipatan.npz"); names = ["F2_tanpa_NM", "F2_adj", "C1", "Floor", "I2_adj", "P3_adj"]; tc = stats.t.ppf(0.975, 49)
rows = []
for k in names:
    f = Z[f"fold_{k}"]; se = np.sqrt(f.var(ddof=1) * (1 / 50 + 40 / 160)); rows.append(dict(model=k, auc_fold=100 * f.mean(), lo=100 * (f.mean() - tc * se), hi=100 * (f.mean() + tc * se), sd=100 * f.std(ddof=1), auc_pooled=100 * float(Z[f"pooled_{k}"])))
pd.DataFrame(rows).to_csv(f"{OUT}/ringkas_f2_tanpa_nm.csv", index=False)
dr = []
for k in ["F2_adj", "C1", "Floor", "I2_adj"]:
    m, lo, hi, p = S.nb_diff(Z["fold_F2_tanpa_NM"], Z[f"fold_{k}"]); dr.append(dict(kontras=f"F2_tanpa_NM - {k}", selisih=m, lo=lo, hi=hi, p=p, primer=k != "I2_adj"))
D = pd.DataFrame(dr); pr = D[D.primer].p.values; o = np.argsort(pr); h = np.empty(3); run = 0
for r_, i in enumerate(o): run = max(run, (3 - r_) * pr[i]); h[i] = min(1, run)
D["holm"] = np.nan; D.loc[D.primer, "holm"] = h; D.to_csv(f"{OUT}/ringkas_selisih.csv", index=False)
print(pd.DataFrame(rows).round(3).to_string(index=False)); print(D.round(4).to_string(index=False))
