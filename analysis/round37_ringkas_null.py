#!/usr/bin/env python3
"""Summary of the paired Tang null (L vs Lp; rules R2 and R1 in PRAREGISTRASI_putaran37_38.md) from hasil37/tang_{L,Lp}_null_replikasi.csv. CPU."""
import os, numpy as np, pandas as pd
from scipy import stats
H = os.path.dirname(os.path.abspath(__file__)); L = pd.read_csv(f"{H}/hasil37/tang_L_null_replikasi.csv"); P = pd.read_csv(f"{H}/hasil37/tang_Lp_null_replikasi.csv"); rows = []
for scen in ("H1", "H2a"):
    a = L[L.scenario == scen].set_index("rep").auc.astype(float); b = P[P.scenario == scen].set_index("rep").auc.astype(float); idx = a.index.intersection(b.index); a, b = a[idx], b[idx]; d = (b - a).values; n = len(d); se = d.std(ddof=1) / np.sqrt(n); m = d.mean()
    r2 = "tidak ada bukti kebocoran" if (m <= 0.01 and m + 1.645 * se <= 0.03) else ("kebocoran" if m > 0.03 else "pemanfaatan kecil/tidak pasti")
    rows.append(dict(scenario=scen, n=n, auc_L=a.mean(), auc_Lp=b.mean(), sd_Lp=b.std(ddof=1), delta=m, se=se, delta_p95=m + 1.645 * se, t_p=stats.ttest_1samp(d, 0).pvalue, keputusan_R2=r2, R1_mean_plus_1645SE=(b.mean() + 1.645 * b.std(ddof=1) / np.sqrt(n)) if scen == "H1" else np.nan))
S = pd.DataFrame(rows); S.to_csv(f"{H}/hasil37/tang_paired_ringkasan.csv", index=False); print(S.round(4).to_string(index=False))
