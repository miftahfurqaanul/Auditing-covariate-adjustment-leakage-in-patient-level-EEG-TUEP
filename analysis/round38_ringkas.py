#!/usr/bin/env python3
"""Summary of the paired NM null (N0 vs V4) from hasil30/nm_paired_replikasi.csv (data-type fix in the summary step of round38_nm_paired.py; rules R2 and R1 unchanged). CPU."""
import os, numpy as np, pandas as pd
from scipy import stats
H = os.path.dirname(os.path.abspath(__file__)); D = pd.read_csv(f"{H}/hasil30/nm_paired_replikasi.csv"); D["auc"] = D.auc.astype(float); rows = []
for scen in ("H1", "H2a"):
    g = D[D.scenario == scen].pivot_table(index="rep", columns="variant", values="auc").dropna(); a = g["N0"].values.astype(float); b = g["V4"].values.astype(float); d = b - a; n = len(d); se = d.std(ddof=1) / np.sqrt(n); m = d.mean()
    r2 = "tidak ada bukti kebocoran" if (m <= 0.01 and m + 1.645 * se <= 0.03) else ("kebocoran" if m > 0.03 else "pemanfaatan kecil/tidak pasti")
    rows.append(dict(scenario=scen, n=n, auc_N0=a.mean(), sd_N0=a.std(ddof=1), auc_V4=b.mean(), sd_V4=b.std(ddof=1), delta=m, se=se, delta_p95=m + 1.645 * se, t_p=stats.ttest_1samp(d, 0).pvalue, corr_N0_V4=float(np.corrcoef(a, b)[0, 1]), keputusan_R2=r2,
                     R1_mean_plus_1645SE=(b.mean() + 1.645 * b.std(ddof=1) / np.sqrt(n)) if scen == "H1" else np.nan))
S = pd.DataFrame(rows); S.to_csv(f"{H}/hasil30/nm_paired_ringkasan.csv", index=False); print(S.round(4).to_string(index=False))
