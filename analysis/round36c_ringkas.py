#!/usr/bin/env python3
"""Summary of the paired Rohira null from hasil36/paired_rohira_null.csv (data-type fix in the summary step of round36c_paired.py; rules R2 and R1 unchanged, PRAREGISTRASI_putaran37_38.md). CPU."""
import os, numpy as np, pandas as pd
from scipy import stats
H = os.path.dirname(os.path.abspath(__file__)); D = pd.read_csv(f"{H}/hasil36/paired_rohira_null.csv"); D["auc"] = D.auc.astype(float)
P = D.pivot_table(index=["scenario", "rep"], columns="variant", values="auc").reset_index(); P["delta"] = (P["RF-logpat"] - P["RF-none"]).astype(float); rows = []
for scen, g in P.groupby("scenario"):
    d = g.delta.values.astype(float); n = len(d); se = d.std(ddof=1) / np.sqrt(n); m = d.mean(); a = g["RF-logpat"].values.astype(float); n_ = g["RF-none"].values.astype(float)
    r2 = "tidak ada bukti kebocoran" if (m <= 0.01 and m + 1.645 * se <= 0.03) else ("kebocoran" if m > 0.03 else "pemanfaatan kecil/tidak pasti")
    rows.append(dict(scenario=scen, n=n, auc_none=n_.mean(), auc_logpat=a.mean(), sd_logpat=a.std(ddof=1), delta=m, se=se, delta_p95=m + 1.645 * se, t_p=stats.ttest_1samp(d, 0).pvalue, keputusan_R2=r2,
                     R1_mean_plus_1645SE=(a.mean() + 1.645 * a.std(ddof=1) / np.sqrt(n)) if scen == "H1" else np.nan))
S = pd.DataFrame(rows); S.to_csv(f"{H}/hasil36/paired_rohira_ringkasan.csv", index=False); print(S.round(4).to_string(index=False))
