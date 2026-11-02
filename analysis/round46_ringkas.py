#!/usr/bin/env python3
"""Round 46: summary of R1 on the per-fold AUC against the pooled AUC (fixed rule of PRAREGISTRASI_putaran46.md). Reads hasil46/H1_per_replikasi.csv; writes hasil46/ringkas_R1.csv. CPU."""
import os, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "hasil46")
D = pd.read_csv(f"{OUT}/H1_per_replikasi.csv")


def keputusan(m, se):
    return "tidak ada bukti" if (m <= 0.520 and m + 1.645 * se <= 0.540) else ("kebocoran" if m > 0.55 else "tidak pasti")


rows = []
for v in ["C1", "V0", "V1", "V2", "V3", "V4"]:
    d = D[D.variant == v]; n = len(d); assert n == 100
    r = dict(variant=v, n_rep=n)
    for k in ("fold_mean", "pooled"):
        m, sd = d[k].mean(), d[k].std(ddof=1); se = sd / np.sqrt(n); r[f"{k}_mean"], r[f"{k}_sd"], r[f"{k}_se"], r[f"{k}_dec"] = m, sd, se, keputusan(m, se)
    r["selisih_fold_minus_pooled"] = r["fold_mean_mean"] - r["pooled_mean"]; r["keputusan_berubah"] = r["fold_mean_dec"] != r["pooled_dec"]; rows.append(r)
R = pd.DataFrame(rows); R.to_csv(f"{OUT}/ringkas_R1.csv", index=False)
print(R[["variant", "fold_mean_mean", "fold_mean_se", "fold_mean_dec", "pooled_mean", "pooled_dec", "selisih_fold_minus_pooled", "keputusan_berubah"]].round(4).to_string(index=False))
