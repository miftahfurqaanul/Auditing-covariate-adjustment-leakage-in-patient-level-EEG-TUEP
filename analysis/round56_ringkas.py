#!/usr/bin/env python3
"""Putaran 56: ringkasan sesuai PRAREGISTRASI_putaran56.md (di-hash sebelum hasil dilihat). Tidak melatih model."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
from scipy import stats
from round49_kalibrasi import decide, THR
O = os.path.join(HERE, "hasil56"); R = pd.read_csv(os.path.join(O, "replikasi.csv")); Dg = pd.read_csv(os.path.join(O, "diagnostik_desain.csv"))
assert len(R) == 100; LAB = {0: "no evidence", 1: "uncertain", 2: "leakage"}
dec = lambda x, rule: LAB[int(decide(np.array([np.mean(x)]), np.array([np.std(x, ddof=1) / np.sqrt(len(x))]), THR[rule]["asli"])[0])]
for v in ("C1", "V0", "VR", "V4"):
    print(f"{v}: AUC gabungan {100 * R[v].mean():.1f} (SD {100 * R[v].std(ddof=1):.1f}); per lipatan {100 * R[v + '_fold'].mean():.1f}; R1 {dec(R[v], 'R1')}")
for a, b in (("V0", "C1"), ("VR", "C1"), ("V4", "C1")):
    d = R[a] - R[b]; print(f"{a} - {b}: {100 * d.mean():+.1f} pp (SD {100 * d.std(ddof=1):.1f}); R2 {dec(d, 'R2')}")
d = 100 * (R.VR - R.V0); t, p = stats.ttest_1samp(d, 0); ci = stats.t.interval(0.95, len(d) - 1, d.mean(), d.std(ddof=1) / np.sqrt(len(d)))
print(f"\nKONTRAS UTAMA VR - V0: {d.mean():+.2f} pp (95% CI {ci[0]:+.2f} to {ci[1]:+.2f}); t = {t:.2f}, p = {p:.3g}")
pd.DataFrame([dict(contrast="VR-V0", diff=d.mean(), lo=ci[0], hi=ci[1], t=t, p=p)]).to_csv(os.path.join(O, "kontras_utama.csv"), index=False)
print("\nDiagnostik desain (rata-rata atas 50 lipatan):"); print(Dg.groupby("ref")[["n", "rank", "min_era", "max_leverage", "frac_epi"]].agg(["mean", "min", "max"]).round(3).to_string())
