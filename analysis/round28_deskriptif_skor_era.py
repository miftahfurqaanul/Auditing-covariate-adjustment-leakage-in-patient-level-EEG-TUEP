#!/usr/bin/env python3
"""Mean score by era and class (descriptive; supports the partial test). Writes hasil/4_skor_rata_menurut_era_kelas.csv. CPU only."""
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import round28_analisis as A
P = A.load_models(); rows = []
for m in ("Floor", "C1", "I2", "F2", "LaBraM-FT"):
    s = P[m].mean(0)
    for e in ["<=2009", "2010", "2011", "2012", "2013"]:
        for k in (0, 1):
            i = (A.ERA == e) & (A.Y == k); rows.append(dict(model=m, era=e, cls="epi" if k else "non", n=int(i.sum()), mean=round(float(s[i].mean()), 3)))
d = pd.DataFrame(rows).pivot_table(index=["model", "era"], columns="cls", values=["mean", "n"]); d.to_csv(os.path.join(A.OUT, "4_skor_rata_menurut_era_kelas.csv")); print(d.to_string())
