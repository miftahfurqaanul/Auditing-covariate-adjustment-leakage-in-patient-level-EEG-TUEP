#!/usr/bin/env python3
"""H40-0: reproduction of the pooled AUC of round 39 by round 40. Usage: python round40_reproduksi.py <file hasil40/*.csv> <file hasil39/*.csv> [threshold]. Exits with code 1 if the maximum difference > threshold (default 1e-9)."""
import sys, pandas as pd, numpy as np
a = pd.read_csv(sys.argv[1]); b = pd.read_csv(sys.argv[2]); thr = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-9
K = ["dist", "scen", "n", "m", "p", "imb", "s", "sig", "cfrac", "cscale", "rep", "clf", "transform", "adjust"]
m = a[K + ["auc"]].merge(b[K + ["auc"]], on=K, how="inner", suffixes=("_40", "_39")); m["d"] = (m["auc_40"] - m["auc_39"]).abs()
print(f"round 40 rows: {len(a)}; paired with round 39: {len(m)}; maximum difference {m['d'].max():.3e}; mean {m['d'].mean():.3e}; rows > threshold {int((m['d'] > thr).sum())}")
if len(m) < len(a):
    print("WARNING: some round 40 rows have no partner in round 39 (fewer replicates in round 39?)")
sys.exit(0 if (len(m) > 0 and m["d"].max() <= thr) else 1)
