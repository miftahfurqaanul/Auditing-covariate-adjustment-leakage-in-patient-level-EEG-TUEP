#!/usr/bin/env python3
"""Round 50A: extension of the round-42 plasmode replicates (RF/XGB, H1, gamma 0 and 1) without changing the round-42 code. See PRAREGISTRASI_putaran50.md.
Usage: python round50_A.py --check <RF|XGB> | --run <RF|XGB>   (resumable; output hasil50/A_<fam>.csv; hasil42 is not touched)"""
import os, sys, time
MODE, FAM = sys.argv[1], sys.argv[2]
sys.argv = [sys.argv[0], "--run", FAM]                                     # round42_cpu reads --run on import (GPU only for XGB)
import numpy as np, pandas as pd
from joblib import Parallel, delayed
import round42_cpu as P

OUT = os.path.join(P.C.HERE, "hasil50"); os.makedirs(OUT, exist_ok=True)
START = {"RF": 30, "XGB": 20}; NREP = 120; GAM = (0.0, 1.0); FC = [f"fold_auc_{i}" for i in range(5)]

if __name__ == "__main__":
    t0 = time.time()
    if MODE == "--check":
        new = pd.DataFrame(P.unit_window(FAM, 0, "H1", 0.0)); old = pd.read_csv(os.path.join(P.OUT, f"{FAM}.csv"))
        old = old[(old.rep == 0) & (old.cond == "H1") & (old.gamma == 0.0)]
        m = new.merge(old, on=["transform", "adjust"], suffixes=("", "_42")); assert len(m) == len(new) == 6
        dmax = max(np.abs(m[c] - m[c + "_42"]).max() for c in ["auc"] + FC)
        print(f"{FAM} replicate 0 H1 gamma 0: {len(m)} pipelines, maximum difference from hasil42 = {dmax:.2e} [{time.time() - t0:.0f}s]", flush=True)
        assert dmax < (1e-9 if FAM == "RF" else 1e-6)
    elif MODE == "--run":
        path = f"{OUT}/A_{FAM}.csv"
        done = set(map(tuple, pd.read_csv(path, usecols=["rep", "cond", "gamma"]).drop_duplicates().values.tolist())) if os.path.exists(path) else set()
        units = [(r, "H1", g) for r in range(START[FAM], NREP) for g in GAM if (r, "H1", g) not in done]; print(f"{FAM}: {len(units)} units left", flush=True)
        if FAM == "XGB":
            for i, (r, c, g) in enumerate(units):
                pd.DataFrame(P.unit_window(FAM, r, c, g)).to_csv(path, mode="a", header=not os.path.exists(path), index=False)
                if i % 10 == 9:
                    print(f"{i + 1}/{len(units)} [{time.time() - t0:.0f}s]", flush=True)
        else:
            nj = int(os.environ.get("NJOBS", 16))
            for i in range(0, len(units), nj * 2):
                chunk = units[i:i + nj * 2]; res = Parallel(n_jobs=nj)(delayed(P.unit_window)(FAM, r, c, g) for r, c, g in chunk)
                for rr in res:
                    pd.DataFrame(rr).to_csv(path, mode="a", header=not os.path.exists(path), index=False)
                print(f"{min(i + nj * 2, len(units))}/{len(units)} [{time.time() - t0:.0f}s]", flush=True)
    print(f"done {time.time() - t0:.0f}s", flush=True)
