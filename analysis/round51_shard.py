#!/usr/bin/env python3
"""Putaran 51: pembungkus eksekusi paralel (BUKAN analisis): membagi daftar unit round51_plasmode.UNITS atau (rep, gamma) Bi-LSTM menjadi S shard (indeks % S == k)
dan menulis ke berkas per-shard; setelah semua shard selesai berkas digabung dengan `merge` tanpa mengubah baris. Kode analisis (round51_plasmode.py, round51_tang.py) tidak diubah.
Pemakaian: python round51_shard.py plasmode <tuep|emotiv> <RF|XGB> <S> <k>
           python round51_shard.py tang <S> <k> [n_rep]
           python round51_shard.py merge <tuep|emotiv> <RF|XGB|Tang>   (menggabungkan hasil51/<coh>_<fam>[_s*].csv ke <coh>_<fam>.csv; menolak bila ada unit ganda)"""
import os, sys, time, glob
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); OUT = os.path.join(HERE, "hasil51")
kind = sys.argv[1]
if kind == "merge":
    import pandas as pd
    coh, fam = sys.argv[2], sys.argv[3]; main = f"{OUT}/{coh}_{fam}.csv"; fs = sorted(glob.glob(f"{OUT}/{coh}_{fam}_s*.csv")); parts = [pd.read_csv(f) for f in fs] + ([pd.read_csv(main)] if os.path.exists(main) else [])
    D = pd.concat(parts, ignore_index=True); key = ["rep", "cond", "gamma", "transform", "adjust"]; assert not D.duplicated(key).any(), "unit ganda"
    D.to_csv(main + ".tmp", index=False); os.replace(main + ".tmp", main); print(f"{main}: {len(D)} baris dari {len(parts)} berkas"); sys.exit()
import pandas as pd
if kind == "plasmode":
    coh, fam, S, k = sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]); sys.argv = [sys.argv[0], "--shard", coh, fam]
    import round51_plasmode as M
    path = f"{OUT}/{coh}_{fam}_s{k}.csv"; mine = [u for i, u in enumerate(M.UNITS) if i % S == k]
    done = set(map(tuple, pd.read_csv(path, usecols=["rep", "cond", "gamma"]).drop_duplicates().values.tolist())) if os.path.exists(path) else set()
    units = [u for u in mine if u not in done]; t0 = time.time(); print(f"{coh} {fam} shard {k}/{S}: {len(units)} unit tersisa dari {len(mine)}", flush=True)
    if fam == "XGB":
        for i, (r, c, g) in enumerate(units):
            pd.DataFrame(M.unit(fam, r, c, g)).to_csv(path, mode="a", header=not os.path.exists(path), index=False)
            if i % 20 == 19:
                print(f"{i + 1}/{len(units)} [{time.time() - t0:.0f}s]", flush=True)
    else:
        from joblib import Parallel, delayed
        nj = int(os.environ.get("NJOBS", 12))
        for i in range(0, len(units), nj * 2):
            for rr in Parallel(n_jobs=nj)(delayed(M.unit)(fam, r, c, g) for r, c, g in units[i:i + nj * 2]):
                pd.DataFrame(rr).to_csv(path, mode="a", header=not os.path.exists(path), index=False)
            print(f"{min(i + nj * 2, len(units))}/{len(units)} [{time.time() - t0:.0f}s]", flush=True)
elif kind == "tang":
    S, k = int(sys.argv[2]), int(sys.argv[3]); nrep = int(sys.argv[4]) if len(sys.argv) > 4 else 20; sys.argv = [sys.argv[0], "--shard"]
    import numpy as np, torch
    import round51_tang as T51
    dev = torch.device("cuda"); torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    T, X, subj, owner, pi, seqs = T51.G.tang_setup(); cache = {}; path = f"{OUT}/tuep_Tang_s{k}.csv"; t0 = time.time()
    done = set(map(tuple, pd.read_csv(path, usecols=["rep", "gamma"]).drop_duplicates().values.tolist())) if os.path.exists(path) else set()
    for rep in range(nrep):
        if rep % S != k:
            continue
        for g in T51.GAMMAS:
            if (rep, g) in done:
                continue
            pd.DataFrame(T51.unit(rep, g, T, X, subj, owner, pi, seqs, dev, cache)).to_csv(path, mode="a", header=not os.path.exists(path), index=False); print(f"Tang shard {k}/{S} rep {rep} gamma {g} [{time.time() - t0:.0f}s]", flush=True)
print(f"selesai", flush=True)
