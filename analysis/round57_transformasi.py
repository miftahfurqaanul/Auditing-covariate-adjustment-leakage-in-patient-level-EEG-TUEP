#!/usr/bin/env python3
"""Putaran 57 (M2): sel transformasi TANPA penyesuaian, (symlog|log|rank, none), untuk RF/XGB (TUEP putaran 42+50A, Emotiv 50B) dan LR jendela (putaran 53),
dengan fungsi, benih, lipatan, dan pasien latih yang sama, sehingga empat sel raw, raw+A, T, T+A berpasangan. Lihat PRAREGISTRASI_putaran57_60.md.
Pemakaian: python round57_transformasi.py --check <RF|XGB|LRw> | --run <RF|XGB|LRw>   (RF/LRw: CPU; XGB: GPU berurutan; resumable; keluaran hasil57/<fam>.csv)"""
import os, sys, time
FAM = sys.argv[2] if len(sys.argv) > 2 else "RF"
if FAM != "XGB":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[_v] = "1"
os.environ["PYTHONWARNINGS"] = "ignore"
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, "/home/miftah_f/RTX-5090/tuep_fresh/code")
import warnings; warnings.filterwarnings("ignore")
_argv = sys.argv; sys.argv = [_argv[0]] + (["--run", "XGB"] if FAM == "XGB" else [])
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score
import round42_common as C
import round42_cpu as P42
import round50_emotiv as E
import round53_lr_mentah as L53
sys.argv = _argv
OUT = os.path.join(HERE, "hasil57"); os.makedirs(OUT, exist_ok=True)
PIPES = [("symlog", "none"), ("log", "none"), ("rank", "none")]; FC = [f"fold_auc_{i}" for i in range(5)]
# LR jendela: nilai putaran 53 tidak tereproduksi persis (solver; selisih <= 0.004 AUC), maka keempat sel dihitung ulang dalam run yang sama
PIPES_LRW = [("raw", "none"), ("raw", "pat")] + [(t, a) for t in ("symlog", "log", "rank") for a in ("none", "pat")]
pipes_for = lambda fam: PIPES_LRW if fam == "LRw" else PIPES


def stored(fam, cohort):
    if fam == "LRw":
        return pd.read_csv(os.path.join(HERE, "hasil53", f"LRw_{cohort}.csv"))
    if cohort == "tuep":
        a = pd.read_csv(os.path.join(HERE, "hasil42", f"{fam}.csv")); a = a[a.cond == "H1"]
        return pd.concat([a, pd.read_csv(os.path.join(HERE, "hasil50", f"A_{fam}.csv"))])
    return pd.read_csv(os.path.join(HERE, "hasil50", f"B_{fam}.csv"))


def units(fam):
    out = []
    for cohort in ("tuep", "emotiv"):
        s = stored(fam, cohort)[["rep", "gamma"]].drop_duplicates()
        out += [(cohort, int(r), float(g)) for r, g in s.values]
    return sorted(set(out))


def clf(fam, cohort, Z, owner, y, tr, te, seed, iseed, a):
    if fam == "LRw":
        return L53.lrw_clf(Z, owner, y, tr, te, seed, iseed, a["split"])
    return (P42.window_clf if cohort == "tuep" else E.window_clf)(fam, Z, owner, y, tr, te, seed)


def unit(fam, cohort, rep, gamma, pipes=None):
    pipes = pipes or pipes_for(fam); a = L53.api(cohort); N = a["N"]; y = a["labels"](rep, gamma); folds = a["folds"](rep, gamma, y); X, owner = a["windows"](a["perm"](rep)); P = {p: np.zeros(N) for p in pipes}
    for fi, (tr, te) in enumerate(folds):
        seed = a["seed"](rep, gamma, fi); iseed = a["kseed"]("lrw_inner", rep, gamma, fi); trw = np.isin(owner, tr); cache = {}
        for t, ad in pipes:
            if t not in cache:
                cache[t] = a["tf"](t, X[trw])(X)
            P[(t, ad)][te] = clf(fam, cohort, a["adj"](cache[t], owner, tr, ad), owner, y, tr, te, seed, iseed, a)
    return [dict(fam=fam, cohort=cohort, rep=rep, cond="H1", gamma=gamma, transform=t, adjust=ad, auc=float(roc_auc_score(y, p)), **{f"fold_auc_{i}": v for i, v in enumerate(C.fold_aucs(y, p, folds))}) for (t, ad), p in P.items()]


def check(fam):
    t0 = time.time(); tol = {"XGB": 1e-6, "RF": 1e-9, "LRw": 1e-2}[fam]
    for cohort in ("tuep", "emotiv"):
        new = pd.DataFrame(unit(fam, cohort, 0, 0.0, [("raw", "none"), ("raw", "pat")])); old = stored(fam, cohort); old = old[(old.rep == 0) & (old.gamma == 0.0)]
        m = new.merge(old, on=["transform", "adjust"], suffixes=("", "_o")); assert len(m) == 2, len(m)
        d = max(np.abs(m[c] - m[c + "_o"]).max() for c in ["auc"] + FC); assert d < tol, (fam, cohort, d)
        print(f"{fam} {cohort}: reproduksi raw-none dan raw-pat rep 0 gamma 0 selisih maks {d:.1e} [{time.time() - t0:.0f}s]", flush=True)
    t1 = time.time(); unit(fam, "tuep", 1, 1.0); print(f"waktu satu unit TUEP {time.time() - t1:.0f}s (AUC tidak dicetak); unit total {len(units(fam))}", flush=True)


def run(fam):
    path = os.path.join(OUT, f"{fam}.csv"); t0 = time.time()
    done = set(map(tuple, pd.read_csv(path, usecols=["cohort", "rep", "gamma"]).drop_duplicates().values.tolist())) if os.path.exists(path) else set()
    todo = [u for u in units(fam) if (u[0], u[1], u[2]) not in done]; print(f"{fam}: {len(todo)} unit tersisa", flush=True)
    if fam == "XGB":
        for i, u in enumerate(todo):
            pd.DataFrame(unit(fam, *u)).to_csv(path, mode="a", header=not os.path.exists(path), index=False)
            if i % 20 == 19: print(f"  {i + 1}/{len(todo)} [{time.time() - t0:.0f}s]", flush=True)
    else:
        nj = int(os.environ.get("NJOBS", 22))
        for i in range(0, len(todo), nj * 2):
            for rr in Parallel(n_jobs=nj)(delayed(unit)(fam, *u) for u in todo[i:i + nj * 2]):
                pd.DataFrame(rr).to_csv(path, mode="a", header=not os.path.exists(path), index=False)
            print(f"  {min(i + nj * 2, len(todo))}/{len(todo)} [{time.time() - t0:.0f}s]", flush=True)
    print(f"selesai {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    (check if sys.argv[1] == "--check" else run)(FAM)
