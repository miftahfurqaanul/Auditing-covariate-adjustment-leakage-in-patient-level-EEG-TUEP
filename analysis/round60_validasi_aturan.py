#!/usr/bin/env python3
"""Putaran 60 (paket 3): replikasi simulasi BARU (rep 1000--1199; benih belum pernah dipakai) untuk menguji aturan beku pada audit independen.
Sel simulasi memakai round39_simulasi_kebocoran.run_rep (RF, AUC gabungan); sel referensi kontrol memakai round55_referensi_kontrol.unit. Lihat PRAREGISTRASI_putaran57_60.md.
Pemakaian: python round60_validasi_aturan.py --check | --run   (CPU; resumable; keluaran hasil60/replikasi.csv)"""
import os, sys, time
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[_v] = "1"
os.environ["PYTHONWARNINGS"] = "ignore"
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, "/home/miftah_f/RTX-5090/tuep_fresh/code")
import warnings; warnings.filterwarnings("ignore")
_argv = sys.argv; sys.argv = [_argv[0]]
import numpy as np, pandas as pd
from joblib import Parallel, delayed
import round39_simulasi_kebocoran as S39
import round55_referensi_kontrol as R55
sys.argv = _argv
OUT = os.path.join(HERE, "hasil60"); os.makedirs(OUT, exist_ok=True); CSV = os.path.join(OUT, "replikasi.csv")
REPS = range(1000, 1200)
B = dict(n=200, m=50, p=50, imb=False, scen="H1", sig=1.0, cfrac=0.01, cscale=100.0)
# kode sel -> (jenis, definisi, transformasi lengan penyesuaian, kebenaran yang ditetapkan sebelum run)
SIM = {"N1": (dict(B, dist="lognormal", s=0.0), "raw", "nol"), "N2": (dict(B, dist="contam", s=0.0), "raw", "nol"),
       "M1": (dict(B, dist="contam", s=0.5), "raw", "margin"), "M2": (dict(B, dist="contam", s=1.0, cfrac=0.003), "raw", "margin"),
       "M3": (dict(B, dist="lognormal", s=0.5), "symlog", "margin"), "L1": (dict(B, dist="lognormal", s=0.5), "raw", "bocor"),
       "L2": (dict(B, dist="contam", s=1.0, cscale=1000.0), "raw", "bocor")}
REF = {"R0": (0.0, "nol"), "R1": (0.5, "margin"), "R2": (1.0, "bocor")}


def unit(code, rep):
    if code in SIM:
        cell, t, truth = SIM[code]
        rows = S39.run_rep(dict(cell), rep, ["RF"], [("raw", "none"), (t, "pat")])
        auc = {(r["transform"], r["adjust"]): r["auc"] for r in rows}; base, adj = auc[("raw", "none")], auc[(t, "pat")]
    else:
        s, truth = REF[code]; r = R55.unit(80, "jarang", s, rep); base, adj = r["C1_pooled"], r["CR_pooled"]
    return dict(code=code, truth=truth, rep=rep, auc_base=base, auc_adj=adj, delta=adj - base)


def check():
    t0 = time.time()
    # seed baru tidak tumpang tindih dengan pool tersimpan (round 39: rep < 100; round 55: rep < 100)
    assert min(REPS) >= 1000
    # reproduksi: rep 0 sel L1 harus sama dengan hasil39 tersimpan (lognormal s 0.5, RF raw-pat dan raw-none)
    old = pd.read_csv(os.path.join(HERE, "hasil39", "B_cpu.csv")); o = old[(old.dist == "lognormal") & (old.scen == "H1") & (old.n == 200) & (old.m == 50) & (old.p == 50) & (old.s == 0.5) & (old.sig == 1.0) & (old.rep == 0) & (old.clf == "RF")]
    rows = S39.run_rep(dict(SIM["L1"][0]), 0, ["RF"], [("raw", "none"), ("raw", "pat")]); new = {(r["transform"], r["adjust"]): r["auc"] for r in rows}
    for (t, a), v in new.items():
        ov = o[(o["transform"] == t) & (o.adjust == a)].auc.values; assert len(ov) >= 1 and abs(ov[0] - v) < 1e-9, (t, a, ov, v)
    r55 = pd.read_csv(os.path.join(HERE, "hasil55", "replikasi.csv")); q = r55[(r55.nref == 80) & (r55.design == "jarang") & (r55.s == 1.0) & (r55.rep == 0)].iloc[0]
    u = R55.unit(80, "jarang", 1.0, 0); assert abs(u["CR_pooled"] - q.CR_pooled) < 1e-9 and abs(u["C1_pooled"] - q.C1_pooled) < 1e-9
    print(f"reproduksi rep 0 (hasil39 L1, hasil55 R2) identik [{time.time() - t0:.0f}s]", flush=True)
    t1 = time.time(); unit("L2", 1000); print(f"waktu satu unit sim {time.time() - t1:.0f}s; unit total {len(SIM) * len(REPS) + len(REF) * len(REPS)} (AUC tidak dicetak)", flush=True)


def run():
    t0 = time.time(); done = set(map(tuple, pd.read_csv(CSV, usecols=["code", "rep"]).values.tolist())) if os.path.exists(CSV) else set()
    todo = [(c, r) for c in list(SIM) + list(REF) for r in REPS if (c, r) not in done]; print(f"{len(todo)} unit tersisa", flush=True); nj = int(os.environ.get("NJOBS", 22))
    for i in range(0, len(todo), nj * 4):
        pd.DataFrame(Parallel(n_jobs=nj)(delayed(unit)(c, r) for c, r in todo[i:i + nj * 4])).to_csv(CSV, mode="a", header=not os.path.exists(CSV), index=False)
        print(f"  {min(i + nj * 4, len(todo))}/{len(todo)} [{time.time() - t0:.0f}s]", flush=True)
    print(f"selesai {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    check() if sys.argv[1] == "--check" else run()
