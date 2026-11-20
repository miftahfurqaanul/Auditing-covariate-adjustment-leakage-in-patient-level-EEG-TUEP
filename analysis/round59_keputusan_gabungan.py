#!/usr/bin/env python3
"""Putaran 59 (M10): karakteristik operasi keputusan GABUNGAN (kebocoran bila R1 atau R2 'kebocoran'; tidak ada bukti bila keduanya 'tidak ada bukti')
dari pool replikasi tersimpan (tanpa melatih model). Lihat PRAREGISTRASI_putaran57_60.md. Keluaran: hasil59/frekuensi_gabungan.csv dan ringkas.log (stdout).
Pemakaian: python round59_keputusan_gabungan.py"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, "/home/miftah_f/RTX-5090/tuep_fresh/code")
import numpy as np, pandas as pd
from round49_kalibrasi import decide, THR
import round42_ringkas as R42, round40_ringkas as R40
OUT = os.path.join(HERE, "hasil59"); os.makedirs(OUT, exist_ok=True); H = lambda *p: os.path.join(HERE, *p); RNG = np.random.default_rng(59)
FC = [f"fold_auc_{i}" for i in range(5)]


def freqs(x, d, n, B):
    """Bootstrap indeks replikasi berpasangan; kembalikan frekuensi (tidak ada bukti, tidak pasti, kebocoran) untuk R1, R2, dan gabungan."""
    idx = RNG.integers(0, len(x), (B, n)); X, Dd = x[idx], d[idx]
    r1 = decide(X.mean(1), X.std(1, ddof=1) / np.sqrt(n), THR["R1"]["asli"]); r2 = decide(Dd.mean(1), Dd.std(1, ddof=1) / np.sqrt(n), THR["R2"]["asli"])
    j = np.where((r1 == 2) | (r2 == 2), 2, np.where((r1 == 0) & (r2 == 0), 0, 1))
    f = lambda a: np.bincount(a, minlength=3) / B
    return f(r1), f(r2), f(j)


def pools():
    P = []
    a29 = pd.read_csv(H("hasil29", "A_kebocoran_replikasi.csv")); a29 = a29[a29.scenario == "H1_all"]
    P.append(dict(sumber="EEG", kumpulan="r29 V0 vs C1 (gabungan)", jenis="positif", x=a29.V0.values, d=(a29.V0 - a29.C1).values, B=10000))
    P.append(dict(sumber="EEG", kumpulan="r29 V4 vs C1 (gabungan)", jenis="diuji", x=a29.V4.values, d=(a29.V4 - a29.C1).values, B=10000))
    r56 = pd.read_csv(H("hasil56", "replikasi.csv"))
    P.append(dict(sumber="EEG", kumpulan="r56 VR vs C1 (gabungan)", jenis="diuji", x=r56.VR.values, d=(r56.VR - r56.C1).values, B=10000))
    n40 = pd.read_csv(H("hasil40", "nm_perlipatan_replikasi.csv")); n40["fm"] = n40[FC].mean(1); n40 = n40[n40.scenario == "H1"].pivot_table(index="rep", columns="variant", values="fm").dropna()
    P.append(dict(sumber="EEG", kumpulan="r40 NM V4 vs N0 (per lipatan)", jenis="diuji", x=n40.V4.values, d=(n40.V4 - n40.N0).values, B=10000))
    # plasmode putaran 42 + perluasan 50A (RF, XGB), per lipatan
    D = R42.load(); D = D[D.cond == "H1"]
    for fam in ("RF", "XGB"):
        e = pd.read_csv(H("hasil50", f"A_{fam}.csv")); e["fold_mean"] = e[FC].mean(1); e["pipe"] = e["transform"] + "-" + e["adjust"]; D = pd.concat([D, e[D.columns.intersection(e.columns)]], ignore_index=True)
    for (fam, g), G in D.groupby(["fam", "gamma"]):
        W = G.pivot_table(index="rep", columns="pipe", values="fold_mean")
        for pipe in W.columns:
            t, a = pipe.split("-"); bp = R42.baseline_pipe(fam, t)
            if not R42.is_adj(fam, a) or bp not in W.columns: continue
            Q = W[[pipe, bp]].dropna()
            if len(Q) < 20: continue
            kind = "nol" if g == 0 else ("positif" if pipe in ("raw-win", "raw-pat") and fam in ("RF", "XGB", "Tang") and g == 1 else "diuji")
            P.append(dict(sumber="plasmode", kumpulan=f"{fam} {pipe} gamma={g:g}", jenis=kind, x=Q[pipe].values, d=(Q[pipe] - Q[bp]).values, B=4000))
    # simulasi putaran 40 (per lipatan), H1 saja
    S = R40.paired(R40.load(), "fold_mean"); S = S[S.scen == "H1"]
    for k, g in S.groupby(R40.CELLK + ["clf", "transform", "adjust"]):
        if len(g) < 20: continue
        r = dict(zip(R40.CELLK + ["clf", "transform", "adjust"], k)); x, d = g["fold_mean"].values, g["delta"].values
        kind = "nol" if r["s"] == 0 else ("positif" if (x.mean() > 0.55 or d.mean() > 0.03) else "diuji")
        P.append(dict(sumber="simulasi", kumpulan="|".join(str(r[c]) for c in ("dist", "m", "imb", "s", "sig", "cfrac", "cscale", "clf", "transform", "adjust")), jenis=kind, x=x, d=d, B=2000))
    return P


if __name__ == "__main__":
    rows = []
    for p in pools():
        for n in (20, 50, 100):
            if n > len(p["x"]): continue
            f1, f2, fj = freqs(p["x"], p["d"], n, p["B"])
            rows.append(dict(sumber=p["sumber"], kumpulan=p["kumpulan"], jenis=p["jenis"], ukuran=len(p["x"]), mean_x=float(np.mean(p["x"])), mean_d=float(np.mean(p["d"])), sd_d=float(np.std(p["d"], ddof=1)), n=n,
                             R1_tb=f1[0], R1_bocor=f1[2], R2_tb=f2[0], R2_bocor=f2[2], gab_tb=fj[0], gab_tp=fj[1], gab_bocor=fj[2]))
    F = pd.DataFrame(rows); F.to_csv(os.path.join(OUT, "frekuensi_gabungan.csv"), index=False)
    pd.set_option("display.width", 250)
    for (src, kind), G in F.groupby(["sumber", "jenis"]):
        print(f"\n== {src} / {kind}: {G.kumpulan.nunique()} kumpulan")
        print(G.groupby("n")[["R1_bocor", "R2_bocor", "gab_bocor", "R1_tb", "R2_tb", "gab_tb"]].agg(["mean", "max"]).round(3).to_string())
    print("\n== EEG per kumpulan"); print(F[F.sumber == "EEG"][["kumpulan", "n", "mean_x", "mean_d", "R1_bocor", "R2_bocor", "gab_bocor", "gab_tb"]].round(3).to_string(index=False))
    print("\n== plasmode gamma 0, pohon mentah"); print(F[(F.sumber == "plasmode") & F.kumpulan.str.contains("raw-pat gamma=0")][["kumpulan", "n", "mean_d", "R1_bocor", "R2_bocor", "gab_bocor", "gab_tb"]].round(3).to_string(index=False))
