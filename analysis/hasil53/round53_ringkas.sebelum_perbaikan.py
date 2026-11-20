#!/usr/bin/env python3
"""Putaran 53: ringkasan LR pada fitur mentah (E1, E2, sekunder, kontrol, prediksi). Lihat PRAREGISTRASI_putaran53.md.
Pemakaian: python round53_ringkas.py   (CPU; keluaran hasil53/ringkas_E1.csv, ringkas_E2.csv)"""
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""; os.environ["PYTHONWARNINGS"] = "ignore"
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy import stats
import round49_kalibrasi as K

OUT = os.path.join(HERE, "hasil53"); FC = [f"fold_auc_{i}" for i in range(5)]


def prep(D):
    D = D.copy(); D["fold_mean"] = D[FC].astype(float).mean(1); D["pipe"] = D["transform"] + "-" + D["adjust"]
    return D.drop_duplicates(subset=["rep", "gamma", "pipe"], keep="last")


def wide(D, g):
    W = D[D.gamma == g].pivot_table(index="rep", columns="pipe", values="fold_mean"); assert W.notna().all().all(); return W


def tstat(d):
    d = np.asarray(d, float); n = len(d); m = d.mean(); sd = d.std(ddof=1); se = sd / np.sqrt(n); h = stats.t.ppf(0.975, n - 1) * se; t = m / se
    return dict(n=n, mean=m, sd=sd, se=se, lo=m - h, hi=m + h, p2=2 * stats.t.sf(abs(t), n - 1), p1=stats.t.sf(t, n - 1))


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); adj = np.empty_like(p); run = 0.0
    for k, i in enumerate(o):
        run = max(run, (len(p) - k) * p[i]); adj[i] = min(1.0, run)
    return adj


def trees():
    """RF/XGB per kohort dari hasil tersimpan (TUEP: putaran 42 replikasi awal + 50A; Emotiv: 50B)."""
    T = {}
    for fam, start in (("RF", 30), ("XGB", 20)):
        old = prep(pd.read_csv(os.path.join(HERE, "hasil42", f"{fam}.csv"))); old = old[(old.cond == "H1") & (old.rep < start)]
        new = prep(pd.read_csv(os.path.join(HERE, "hasil50", f"A_{fam}.csv"))); new = new[new.cond == "H1"]
        T[("tuep", fam)] = pd.concat([old, new], ignore_index=True)
        T[("emotiv", fam)] = prep(pd.read_csv(os.path.join(HERE, "hasil50", f"B_{fam}.csv")))
    return T


if __name__ == "__main__":
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    L = {(c, f): prep(pd.read_csv(os.path.join(OUT, f"{f}_{c}.csv"))) for c in ("tuep", "emotiv") for f in ("LRw", "LRp")}
    for (c, f), D in L.items():
        exp = {"tuep": {0.0: 120, 1.0: 120, 0.5: 30}, "emotiv": {0.0: 30, 0.5: 30, 1.0: 30}}[c]
        got = D.groupby("gamma").rep.nunique().to_dict(); assert got == exp, (c, f, got); assert D.groupby(["rep", "gamma"]).size().eq(6).all(), (c, f)
    T = trees()
    # ---- E1 dan sekunder deskriptif
    CON = {"LRw": [("raw-win", "raw-none"), ("raw-pat", "raw-none"), ("symlog-pat", "raw-none"), ("rank-pat", "raw-none"), ("log-pat", "raw-none")],
           "LRp": [("raw-pat", "raw-none"), ("log-pat", "log-none"), ("rank-pat", "rank-none")]}
    rows = []
    for (c, f), D in L.items():
        for g in sorted(D.gamma.unique()):
            W = wide(D, g)
            for pipe in sorted(W.columns):
                rows.append(dict(cohort=c, fam=f, gamma=g, pipe=pipe, ref="", n=len(W), auc=W[pipe].mean() * 100, R1=K.LAB[K.dec_pool(W[pipe].values, "R1")]))
            for a, b in CON[f]:
                d = (W[a] - W[b]).values; s = tstat(d * 100)
                rows.append(dict(cohort=c, fam=f, gamma=g, pipe=a, ref=b, n=s["n"], auc=W[a].mean() * 100, R1=K.LAB[K.dec_pool(W[a].values, "R1")], delta=s["mean"], sd=s["sd"], se=s["se"],
                                 lo=s["lo"], hi=s["hi"], p2=s["p2"], R2=K.LAB[K.dec_pool(d, "R2")]))
    E1 = pd.DataFrame(rows); E1.to_csv(f"{OUT}/ringkas_E1.csv", index=False)
    print("== E1 dan sekunder (AUC per lipatan, poin)"); print(E1[E1.ref != ""].round(3).to_string(index=False))
    print(E1[E1.ref == ""][["cohort", "fam", "gamma", "pipe", "n", "auc", "R1"]].round(2).to_string(index=False))
    # ---- E2 (primer: RF) dan sekunder (XGB): interaksi berpasangan per replikasi
    rows = []
    for fam in ("RF", "XGB"):
        for c in ("tuep", "emotiv"):
            for g in (0.0, 1.0):
                Wt = wide(T[(c, fam)], g); Wl = wide(L[(c, "LRw")], g); reps = sorted(set(Wt.index) & set(Wl.index))
                dT = (Wt.loc[reps, "raw-pat"] - Wt.loc[reps, "raw-none"]).values * 100; dL = (Wl.loc[reps, "raw-pat"] - Wl.loc[reps, "raw-none"]).values * 100; s = tstat(dT - dL)
                rows.append(dict(famili="primer" if fam == "RF" else "sekunder", tree=fam, cohort=c, gamma=g, n=s["n"], delta_tree=dT.mean(), delta_LRw=dL.mean(), I=s["mean"], sd=s["sd"],
                                 lo=s["lo"], hi=s["hi"], p1=s["p1"]))
    E2 = pd.DataFrame(rows)
    for fm in ("primer", "sekunder"):
        k = (E2.famili == fm) & (E2.gamma == 1.0); E2.loc[k, "holm"] = holm(E2.loc[k, "p1"].values)
    E2.to_csv(f"{OUT}/ringkas_E2.csv", index=False); print("== E2 (I = delta pohon - delta LRw, raw-pat minus raw-none; uji t satu sisi; Holm pada gamma 1)"); print(E2.round(4).to_string(index=False))
    # ---- prediksi
    e = E1[(E1.fam == "LRw") & (E1.pipe == "raw-pat") & (E1.ref == "raw-none")].set_index(["cohort", "gamma"])
    p = E1[(E1.fam == "LRp") & (E1.pipe == "raw-pat") & (E1.ref == "raw-none")].set_index(["cohort", "gamma"])
    pr = E2[(E2.famili == "primer") & (E2.gamma == 1.0)].set_index("cohort")
    print("== prediksi")
    print(f"P53-1 LRw TUEP gamma 1 bukan R2 kebocoran: {e.loc[('tuep', 1.0), 'R2'] != 'kebocoran'} ({e.loc[('tuep', 1.0), 'delta']:.2f} poin; {e.loc[('tuep', 1.0), 'R2']})")
    print(f"P53-2 LRw Emotiv gamma 1 bukan R2 kebocoran: {e.loc[('emotiv', 1.0), 'R2'] != 'kebocoran'} ({e.loc[('emotiv', 1.0), 'delta']:.2f} poin; {e.loc[('emotiv', 1.0), 'R2']})")
    print(f"P53-3 E2 Holm < 0.05 di kedua kohort: {bool((pr.holm < 0.05).all())} (Holm {pr.holm.round(5).to_dict()})")
    q = p.loc[("tuep", 1.0)]; print(f"P53-4 LRp TUEP raw-pat |delta| <= 1.0 atau R2 bukan kebocoran: {bool(abs(q.delta) <= 1.0 or q.R2 != 'kebocoran')} ({q.delta:.2f} poin; {q.R2})")
    ok5 = all(x.loc[(c, 0.0), "R2"] != "kebocoran" for x in (e, p) for c in ("tuep", "emotiv"))
    print(f"P53-5 kontrol gamma 0 (LRw dan LRp raw-pat bukan R2 kebocoran di kedua kohort): {ok5} (" + "; ".join(f"{n_} {c} {x.loc[(c, 0.0), 'delta']:.2f} {x.loc[(c, 0.0), 'R2']}" for n_, x in (("LRw", e), ("LRp", p)) for c in ("tuep", "emotiv")) + ")")
