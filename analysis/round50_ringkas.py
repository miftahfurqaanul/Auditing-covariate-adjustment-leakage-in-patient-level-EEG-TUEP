#!/usr/bin/env python3
"""Round 50: summaries of 50A (extension of the TUEP plasmode replicates) and 50B (Emotiv plasmode). See PRAREGISTRASI_putaran50.md.
Usage: CUDA_VISIBLE_DEVICES="" python round50_ringkas.py   (output hasil50/ringkas_A.csv, ringkas_A_frek.csv, ringkas_B.csv, ringkas_B_uji.csv)"""
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""; os.environ["PYTHONWARNINGS"] = "ignore"
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy import stats
import round49_kalibrasi as K

OUT = os.path.join(HERE, "hasil50"); FC = [f"fold_auc_{i}" for i in range(5)]; TREE = ("RF", "XGB")


def prep(D):
    D = D.copy(); D["fold_mean"] = D[FC].astype(float).mean(1); D["pipe"] = D["transform"] + "-" + D["adjust"]; return D


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); adj = np.empty_like(p); run = 0.0
    for k, i in enumerate(o):
        run = max(run, (len(p) - k) * p[i]); adj[i] = min(1.0, run)
    return adj


def tstat(d):
    d = np.asarray(d, float); n = len(d); m = d.mean(); se = d.std(ddof=1) / np.sqrt(n); h = stats.t.ppf(0.975, n - 1) * se
    return m, d.std(ddof=1), se, m - h, m + h, 2 * stats.t.sf(abs(m / se), n - 1)


def bagian_A(rng):
    rows, frows = [], []
    for fam, start in (("RF", 30), ("XGB", 20)):
        old = prep(pd.read_csv(os.path.join(HERE, "hasil42", f"{fam}.csv"))); old = old[(old.cond == "H1") & old.gamma.isin([0.0, 1.0]) & (old.rep < start)]
        new = prep(pd.read_csv(os.path.join(OUT, f"A_{fam}.csv"))); D = pd.concat([old, new], ignore_index=True)
        for g, G in D.groupby("gamma"):
            W = G.pivot_table(index="rep", columns="pipe", values="fold_mean"); assert len(W) == 120 and W.notna().all().all(), (fam, g, W.shape)
            for pipe in ("raw-win", "raw-pat"):
                d = (W[pipe] - W["raw-none"]).values; x = W[pipe].values; m, sd, se, lo, hi, p = tstat(d * 100)
                rows.append(dict(fam=fam, gamma=g, pipe=pipe, n=len(d), auc_adj=x.mean() * 100, auc_none=W["raw-none"].mean() * 100, delta=m, sd=sd, se=se, lo=lo, hi=hi, p=p,
                                 R2_120=K.LAB[K.dec_pool(d, "R2")], R1_120=K.LAB[K.dec_pool(x, "R1")],
                                 delta_awal=(d[:start].mean() * 100), sd_awal=(d[:start].std(ddof=1) * 100)))
                for rule, v in (("R2", d), ("R1", x)):
                    for n in (20, 50, 100):
                        f = K.freq(v, rule, n, rng); ci = K.freq_ci(v, rule, n, rng, outer=500, B=2000)
                        frows.append(dict(fam=fam, gamma=g, pipe=pipe, aturan=rule, n_rep=n, tidak_ada_bukti=f[0] * 100, tidak_pasti=f[1] * 100, kebocoran=f[2] * 100,
                                          kebocoran_lo=ci[0][2] * 100, kebocoran_hi=ci[1][2] * 100, tidak_ada_bukti_lo=ci[0][0] * 100, tidak_ada_bukti_hi=ci[1][0] * 100))
                # expectation under a normal model with the SD of the 120-replicate pool and a true difference of zero (comparison with 49B)
                if g == 0.0:
                    for n in (20, 50, 100):
                        S = rng.normal(0, sd / 100, (20000, n)); dd = K.decide(S.mean(1), S.std(1, ddof=1) / np.sqrt(n), K.THR["R2"]["asli"]); f = np.bincount(dd, minlength=3) / len(dd)
                        frows.append(dict(fam=fam, gamma=g, pipe=pipe, aturan="R2 normal (selisih 0)", n_rep=n, tidak_ada_bukti=f[0] * 100, tidak_pasti=f[1] * 100, kebocoran=f[2] * 100))
    A = pd.DataFrame(rows); F = pd.DataFrame(frows); A.to_csv(f"{OUT}/ringkas_A.csv", index=False); F.to_csv(f"{OUT}/ringkas_A_frek.csv", index=False)
    pd.set_option("display.width", 250); print(A.round(3).to_string()); print(F.round(2).to_string())
    nul = A[A.gamma == 0.0]; print(f"P50A-1 (|mean| <= 1.0 and CI includes 0, all four null cells): {bool(((nul.delta.abs() <= 1.0) & (nul.lo <= 0) & (nul.hi >= 0)).all())}")
    f0 = F[(F.gamma == 0.0) & (F.aturan == "R2")]
    print(f"P50A-2a (R2 leakage <= 5% with 50 replicates, all four null cells): {bool((f0[f0.n_rep == 50].kebocoran <= 5).all())}; P50A-2b (< 31.5% with 20 replicates, >= 3 of 4 cells): {int((f0[f0.n_rep == 20].kebocoran < 31.5).sum())} of 4")
    f1 = F[(F.gamma == 1.0) & (F.aturan == "R2") & (F["pipe"] == "raw-pat") & (F.n_rep == 20)]; print(f"P50A-3 (R2 leakage >= 95% with 20 replicates, raw-pat gamma 1): {f1.kebocoran.round(1).tolist()} -> {bool((f1.kebocoran >= 95).all())}")


def bagian_B():
    D = prep(pd.concat([pd.read_csv(f"{OUT}/B_{f}.csv") for f in ("LR", "RF", "XGB")], ignore_index=True)); rows = []; reps = {}
    for (fam, g), G in D.groupby(["fam", "gamma"]):
        W = G.pivot_table(index="rep", columns="pipe", values="fold_mean")
        for pipe in W.columns:
            t, a = pipe.split("-"); x = W[pipe].values; r = dict(fam=fam, gamma=g, pipe=pipe, n=len(x), auc=x.mean() * 100, sd=x.std(ddof=1) * 100, R1=K.LAB[K.dec_pool(x, "R1")] if a != "none" else "--",
                                                                  auc_cov=G.auc_cov.mean() * 100)
            bp = "raw-none" if fam in TREE else f"{t}-none"
            if a != "none":
                d = (W[pipe] - W[bp]).values; m, sd, se, lo, hi, p = tstat(d * 100); r.update(delta=m, delta_se=se, delta_lo=lo, delta_hi=hi, delta_p=p, R2=K.LAB[K.dec_pool(d, "R2")]); reps[(fam, g, pipe)] = d * 100
            rows.append(r)
    B = pd.DataFrame(rows); B.to_csv(f"{OUT}/ringkas_B.csv", index=False); pd.set_option("display.width", 250); print(B.round(3).to_string())
    u = []
    for fam in TREE:
        m, sd, se, lo, hi, p = tstat(reps[(fam, 1.0, "raw-pat")]); u.append(dict(famili="primer", uji=f"{fam} raw-pat - raw-none (gamma 1)", delta=m, lo=lo, hi=hi, p=p))
    for fam in TREE:
        W = D[(D.fam == fam) & (D.gamma == 1.0)].pivot_table(index="rep", columns="pipe", values="fold_mean"); m, sd, se, lo, hi, p = tstat((W["raw-pat"] - W["rank-pat"]).values * 100)
        u.append(dict(famili="sekunder", uji=f"{fam} raw-pat - rank-pat (gamma 1)", delta=m, lo=lo, hi=hi, p=p))
    U = pd.DataFrame(u)
    for fm in ("primer", "sekunder"):
        k = U.famili == fm; U.loc[k, "holm"] = holm(U.loc[k, "p"].values)
    U.to_csv(f"{OUT}/ringkas_B_uji.csv", index=False); print(U.round(4).to_string())
    adj0 = B[(B.gamma == 0.0) & (B.R1 != "--")]; print(f"control gamma 0: {int((adj0.R1 != 'kebocoran').sum())} of {len(adj0)} adjusted cells not R1 leakage ({(adj0.R1 != 'kebocoran').mean() * 100:.1f}%)")
    b1 = B[(B.gamma == 1.0)]
    print(f"P50B-1: R2 RF/XGB raw-pat = {b1[(b1.fam.isin(TREE)) & (b1["pipe"] == "raw-pat")].R2.tolist()}; Holm primary = {U[U.famili == 'primer'].holm.round(4).tolist()}")
    print(f"P50B-2: reduction raw-pat -> rank-pat = {U[U.famili == 'sekunder'].delta.round(2).tolist()}")
    lr = B[(B.fam == 'LR') & (B.R1 != '--')]; print(f"P50B-3: LR delta = {lr.delta.round(2).tolist()}; R2 = {lr.R2.tolist()}")


if __name__ == "__main__":
    rng = np.random.default_rng(20261102)
    if os.path.exists(f"{OUT}/B_XGB.csv") and "--A" not in sys.argv:
        bagian_B()
    if os.path.exists(f"{OUT}/A_RF.csv") and os.path.exists(f"{OUT}/A_XGB.csv") and "--B" not in sys.argv:
        bagian_A(rng)
