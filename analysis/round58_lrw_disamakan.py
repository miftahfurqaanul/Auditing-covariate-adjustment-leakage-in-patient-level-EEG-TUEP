#!/usr/bin/env python3
"""Putaran 58 (M4): faktorial LR tingkat jendela pada fitur Rohira mentah: pasien latih (80% resep pohon | 100%) x bobot (per jendela | per pasien)
x agregasi skor pasien (rata-rata probabilitas | rata-rata logit), dengan dan tanpa penyesuaian per pasien (rata-rata tak tertimbang semua pasien latih).
Label, pertukaran, lipatan, benih, dan lipatan dalam sama dengan putaran 53. Lihat PRAREGISTRASI_putaran57_60.md.
Pemakaian: python round58_lrw_disamakan.py --check | --run   (CPU; resumable; keluaran hasil58/replikasi.csv)"""
import os, sys, time
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[_v] = "1"
os.environ["PYTHONWARNINGS"] = "ignore"
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, "/home/miftah_f/RTX-5090/tuep_fresh/code")
import warnings; warnings.filterwarnings("ignore")
_argv = sys.argv; sys.argv = [_argv[0]]
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import evalkit as ek
import round42_common as C
import round53_lr_mentah as L53
sys.argv = _argv
OUT = os.path.join(HERE, "hasil58"); os.makedirs(OUT, exist_ok=True); CSV = os.path.join(OUT, "replikasi.csv")
FC = [f"fold_auc_{i}" for i in range(5)]


def units():
    return [("tuep", r, g) for g in (0.0, 1.0) for r in range(120)] + [("emotiv", r, g) for g in (0.0, 1.0) for r in range(30)]


def fit_predict(Z, owner, y, fitp, te, iseed, weight):
    fit = np.isin(owner, fitp); sel = np.isin(owner, te); yf, gf = y[owner[fit]], owner[fit]; sc = StandardScaler().fit(Z[fit])
    sw = None if weight == "win" else 1.0 / np.bincount(gf, minlength=owner.max() + 1)[gf]
    m = LogisticRegressionCV(Cs=ek.CS, cv=L53.inner_splits(gf, yf, iseed), scoring="roc_auc", class_weight="balanced", max_iter=5000).fit(sc.transform(Z[fit]), yf, sample_weight=sw)
    lg = m.decision_function(sc.transform(Z[sel])); s = pd.DataFrame(dict(o=owner[sel], p=1 / (1 + np.exp(-lg)), l=lg)).groupby("o").mean().reindex(te)
    return s.p.values, s.l.values


def unit(cohort, rep, gamma):
    a = L53.api(cohort); N = a["N"]; y = a["labels"](rep, gamma); folds = a["folds"](rep, gamma, y); X, owner = a["windows"](a["perm"](rep))
    keys = [(adj, P, W, A) for adj in ("none", "pat") for P in ("p80", "p100") for W in ("win", "pat") for A in ("prob", "logit")]; S = {k: np.zeros(N) for k in keys}
    for fi, (tr, te) in enumerate(folds):
        seed = a["seed"](rep, gamma, fi); iseed = a["kseed"]("lrw_inner", rep, gamma, fi); tr80, _ = a["split"](tr, y, seed)
        for adj in ("none", "pat"):
            Z = a["adj"](X, owner, tr, adj)
            for P, fitp in (("p80", tr80), ("p100", tr)):
                for W in ("win", "pat"):
                    pr, lg = fit_predict(Z, owner, y, fitp, te, iseed, W); S[(adj, P, W, "prob")][te] = pr; S[(adj, P, W, "logit")][te] = lg
    return [dict(cohort=cohort, rep=rep, gamma=gamma, adjust=adj, patients=P, weight=W, aggregate=A, auc=float(roc_auc_score(y, s)),
                 **{f"fold_auc_{i}": v for i, v in enumerate(C.fold_aucs(y, s, folds))}) for (adj, P, W, A), s in S.items()]


def check():
    t0 = time.time()
    for cohort in ("tuep", "emotiv"):
        new = pd.DataFrame(unit(cohort, 0, 0.0)); new = new[(new["patients"] == "p80") & (new["weight"] == "win") & (new["aggregate"] == "prob")]
        old = pd.read_csv(os.path.join(HERE, "hasil53", f"LRw_{cohort}.csv")); old = old[(old.rep == 0) & (old.gamma == 0.0) & (old["transform"] == "raw") & (old.adjust.isin(["none", "pat"]))]
        m = new.merge(old, on="adjust", suffixes=("", "_o")); assert len(m) == 2
        d = max(np.abs(m[c] - m[c + "_o"]).max() for c in ["auc"] + FC); assert d < 1e-2, (cohort, d)
        print(f"{cohort}: varian resep pohon = LRw putaran 53 (raw none/pat), selisih maks {d:.1e} [{time.time() - t0:.0f}s]", flush=True)
    # bobot per pasien: jumlah bobot sama untuk setiap pasien latih; skor logit = rata-rata logit jendela
    a = L53.api("tuep"); y = a["labels"](2, 1.0); tr, te = a["folds"](2, 1.0, y)[0]; X, owner = a["windows"](a["perm"](2)); gf = owner[np.isin(owner, tr)]
    sw = 1.0 / np.bincount(gf, minlength=owner.max() + 1)[gf]; tot = pd.Series(sw).groupby(gf).sum(); assert np.allclose(tot.values, 1.0)
    y2 = y.copy(); y2[te] = 1 - y2[te]; iseed = a["kseed"]("lrw_inner", 2, 1.0, 0); Z = a["adj"](X, owner, tr, "pat")
    assert np.allclose(fit_predict(Z, owner, y, tr, te, iseed, "pat")[1], fit_predict(Z, owner, y2, tr, te, iseed, "pat")[1]); print("bobot per pasien dan invariansi label uji lulus", flush=True)
    t1 = time.time(); unit("tuep", 1, 1.0); print(f"waktu satu unit TUEP {time.time() - t1:.0f}s (AUC tidak dicetak); unit total {len(units())}", flush=True)


def run():
    t0 = time.time(); done = set(map(tuple, pd.read_csv(CSV, usecols=["cohort", "rep", "gamma"]).drop_duplicates().values.tolist())) if os.path.exists(CSV) else set()
    todo = [u for u in units() if u not in done]; print(f"{len(todo)} unit tersisa", flush=True); nj = int(os.environ.get("NJOBS", 22))
    for i in range(0, len(todo), nj * 2):
        for rr in Parallel(n_jobs=nj)(delayed(unit)(*u) for u in todo[i:i + nj * 2]):
            pd.DataFrame(rr).to_csv(CSV, mode="a", header=not os.path.exists(CSV), index=False)
        print(f"  {min(i + nj * 2, len(todo))}/{len(todo)} [{time.time() - t0:.0f}s]", flush=True)
    print(f"selesai {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    check() if sys.argv[1] == "--check" else run()
