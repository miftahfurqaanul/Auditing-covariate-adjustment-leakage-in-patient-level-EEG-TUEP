#!/usr/bin/env python3
"""Putaran 54: LR tingkat jendela dengan sampel dan bobot pengklasifikasi diselaraskan dengan pemasangan penyesuaian per pasien.
Lengan A0 (= putaran 53), W (bobot 1/n_p), S (semua pasien latih), WS (keduanya); raw-none dan raw-pat. Lihat PRAREGISTRASI_putaran54.md.
Pemakaian: python round54_lr_bobot.py --check | --run <tuep|emotiv>   (CPU saja; resumable; keluaran hasil54/LRw4_<kohort>.csv)"""
import os, sys, time
os.environ["CUDA_VISIBLE_DEVICES"] = ""; os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ["PYTHONWARNINGS"] = "ignore"
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
import round53_lr_mentah as R53
sys.argv = _argv

OUT = os.path.join(HERE, "hasil54"); os.makedirs(OUT, exist_ok=True)
ARMS = [("A0", False, False), ("W", True, False), ("S", False, True), ("WS", True, True)]   # (nama, bobot 1/n_p, semua pasien latih)
PIPES = [("raw", "none"), ("raw", "pat")]


def units(cohort):
    return [(r, g) for g in (0.0, 1.0) for r in range(120 if cohort == "tuep" else 30)]


def fit_patients(tr, y, seed, split, allpat):
    return np.asarray(tr) if allpat else split(tr, y, seed)[0]


def window_weights(owner_fit):
    cnt = np.bincount(owner_fit); w = 1.0 / cnt[owner_fit]; return w / w.mean()


def lrw_arm(Z, owner, y, tr, te, seed, inner_seed, split, weighted, allpat):
    tr_p = fit_patients(tr, y, seed, split, allpat); fit = np.isin(owner, tr_p); sel = np.isin(owner, te)
    yf, gf = y[owner[fit]], owner[fit]; sc = StandardScaler().fit(Z[fit])
    m = LogisticRegressionCV(Cs=ek.CS, cv=R53.inner_splits(gf, yf, inner_seed), scoring="roc_auc", class_weight="balanced", max_iter=5000)
    if weighted:
        m.fit(sc.transform(Z[fit]), yf, sample_weight=window_weights(gf))
    else:
        m.fit(sc.transform(Z[fit]), yf)
    return pd.Series(m.predict_proba(sc.transform(Z[sel]))[:, 1]).groupby(owner[sel]).mean().reindex(te).values


def unit(cohort, rep, gamma):
    a = R53.api(cohort); N = a["N"]; y = a["labels"](rep, gamma); perm = a["perm"](rep); folds = a["folds"](rep, gamma, y); X, owner = a["windows"](perm)
    P = {(arm, t, ad): np.zeros(N) for arm, _, _ in ARMS for t, ad in PIPES}
    for fi, (tr, te) in enumerate(folds):
        seed = a["seed"](rep, gamma, fi); iseed = a["kseed"]("lrw_inner", rep, gamma, fi); trw = np.isin(owner, tr)
        Zraw = a["tf"]("raw", X[trw])(X)
        for t, ad in PIPES:
            Z = a["adj"](Zraw, owner, tr, ad)
            for arm, wt, ap in ARMS:
                P[(arm, t, ad)][te] = lrw_arm(Z, owner, y, tr, te, seed, iseed, a["split"], wt, ap)
    return [dict(cohort=cohort, rep=rep, cond="H1", gamma=gamma, arm=arm, transform=t, adjust=ad, auc=float(roc_auc_score(y, p)),
                 **{f"fold_auc_{i}": v for i, v in enumerate(C.fold_aucs(y, p, folds))}) for (arm, t, ad), p in P.items()]


def checks():
    n = 0; t0 = time.time()
    for cohort in ("tuep", "emotiv"):
        a = R53.api(cohort); N = a["N"]
        # 1. A0 mereproduksi LRw putaran 53 (replikasi 0, gamma 0)
        new = pd.DataFrame(unit(cohort, 0, 0.0)); new = new[new.arm == "A0"]; FC = ["auc"] + [f"fold_auc_{i}" for i in range(5)]
        cur = pd.DataFrame(R53.unit(cohort, 0, 0.0)["LRw"]); cur = cur[(cur["transform"] == "raw") & cur.adjust.isin(["none", "pat"])]     # adendum 1 (a)
        m = new.merge(cur, on=["transform", "adjust"], suffixes=("", "_o")); assert len(m) == 2
        d0 = max(np.abs(m[c] - m[c + "_o"]).max() for c in FC); assert d0 < 1e-12, d0; n += 1
        old = pd.read_csv(os.path.join(HERE, "hasil53", f"LRw_{cohort}.csv")); old = old[(old.rep == 0) & (old.gamma == 0.0) & (old["transform"] == "raw") & old.adjust.isin(["none", "pat"])]
        m = new.merge(old, on=["transform", "adjust"], suffixes=("", "_o")); assert len(m) == 2
        d = max(np.abs(m[c] - m[c + "_o"]).max() for c in FC); assert d < 0.01, d; n += 1                                          # adendum 1 (b)
        print(f"{cohort}: A0 = putaran 53 dijalankan ulang (selisih {d0:.1e}); terhadap CSV putaran 53 selisih maks {d:.4f}", flush=True)
        # 2. kovarians silang residual-desain atas pasien latih pengklasifikasi
        D = C.D if cohort == "tuep" else R53.E.D
        y = a["labels"](3, 1.0); folds = a["folds"](3, 1.0, y); tr, te = folds[1]; s = a["seed"](3, 1.0, 1)
        X, owner = a["windows"](a["perm"](3)); trw = np.isin(owner, tr); Zr = a["tf"]("raw", X[trw])(X); Z = a["adj"](Zr, owner, tr, "pat")
        pm = lambda A: np.vstack([A[owner == p].mean(0) for p in range(N)])
        Zb, Xb = pm(Z), pm(Zr); cnt = np.bincount(owner, minlength=N).astype(float)
        for arm, wt, ap in ARMS:
            tp = fit_patients(tr, y, s, a["split"], ap); v = np.ones(len(tp)) if wt else cnt[tp]
            Dc = D[tp] - np.average(D[tp], axis=0, weights=v)                  # kovarians (bukan momen mentah): pusatkan desain dengan bobot yang sama
            num = np.linalg.norm((Zb[tp] * v[:, None]).T @ Dc[:, 1:]); den = np.linalg.norm((Xb[tp] * v[:, None]).T @ Dc[:, 1:])
            rel = num / den; print(f"{cohort} {arm}: norma relatif kovarians silang residual-desain {rel:.2e}", flush=True)
            if arm == "WS":
                assert rel < 1e-8, rel; n += 1
            else:
                assert rel > 1e-6, rel; n += 1
        # 3. invariansi terhadap label uji
        y2 = y.copy(); y2[te] = 1 - y2[te]; iseed = a["kseed"]("lrw_inner", 3, 1.0, 1)
        for arm, wt, ap in ARMS:
            assert np.allclose(lrw_arm(Z, owner, y, tr, te, s, iseed, a["split"], wt, ap), lrw_arm(Z, owner, y2, tr, te, s, iseed, a["split"], wt, ap)); n += 1
        # 4. lipatan dalam berkelompok; 5. bobot per pasien sama
        for ap in (False, True):
            tp = fit_patients(tr, y, s, a["split"], ap); fit = np.isin(owner, tp)
            for a_, b_ in R53.inner_splits(owner[fit], y[owner[fit]], iseed):
                assert not set(owner[fit][a_]) & set(owner[fit][b_]); n += 1
            w = window_weights(owner[fit]); tot = pd.Series(w).groupby(owner[fit]).sum().values; assert np.allclose(tot, tot[0]); n += 1
        t1 = time.time(); r = unit(cohort, 1, 1.0); assert len(r) == 8
        print(f"waktu satu unit {cohort}: {time.time() - t1:.0f}s (AUC tidak dicetak)", flush=True)
    print(f"{n} pemeriksaan lulus [{time.time() - t0:.0f}s]", flush=True)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "--check"; t0 = time.time()
    if mode == "--check":
        checks()
    elif mode == "--run":
        cohort = sys.argv[2]; pw = f"{OUT}/LRw4_{cohort}.csv"
        done = set(map(tuple, pd.read_csv(pw, usecols=["rep", "gamma"]).drop_duplicates().values.tolist())) if os.path.exists(pw) else set()
        todo = [u for u in units(cohort) if u not in done]; print(f"{cohort}: {len(todo)} unit tersisa dari {len(units(cohort))}", flush=True)
        nj = int(os.environ.get("NJOBS", 8))
        for i in range(0, len(todo), nj):
            chunk = todo[i:i + nj]; res = Parallel(n_jobs=nj)(delayed(unit)(cohort, r, g) for r, g in chunk)
            for rr in res:
                pd.DataFrame(rr).to_csv(pw, mode="a", header=not os.path.exists(pw), index=False)
            print(f"{min(i + nj, len(todo))}/{len(todo)} [{time.time() - t0:.0f}s]", flush=True)
    print(f"selesai {time.time() - t0:.0f}s", flush=True)
