#!/usr/bin/env python3
"""Putaran 53: regresi logistik pada fitur jendela Rohira MENTAH yang identik dengan RF/XGB plasmode (TUEP putaran 42+50A, Emotiv putaran 50B).
LRw = LR tingkat jendela (pasien latih 80% sama dengan RF; C dari CV dalam berkelompok per pasien; skor pasien = rata-rata probabilitas jendela).
LRp = LR tingkat pasien pada median fitur jendela mentah per pasien (evalkit.nested_lr pada semua pasien latih, seperti LR putaran 42).
Lihat PRAREGISTRASI_putaran53.md. Kode putaran 42/50 diimpor tanpa diubah.
Pemakaian: python round53_lr_mentah.py --check | --run <tuep|emotiv>   (CPU saja; resumable; keluaran hasil53/LRw_<kohort>.csv dan LRp_<kohort>.csv)"""
import os, sys, time
os.environ["CUDA_VISIBLE_DEVICES"] = ""; os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ["PYTHONWARNINGS"] = "ignore"
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, "/home/miftah_f/RTX-5090/tuep_fresh/code")
import warnings; warnings.filterwarnings("ignore")
_argv = sys.argv; sys.argv = [_argv[0]]                                   # modul lama membaca --run saat diimpor; di sini tidak ada yang memakai GPU
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
import evalkit as ek
import round42_common as C
import round42_cpu as P42
import round50_emotiv as E
sys.argv = _argv

OUT = os.path.join(HERE, "hasil53"); os.makedirs(OUT, exist_ok=True)
PIPES_W = [("raw", "none"), ("raw", "win"), ("raw", "pat"), ("symlog", "pat"), ("rank", "pat"), ("log", "pat")]      # = PIPES_RF putaran 42 / PIPES_W putaran 50B
PIPES_P = [(t, a) for t in ("raw", "log", "rank") for a in ("none", "pat")]
assert PIPES_W == P42.PIPES_RF == E.PIPES_W


def units(cohort):
    if cohort == "tuep":
        return [(r, g) for g in (0.0, 1.0) for r in range(120)] + [(r, 0.5) for r in range(30)]
    return [(r, g) for g in (0.0, 0.5, 1.0) for r in range(30)]


def api(cohort):
    """Fungsi dan objek putaran lama per kohort (tanpa perubahan)."""
    if cohort == "tuep":
        return dict(N=C.N, labels=C.gen_labels, perm=lambda r: C.gen_perm(r, "H1"), folds=C.folds_for, windows=P42.windows_for, split=C.split_val,
                    seed=lambda r, g, fi: C.kseed("plasmode_clf", r, g, fi), tf=C.make_transform, adj=C.adjust_units, eta=C.ETA, kseed=C.kseed)
    return dict(N=E.N, labels=E.gen_labels, perm=E.gen_perm, folds=E.folds_for, windows=E.windows_for, split=E.split_val,
                seed=lambda r, g, fi: E.kseed("emotiv_clf", r, g, fi), tf=E.make_transform, adj=E.adjust_units, eta=E.ETA, kseed=E.kseed)


def inner_splits(owner_fit, y_fit, seed):
    return list(StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(np.zeros(len(y_fit)), y_fit, owner_fit))


def lrw_clf(Z, owner, y, tr, te, seed, inner_seed, split):
    """LR tingkat jendela pada pasien latih yang SAMA dengan RF (split_val dengan benih RF); validasi 20% tidak dipakai, seperti RF."""
    tr_p, _ = split(tr, y, seed); fit = np.isin(owner, tr_p); sel = np.isin(owner, te)
    yf, gf = y[owner[fit]], owner[fit]; sc = StandardScaler().fit(Z[fit])
    m = LogisticRegressionCV(Cs=ek.CS, cv=inner_splits(gf, yf, inner_seed), scoring="roc_auc", class_weight="balanced", max_iter=5000).fit(sc.transform(Z[fit]), yf)
    return pd.Series(m.predict_proba(sc.transform(Z[sel]))[:, 1]).groupby(owner[sel]).mean().reindex(te).values


def patient_median(X, owner, N):
    return np.vstack([np.median(X[owner == p], axis=0) for p in range(N)])


def unit(cohort, rep, gamma):
    a = api(cohort); N = a["N"]; y = a["labels"](rep, gamma); perm = a["perm"](rep); folds = a["folds"](rep, gamma, y); X, owner = a["windows"](perm)
    PM = patient_median(X, owner, N); auc_cov = float(roc_auc_score(y, a["eta"])); Pw = {p: np.zeros(N) for p in PIPES_W}; Pp = {p: np.zeros(N) for p in PIPES_P}
    for fi, (tr, te) in enumerate(folds):
        seed = a["seed"](rep, gamma, fi); iseed = a["kseed"]("lrw_inner", rep, gamma, fi); trw = np.isin(owner, tr); cache = {}
        for t, ad in PIPES_W:
            if t not in cache:
                cache[t] = a["tf"](t, X[trw])(X)
            Pw[(t, ad)][te] = lrw_clf(a["adj"](cache[t], owner, tr, ad), owner, y, tr, te, seed, iseed, a["split"])
        for t, ad in PIPES_P:
            Zp = a["adj"](a["tf"](t, PM[tr])(PM), np.arange(N), tr, ad); Pp[(t, ad)][te] = ek.nested_lr().fit(Zp[tr], y[tr]).predict_proba(Zp[te])[:, 1]
    out = {}
    for fam, P in (("LRw", Pw), ("LRp", Pp)):
        out[fam] = [dict(fam=fam, cohort=cohort, rep=rep, cond="H1", gamma=gamma, transform=t, adjust=ad, auc=float(roc_auc_score(y, p)),
                         **{f"fold_auc_{i}": v for i, v in enumerate(C.fold_aucs(y, p, folds))}, auc_cov=auc_cov) for (t, ad), p in P.items()]
    return out


def checks():
    n = 0; t0 = time.time()
    # 1. reproduksi RF lama dengan kode yang diimpor
    new = pd.DataFrame(P42.unit_window("RF", 0, "H1", 0.0)); old = pd.read_csv(os.path.join(HERE, "hasil42", "RF.csv")); old = old[(old.rep == 0) & (old.cond == "H1") & (old.gamma == 0.0)]
    m = new.merge(old, on=["transform", "adjust"], suffixes=("", "_o")); FC = ["auc"] + [f"fold_auc_{i}" for i in range(5)]; assert len(m) == 6
    d1 = max(np.abs(m[c] - m[c + "_o"]).max() for c in FC); assert d1 < 1e-9, d1; n += 1
    new = pd.DataFrame(E.unit_window("RF", 0, 0.0)); old = pd.read_csv(os.path.join(HERE, "hasil50", "B_RF.csv")); old = old[(old.rep == 0) & (old.gamma == 0.0)]
    m = new.merge(old, on=["transform", "adjust"], suffixes=("", "_o")); assert len(m) == 6
    d2 = max(np.abs(m[c] - m[c + "_o"]).max() for c in FC); assert d2 < 1e-9, d2; n += 1
    for cohort in ("tuep", "emotiv"):
        a = api(cohort); N = a["N"]
        # 2. label/pertukaran/lipatan identik dengan fungsi lama; pasien latih LRw = pasien latih RF
        y = a["labels"](7, 1.0); folds = a["folds"](7, 1.0, y); tr, te = folds[2]; s = a["seed"](7, 1.0, 2)
        tr_p, va = a["split"](tr, y, s); tr_p2, _ = a["split"](tr, y, s); assert np.array_equal(tr_p, tr_p2) and not set(tr_p) & set(te) and not set(va) & set(te); n += 1
        # 3. invariansi terhadap label uji (LRw dan LRp)
        X, owner = a["windows"](a["perm"](7)); trw = np.isin(owner, tr); Z = a["adj"](a["tf"]("raw", X[trw])(X), owner, tr, "pat")
        y2 = y.copy(); y2[te] = 1 - y2[te]; iseed = a["kseed"]("lrw_inner", 7, 1.0, 2)
        assert np.allclose(lrw_clf(Z, owner, y, tr, te, s, iseed, a["split"]), lrw_clf(Z, owner, y2, tr, te, s, iseed, a["split"])); n += 1
        PM = patient_median(X, owner, N); Zp = a["adj"](a["tf"]("log", PM[tr])(PM), np.arange(N), tr, "pat")
        assert np.allclose(ek.nested_lr().fit(Zp[tr], y[tr]).predict_proba(Zp[te])[:, 1], ek.nested_lr().fit(Zp[tr], y2[tr]).predict_proba(Zp[te])[:, 1]); n += 1
        # 4. lipatan dalam berkelompok per pasien
        fit = np.isin(owner, tr_p); sp = inner_splits(owner[fit], y[owner[fit]], iseed); assert len(sp) == 5
        for a_, b_ in sp:
            assert not set(owner[fit][a_]) & set(owner[fit][b_]); n += 1
        # transformasi dan penyesuaian tidak memakai data uji (offset latih tetap bila data uji diubah)
        Xp = X.copy(); Xp[~trw] = Xp[~trw] * 5 + 3
        for t in ("symlog", "log", "rank"):
            assert np.allclose(a["tf"](t, X[trw])(X[trw]), a["tf"](t, Xp[trw])(Xp[trw])); n += 1
        Zr = a["tf"]("rank", X[trw])(X); Zr2 = Zr.copy(); Zr2[~trw] = Zr2[~trw] * 5 + 3
        for ad in ("win", "pat"):
            assert np.allclose(a["adj"](Zr, owner, tr, ad)[trw], a["adj"](Zr2, owner, tr, ad)[trw]); n += 1
    print(f"{n} pemeriksaan lulus; reproduksi RF: TUEP {d1:.1e}, Emotiv {d2:.1e} [{time.time() - t0:.0f}s]", flush=True)
    for cohort in ("tuep", "emotiv"):
        t1 = time.time(); r = unit(cohort, 0, 0.0); assert len(r["LRw"]) == 6 and len(r["LRp"]) == 6
        print(f"waktu satu unit {cohort}: {time.time() - t1:.0f}s (AUC tidak dicetak)", flush=True)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "--check"; t0 = time.time()
    if mode == "--check":
        checks()
    elif mode == "--run":
        cohort = sys.argv[2]; pw, pp = f"{OUT}/LRw_{cohort}.csv", f"{OUT}/LRp_{cohort}.csv"
        done = set(map(tuple, pd.read_csv(pw, usecols=["rep", "gamma"]).drop_duplicates().values.tolist())) if os.path.exists(pw) else set()
        if os.path.exists(pp):
            done &= set(map(tuple, pd.read_csv(pp, usecols=["rep", "gamma"]).drop_duplicates().values.tolist()))
        todo = [u for u in units(cohort) if u not in done]; print(f"{cohort}: {len(todo)} unit tersisa dari {len(units(cohort))}", flush=True)
        nj = int(os.environ.get("NJOBS", 12))
        for i in range(0, len(todo), nj):
            chunk = todo[i:i + nj]; res = Parallel(n_jobs=nj)(delayed(unit)(cohort, r, g) for r, g in chunk)
            for rr in res:
                pd.DataFrame(rr["LRw"]).to_csv(pw, mode="a", header=not os.path.exists(pw), index=False)
                pd.DataFrame(rr["LRp"]).to_csv(pp, mode="a", header=not os.path.exists(pp), index=False)
            print(f"{min(i + nj, len(todo))}/{len(todo)} [{time.time() - t0:.0f}s]", flush=True)
    print(f"selesai {time.time() - t0:.0f}s", flush=True)
