#!/usr/bin/env python3
"""Putaran 51 (E1, E2): plasmode TUEP dan Emotiv dengan penyesuaian EB, kondisi E (efek kovariat sebenarnya diketahui), dan indeks keterpulihan kovariat.
Kode putaran 42/50 TIDAK diubah: label, permutasi, lipatan, pembagian validasi, dan pengklasifikasi diimpor apa adanya. Lihat PRAREGISTRASI_putaran51.md.
Pemakaian: python round51_plasmode.py --check <tuep|emotiv> <RF|XGB>
           python round51_plasmode.py --run <tuep|emotiv> <RF|XGB>      (resumable; hasil51/<kohort>_<fam>.csv)
           python round51_plasmode.py --rho <tuep|emotiv>               (CPU; hasil51/rho_<kohort>.csv)"""
import os, sys, time
MODE, COH = sys.argv[1], sys.argv[2]; FAM = sys.argv[3] if len(sys.argv) > 3 else "RF"
sys.argv = [sys.argv[0], "--run", FAM]                                     # modul lama membaca --run saat diimpor (GPU hanya untuk XGB)
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ["PYTHONWARNINGS"] = "ignore"
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import round51_common as K

OUT = os.path.join(HERE, "hasil51"); os.makedirs(OUT, exist_ok=True)
PIPES = [("raw", "none"), ("raw", "pat"), ("log", "pat"), ("rank", "pat"), ("raw", "ash"), ("log", "ash"), ("rank", "ash"), ("log", "eb"), ("rank", "eb")]
FC = [f"fold_auc_{i}" for i in range(5)]

if COH == "tuep":
    import round42_cpu as P42
    C = P42.C; N, D, ETA = C.N, C.D, C.ETA; XR = P42.XR
    _, OWN0 = P42.windows_for(np.arange(N))
    gen_labels, folds_for, make_transform, windows_for = C.gen_labels, C.folds_for, C.make_transform, P42.windows_for
    gen_perm = lambda rep, cond: C.gen_perm(rep, "H2a" if cond == "H2a" else "H1")
    clf = lambda fam, Z, owner, y, tr, te, rep, gamma, fi: P42.window_clf(fam, Z, owner, y, tr, te, C.kseed("plasmode_clf", rep, gamma, fi))
    kseed = C.kseed; OLD = {"RF": os.path.join(HERE, "hasil42", "RF.csv")}
    UNITS = [(r, c, g) for c, reps in (("H1", 120), ("E", 120), ("H2a", 30)) for r in range(reps) for g in (0.0, 1.0)]; RHO_CONDS = ("H1", "E", "H2a")
elif COH == "emotiv":
    import round50_emotiv as EM
    N, D, ETA, XR, OWN0 = EM.N, EM.D, EM.ETA, EM.XR, EM.OWN
    gen_labels, folds_for, make_transform, windows_for = EM.gen_labels, EM.folds_for, EM.make_transform, EM.windows_for
    gen_perm = lambda rep, cond: EM.gen_perm(rep)
    clf = lambda fam, Z, owner, y, tr, te, rep, gamma, fi: EM.window_clf(fam, Z, owner, y, tr, te, EM.kseed("emotiv_clf", rep, gamma, fi))
    kseed = EM.kseed; OLD = {"RF": os.path.join(HERE, "hasil50", "B_RF.csv")}
    UNITS = [(r, c, g) for c in ("H1", "E") for r in range(120) for g in (0.0, 1.0)]; RHO_CONDS = ("H1", "E")
else:
    raise ValueError(COH)
BT = K.true_effect(XR, OWN0, D)                                            # efek kovariat sebenarnya untuk kondisi E (tanpa label)


def data_for(rep, cond):
    X, owner = windows_for(gen_perm(rep, cond))
    if cond == "E":
        X = K.apply_effect(X, owner, D, BT)
    return X, owner


def unit(fam, rep, cond, gamma, pipes=PIPES):
    y = gen_labels(rep, gamma); folds = folds_for(rep, gamma, y); X, owner = data_for(rep, cond); P = {p: np.zeros(N) for p in pipes}
    for fi, (tr, te) in enumerate(folds):
        trw = np.isin(owner, tr); cache = {}
        for t, a in pipes:
            if t not in cache:
                cache[t] = make_transform(t, X[trw])(X)
            P[(t, a)][te] = clf(fam, K.adjust_units(cache[t], owner, tr, a, D), owner, y, tr, te, rep, gamma, fi)
    return [dict(cohort=COH, fam=fam, rep=rep, cond=cond, gamma=gamma, transform=t, adjust=a, auc=float(roc_auc_score(y, p)),
                 **{f"fold_auc_{i}": v for i, v in enumerate([float(roc_auc_score(y[te], p[te])) for _, te in folds])}, auc_cov=float(roc_auc_score(y, ETA))) for (t, a), p in P.items()]


def rho_unit(rep, cond):
    """Indeks keterpulihan kovariat (tanpa label): rho Spearman antara prediksi out-of-fold RF-regresor (jendela -> ETA, rata-rata per pasien) dan ETA."""
    X, owner = data_for(rep, cond); folds = list(KFold(5, shuffle=True, random_state=kseed("r51_rho", COH, rep)).split(np.zeros(N))); P = {p: np.zeros(N) for p in PIPES}
    for fi, (tr, te) in enumerate(folds):
        trw = np.isin(owner, tr); tew = np.isin(owner, te); cache = {}
        for t, a in PIPES:
            if t not in cache:
                cache[t] = make_transform(t, X[trw])(X)
            Z = K.adjust_units(cache[t], owner, tr, a, D)
            m = RandomForestRegressor(n_estimators=100, min_samples_leaf=5, n_jobs=1, random_state=kseed("r51_rhorf", COH, rep, fi)).fit(Z[trw], ETA[owner[trw]])
            P[(t, a)][te] = pd.Series(m.predict(Z[tew])).groupby(owner[tew]).mean().reindex(te).values
    return [dict(cohort=COH, rep=rep, cond=cond, transform=t, adjust=a, rho=float(spearmanr(p, ETA)[0])) for (t, a), p in P.items()]


def checks():
    n = 0; rng = np.random.default_rng(0)
    # (1) EB: tanpa efek sebenarnya -> lambda kecil; efek kuat -> mendekati OLS; intersep: rata-rata residu latih 0; invarian skala fitur
    Dt = D[:160]; M0 = rng.standard_normal((160, 200)); B0, l0 = K.eb_coef(Dt, M0); assert np.median(l0[1:]) < 0.5, l0; n += 1
    Bs = rng.standard_normal((D.shape[1], 200)) * 0.5; Bs[1] /= D[:, 1].std(); M1 = Dt @ Bs + 0.05 * rng.standard_normal((160, 200)); B1, l1 = K.eb_coef(Dt, M1); assert l1[1:].min() > 0.99, l1; assert np.abs(B1 - K.ols_coef(Dt, M1)).max() < 0.05; n += 1
    assert np.abs((M1 - Dt @ B1).mean(0)).max() < 1e-9; n += 1
    sc = rng.uniform(0.01, 100, 200); assert np.allclose(K.eb_coef(Dt, M0 * sc)[1], l0) and np.allclose(K.eb_coef(Dt, M0 * sc)[0], B0 * sc); n += 1
    # (1b) ash: tanpa efek -> koefisien ~0; efek jarang kuat -> efek kuat dipertahankan, galat lebih kecil daripada OLS dan EB normal tunggal
    Ba, fa = K.ash_coef(Dt, M0); assert np.abs(fa[1:]).mean(1).max() < 0.05 and np.median(np.abs(fa[1:])) < 0.01; n += 1          # tanpa efek: rata-rata faktor < 0.05, median < 0.01
    Bt = np.zeros((D.shape[1], 120)); Bt[1, :20] = rng.choice([-1, 1], 20) * 2.5 / D[:, 1].std(); M2 = Dt @ Bt + rng.standard_normal((160, 120))
    Ba, fa = K.ash_coef(Dt, M2); Be = K.eb_coef(Dt, M2)[0]; Bo = K.ols_coef(Dt, M2); e = lambda B: np.abs(B[1] - Bt[1]).mean()
    assert fa[1, :20].mean() > 0.9 and e(Ba) < e(Be) and e(Ba) < e(Bo), (fa[1, :20].mean(), e(Ba), e(Be), e(Bo)); n += 1
    assert np.allclose(K.ash_coef(Dt, M2 * sc[:120])[0], Ba * sc[:120]); n += 1                            # invarian skala fitur
    # (2) penyesuaian hanya dipasang pada pasien latih
    X, owner = data_for(1, "H1"); tr = np.arange(0, int(0.8 * N)); te_rows = ~np.isin(owner, tr); Z = make_transform("rank", X[np.isin(owner, tr)])(X)
    for a in ("pat", "eb", "ash"):
        Xp = Z.copy(); Xp[te_rows] = Xp[te_rows] * 5 + 3; assert np.allclose(K.adjust_units(Z, owner, tr, a, D)[~te_rows], K.adjust_units(Xp, owner, tr, a, D)[~te_rows]); n += 1
    # (3) 'pat' identik dengan fungsi putaran 42/50
    old_adj = (C.adjust_units if COH == "tuep" else EM.adjust_units); assert np.allclose(K.adjust_units(Z, owner, tr, "pat", D), old_adj(Z, owner, tr, "pat")); n += 1
    # (4) kondisi E: efek dalam skala log persis Dc @ BT; H1 dan E memakai permutasi yang sama
    XE, oE = data_for(1, "E"); assert np.array_equal(oE, owner); Dc = D[:, 1:] - D[:, 1:].mean(0); assert np.abs(np.log(XE) - np.log(X.astype(np.float64)) - Dc[owner] @ BT).max() < 1e-5; n += 1                   # XR TUEP float32
    # (5) BT bukan fungsi label: mengubah Y tidak memengaruhi (BT hanya dari XR, owner, D)
    assert np.allclose(K.true_effect(XR, OWN0, D), BT); n += 1
    print(f"{n} pemeriksaan lulus; |BT| median {np.median(np.abs(BT)):.3f}, maks {np.abs(BT).max():.3f}; jendela {len(XR)}, pasien {N}", flush=True)
    # (6) reproduksi: rep 0, H1, gamma 0, empat pipeline lama harus identik dengan hasil tersimpan (RF; XGB GPU dicatat saja)
    if FAM == "RF":
        t0 = time.time(); old_pipes = [("raw", "none"), ("raw", "pat"), ("log", "pat"), ("rank", "pat")]; new = pd.DataFrame(unit("RF", 0, "H1", 0.0, old_pipes)); old = pd.read_csv(OLD["RF"])
        old = old[(old.rep == 0) & (old.gamma == 0.0) & ((old["cond"] == "H1") if "cond" in old else True)]
        m = new.merge(old, on=["transform", "adjust"], suffixes=("", "_old")); assert len(m) == 4, len(m)
        d = max(np.abs(m[c] - m[c + "_old"]).max() for c in ["auc"] + FC); print(f"reproduksi RF rep 0 H1 gamma 0 terhadap {os.path.basename(OLD['RF'])}: selisih maks {d:.2e} [{time.time() - t0:.0f}s]", flush=True); assert d < 1e-9


if __name__ == "__main__":
    t0 = time.time()
    if MODE == "--check":
        checks()
    elif MODE == "--run":
        path = f"{OUT}/{COH}_{FAM}.csv"
        done = set(map(tuple, pd.read_csv(path, usecols=["rep", "cond", "gamma"]).drop_duplicates().values.tolist())) if os.path.exists(path) else set()
        units = [u for u in UNITS if u not in done]; print(f"{COH} {FAM}: {len(units)} unit tersisa dari {len(UNITS)}", flush=True)
        if FAM == "XGB":
            for i, (r, c, g) in enumerate(units):
                pd.DataFrame(unit(FAM, r, c, g)).to_csv(path, mode="a", header=not os.path.exists(path), index=False)
                if i % 20 == 19:
                    print(f"{i + 1}/{len(units)} [{time.time() - t0:.0f}s]", flush=True)
        else:
            nj = int(os.environ.get("NJOBS", 12))
            for i in range(0, len(units), nj * 2):
                res = Parallel(n_jobs=nj)(delayed(unit)(FAM, r, c, g) for r, c, g in units[i:i + nj * 2])
                for rr in res:
                    pd.DataFrame(rr).to_csv(path, mode="a", header=not os.path.exists(path), index=False)
                print(f"{min(i + nj * 2, len(units))}/{len(units)} [{time.time() - t0:.0f}s]", flush=True)
    elif MODE == "--rho":
        path = f"{OUT}/rho_{COH}.csv"; units = [(r, c) for c in RHO_CONDS for r in range(30)]
        res = Parallel(n_jobs=int(os.environ.get("NJOBS", 12)))(delayed(rho_unit)(r, c) for r, c in units); pd.DataFrame(sum(res, [])).to_csv(path, index=False)
    print(f"selesai {time.time() - t0:.0f}s", flush=True)
