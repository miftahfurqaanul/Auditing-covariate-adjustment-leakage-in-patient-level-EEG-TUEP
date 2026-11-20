#!/usr/bin/env python3
"""Putaran 55 (CPU): simulasi referensi kontrol untuk klaim (ii). Lihat PRAREGISTRASI_putaran55.md (SHA-256 dicatat sebelum run).
Pemakaian: CUDA_VISIBLE_DEVICES="" python round55_referensi_kontrol.py --check | --run [njobs]   (keluaran ke hasil55/)
Run dapat dilanjutkan: unit (sel, replikasi) yang sudah ada di hasil55/replikasi.csv dilewati."""
import os, sys, time, json, hashlib, zlib
os.environ["CUDA_VISIBLE_DEVICES"] = ""; os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ["PYTHONWARNINGS"] = "ignore"
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, "/home/miftah_f/RTX-5090/tuep_fresh/code")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from joblib import Parallel, delayed
import evalkit as ek, round4 as r4

OUT = os.path.join(HERE, "hasil55"); os.makedirs(OUT, exist_ok=True); CSV = os.path.join(OUT, "replikasi.csv")
NREFS = (20, 40, 80, 200, 1000); DESIGNS = ("jarang", "cermin"); SS = (0.0, 0.5, 1.0); NREP = 100; P = 114
ERA_LEVELS = sorted(set(r4.YEAR_G)); ERA_REF = pd.Series(r4.YEAR_G).value_counts().idxmax()
COV = np.column_stack([r4.AGE, r4.MALE, np.array([ERA_LEVELS.index(e) for e in r4.YEAR_G])]); YT = np.asarray(r4.Y)
R = np.corrcoef(np.asarray(r4.BP, float), rowvar=False); LCH = np.linalg.cholesky(R + 1e-10 * np.eye(P))


def design(cov):
    era = cov[:, 2].astype(int)
    return np.column_stack([np.ones(len(cov)), cov[:, 0], cov[:, 1]] + [(era == ERA_LEVELS.index(l)).astype(float) for l in ERA_LEVELS if l != ERA_REF])


def fit_norm(X, D, idx):
    """identik dengan round29.fit_norm / round4.normative_z: OLS pada idx, SD residu dengan ddof = rank."""
    B = np.linalg.lstsq(D[idx], X[idx], rcond=None)[0]
    sd = (X[idx] - D[idx] @ B).std(0, ddof=np.linalg.matrix_rank(D[idx])); return (X - D @ B) / sd


def seed_of(*k):
    return zlib.crc32("|".join(map(str, k)).encode()) + 55_000_000


def gen(nref, des, s, rep):
    rng = np.random.default_rng(seed_of("data", nref, des, s, rep)); N = int(2.5 * nref); y = np.r_[np.zeros(N // 2, int), np.ones(N // 2, int)]
    src = y.copy() if des == "jarang" else 1 - y            # kelas sumber baris TUEP: jarang -> kontrol dari non-epilepsi (0); cermin -> ditukar
    own = rng.random(N) < s; rows = np.empty(N, int)
    for i in range(N):
        pool = np.where(YT == src[i])[0] if own[i] else np.arange(len(YT)); rows[i] = pool[rng.integers(len(pool))]
    X = rng.standard_normal((N, P)) @ LCH.T
    return X, y, design(COV[rows])


def unit(nref, des, s, rep):
    X, y, D = gen(nref, des, s, rep); N = len(y)
    skf = StratifiedKFold(5, shuffle=True, random_state=seed_of("fold", nref, des, s, rep) % (2**31))
    sc = {v: np.full(N, np.nan) for v in ("C1", "CR", "AP")}; fa = {v: [] for v in sc}
    for tr, te in skf.split(np.zeros(N), y):
        ctrl = tr[y[tr] == 0]
        for v, F in (("C1", X), ("CR", fit_norm(X, D, ctrl)), ("AP", fit_norm(X, D, tr))):
            p = ek.nested_lr().fit(F[tr], y[tr]).predict_proba(F[te])[:, 1]; sc[v][te] = p; fa[v].append(roc_auc_score(y[te], p))
    era = D[:, 3:]; ctrl_all = y == 0
    lv = np.c_[era, 1 - era.sum(1)]; mc = int(lv[ctrl_all].sum(0).min())   # sel era terkecil di antara semua kontrol (sebelum pembagian)
    return dict(nref=nref, design=des, s=s, rep=rep, N=N, min_ctrl_era_cell=mc,
                **{f"{v}_pooled": roc_auc_score(y, sc[v]) for v in sc}, **{f"{v}_fold": float(np.mean(fa[v])) for v in sc})


def cells():
    return [(n, d, s) for n in NREFS for d in DESIGNS for s in SS]


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def check():
    t0 = time.time(); ok = 0
    # (1) fit_norm identik dengan round4.normative_z pada data TUEP nyata (lipatan kanonik 0, kontrol latih)
    import round17_common as c17
    tr = c17.JOBS[0][2]; z4 = r4.normative_z(r4.DESIGN["I2"], tr); z55 = fit_norm(np.asarray(r4.BP, float), r4.DESIGN["I2"], tr[YT[tr] == 0])
    assert np.max(np.abs(z4 - z55)) < 1e-9, np.max(np.abs(z4 - z55)); ok += 1
    assert np.allclose(design(COV), r4.DESIGN["I2"]), "desain sintetis != DESIGN I2"; ok += 1
    # (2) sifat data sintetis
    X, y, D = gen(80, "jarang", 1.0, 0); assert len(y) == 200 and y.sum() == 100 and X.shape == (200, P); ok += 1
    big = [gen(1000, d, 1.0, 0) for d in DESIGNS]
    for (Xb, yb, Db), d in zip(big, DESIGNS):
        e0 = Db[yb == 0][:, 3:]; frac = (np.c_[e0, 1 - e0.sum(1)]).mean(0); print(d, "proporsi era kontrol (level non-ref..., ref):", frac.round(3))
    Xb, yb, Db = big[0]; assert abs(np.corrcoef(Xb, rowvar=False)[0, 1] - R[0, 1]) < 0.05; ok += 1
    X0, y0, D0 = gen(1000, "jarang", 0.0, 0); assert abs(D0[y0 == 0, 1].mean() - D0[y0 == 1, 1].mean()) < 3.0; ok += 1   # s=0: usia sama
    # (3) residu kontrol ortogonal terhadap desain pada kontrol latih (CR), tidak pada kasus
    ctrl = np.where(y == 0)[0]; Z = fit_norm(X, D, ctrl); assert np.abs(D[ctrl].T @ Z[ctrl]).max() < 1e-8; ok += 1
    # (4) determinisme dan waktu satu unit per n_ref
    a = unit(20, "jarang", 1.0, 3); b = unit(20, "jarang", 1.0, 3); assert a == b; ok += 1
    for n in NREFS:
        t1 = time.time(); u = unit(n, "jarang", 1.0, 0); print(f"n_ref {n}: {time.time() - t1:.1f} s, min sel era kontrol {u['min_ctrl_era_cell']} (AUC tidak dicetak)")
    print(f"PEMERIKSAAN: {ok} lulus ({time.time() - t0:.0f} s)")


def run(njobs):
    done = set()
    if os.path.exists(CSV):
        d = pd.read_csv(CSV); done = set(zip(d.nref, d.design, d.s, d.rep))
    todo = [(n, d, s, r) for (n, d, s) in cells() for r in range(NREP) if (n, d, s, r) not in done]
    todo.sort(key=lambda u: -u[0])                      # unit besar dulu
    json.dump(dict(prereg_sha256=sha(os.path.join(HERE, "PRAREGISTRASI_putaran55.md")), code_sha256=sha(__file__), njobs=njobs, todo=len(todo)), open(os.path.join(OUT, "meta.json"), "w"), indent=1)
    print(f"unit tersisa {len(todo)} dari {len(cells()) * NREP}", flush=True); t0 = time.time(); k = 0
    for row in Parallel(n_jobs=njobs, return_as="generator_unordered", batch_size=1)(delayed(unit)(*u) for u in todo):
        pd.DataFrame([row]).to_csv(CSV, mode="a", header=not os.path.exists(CSV), index=False); k += 1
        if k % 100 == 0:
            print(f"  {k}/{len(todo)} ({(time.time() - t0) / 60:.1f} menit)", flush=True)
    print(f"selesai {k} unit dalam {(time.time() - t0) / 60:.1f} menit", flush=True)


if __name__ == "__main__":
    m = sys.argv[1] if len(sys.argv) > 1 else "--check"
    check() if m == "--check" else run(int(sys.argv[2]) if len(sys.argv) > 2 else 22)
