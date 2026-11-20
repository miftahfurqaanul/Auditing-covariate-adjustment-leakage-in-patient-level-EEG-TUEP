#!/usr/bin/env python3
"""Putaran 52 (GPU): NormMamba (NM) dengan penyesuaian ash (round51_common.ash_coef) di tingkat segmen via rata-rata per pasien.
(a) label nyata: nesting ketat seperti round49_nm_ketat (koefisien dan standardisasi hanya dari pasien latih tanpa validasi); (b) plasmode H1 seperti round42_gpu_nn.nm_unit
(koefisien dari semua pasien latih), pipeline raw-ash dan rank-ash. Kode putaran 42/49/51 tidak diubah. Lihat PRAREGISTRASI_putaran52.md.
Pemakaian: python round52_nm.py --check | --run real [--shard i n] | --run plasmode [--shard i n] [n_rep]   (resumable; hasil52/)"""
import os, sys, json, time, hashlib
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, "/home/miftah_f/RTX-5090/tuep_fresh/code")
os.environ["PYTHONWARNINGS"] = "ignore"; os.environ.pop("CUDA_VISIBLE_DEVICES", None)
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, torch
torch.zeros(1, device="cuda"); assert torch.cuda.device_count() >= 1, "GPU tidak terlihat"
from sklearn.metrics import roc_auc_score
import round42_gpu_nn as G
import round49_nm_ketat as R49
import round51_common as K
M = R49.M; nm, tf, XSEG, PID, Y, AS, DES, N, POS, JOBS, SUBJ = R49.nm, R49.tf, R49.XSEG, R49.PID, R49.Y, R49.AS, R49.DES, R49.N, R49.POS, R49.JOBS, R49.SUBJ
OUT = os.path.join(HERE, "hasil52"); os.makedirs(f"{OUT}/nm_ash", exist_ok=True)


def tokens_ash(X, pid, fit_p, no_as=False, coef="ash"):
    """Seperti R49.tokens_fit (coef='ols_seg': OLS tertimbang segmen) tetapi koefisien dari ash atas rata-rata per pasien (coef='ash'); statistik hanya dari pasien fit_p.
    no_as=True meniru round30_nm.tokens(..., no_as=True) (kolom usia/jenis kelamin summ = 0)."""
    fm = np.isin(pid, fit_p); Dseg = DES[pid]
    if coef == "ash":
        Mx = K.patient_means(X, pid, N); B = K.ash_coef(DES[fit_p], Mx[fit_p])[0]
    else:
        B = np.linalg.lstsq(Dseg[fm], X[fm], rcond=None)[0]
    sd = (X[fm] - Dseg[fm] @ B).std(0, ddof=np.linalg.matrix_rank(Dseg[fm])); T = (X - Dseg @ B) / sd; toks = [T[pid == i].astype(np.float32) for i in range(N)]
    summ = np.column_stack([np.vstack([np.median(t, 0) for t in toks]), (AS - AS[fit_p].mean(0)) / AS[fit_p].std(0)]).astype(np.float32)
    if no_as:
        summ[:, 114:] = 0.0
    return toks, summ


def run_fold(r, k, tr, te):
    ylab = {s: int(Y[POS[s]]) for s in SUBJ}; tr_p, va_p = tf.split_val(list(SUBJ[tr]), ylab, 100 * r + k)
    tr_i, va_i = np.array([POS[s] for s in tr_p]), np.array([POS[s] for s in va_p])
    assert not (set(tr_i) | set(va_i)) & set(te) and set(tr_i) | set(va_i) == set(tr) and not set(tr_i) & set(va_i)
    toks, summ = tokens_ash(XSEG, PID, tr_i); t0 = time.time()
    g = {f"{d}_{L}": nm.train_one(toks, summ, tr_i, va_i, d, L, 1000 * r + 10 * k, True, nm.MAX_EPOCHS)[1] for d, L in nm.GRID}
    bd, bL = map(int, max(g, key=g.get).split("_")); preds, eps = [], []
    for s in range(nm.SEEDS["F"]):
        m, va, ep = nm.train_one(toks, summ, tr_i, va_i, bd, bL, 7919 * (s + 1) + 1000 * r + 10 * k, True, nm.MAX_EPOCHS); preds.append(nm.predict(m, toks, summ, te, bL)); eps.append(ep)
    return dict(r=r, k=k, variant="ash-ketat", grid_val_auc=g, chosen=[bd, bL], epochs=eps, test={SUBJ[i]: float(p) for i, p in zip(te, np.mean(preds, 0))},
                seconds=time.time() - t0, test_auc=float(roc_auc_score(Y[te], np.mean(preds, 0))))


PIPES_ASH = [("raw", "ash"), ("rank", "ash")]


def nm_unit_ash(rep, gamma, M_, N38, dev):
    """Salinan G.nm_unit; satu-satunya perubahan: pipeline ash dengan tokens_ash (koefisien dari semua pasien latih tr, seperti M.tokens(..., 'V4', no_as=True))."""
    y = G.C.gen_labels(rep, gamma); auc_cov = float(roc_auc_score(y, G.C.ETA)); pi = G.C.gen_perm(rep, "H1"); XSEG_, PID_ = M_.XSEG, M_.PID
    X2 = np.vstack([XSEG_[PID_ == pi[i]] for i in range(N)]); pid2 = np.concatenate([np.full(int((PID_ == pi[i]).sum()), i) for i in range(N)]); folds = G.C.folds_for(rep, gamma, y)
    nm.Y = y.astype(M_.Y.dtype if hasattr(M_.Y, "dtype") else int); ylab = {s: int(y[i]) for i, s in enumerate(G.C.SUBJ)}
    P = {p: np.full(N, np.nan) for p in PIPES_ASH}
    for fi, (tr, te) in enumerate(folds):
        tr_p, va_p = N38.tf.split_val(list(G.C.SUBJ[tr]), ylab, G.C.kseed("plasmode_val", rep, gamma, fi)); tr_i, va_i = np.array([M_.POS[s] for s in tr_p]), np.array([M_.POS[s] for s in va_p]); seed = G.C.kseed("plasmode_nm", rep, gamma, fi) % 100000
        for (t, v) in PIPES_ASH:
            Xt = G.nm_rank(X2, np.isin(pid2, tr)) if t == "rank" else X2
            toks, summ = tokens_ash(Xt, pid2, tr, no_as=True); m, va, ep = nm.train_one(toks, summ, tr_i, va_i, 32, 32, seed, True, 20); P[(t, v)][te] = nm.predict(m, toks, summ, te, 32)
    rows = []
    for (t, v), p in P.items():
        fa = G.C.fold_aucs(y, p, folds); rows.append(dict(fam="NM", rep=rep, cond="H1", gamma=gamma, transform=t, adjust=v, auc=float(roc_auc_score(y, p)), **{f"fold_auc_{i}": x for i, x in enumerate(fa)}, auc_cov=auc_cov))
    return rows


def check():
    n = 0; tr = JOBS[0][2]; a = np.array(sorted(tr))[:120]
    t1, s1 = tokens_ash(XSEG, PID, a); X7 = XSEG.copy(); X7[~np.isin(PID, a)] *= 7.0; t2, s2 = tokens_ash(X7, PID, a)
    assert all(np.allclose(t1[i], t2[i]) for i in a) and np.allclose(s1[a], s2[a]); n += 1                                       # segmen di luar himpunan pemasangan tidak mengubah fitur pemasangan
    to, so = R49.tokens_fit(XSEG, PID, a); tp, sp = tokens_ash(XSEG, PID, a, coef="ols_seg"); assert all(np.allclose(to[i], tp[i]) for i in range(N)) and np.allclose(so, sp); n += 1   # pipa identik dengan round49 bila koefisien = OLS tertimbang segmen
    tn, sn = tokens_ash(XSEG, PID, a, no_as=True); assert np.all(sn[:, 114:] == 0) and np.allclose(sn[:, :114], s1[:, :114]); n += 1
    Mx = K.patient_means(XSEG, PID, N); Ba, fac = K.ash_coef(DES[a], Mx[a]); assert np.isfinite(Ba).all() and fac[1:].max() <= 1.0 + 1e-9; n += 1
    print(f"{n} pemeriksaan lulus; faktor ash pada rata-rata segmen (lipatan 0, 120 pasien): median {np.median(fac[1:]):.3f}, maks {fac[1:].max():.3f}", flush=True)
    M_, N38 = G.nm_setup(); dev = torch.device("cuda"); t0 = time.time()
    new = pd.DataFrame(G.nm_unit(0, 0.0, M_, N38, dev)); old = pd.read_csv(os.path.join(HERE, "hasil42", "NM.csv")); old = old[(old.rep == 0) & (old.gamma == 0.0)]
    m = new.merge(old, on=["transform", "adjust"], suffixes=("", "_42")); FC = [f"fold_auc_{i}" for i in range(5)]; d = max(np.abs(m[c] - m[c + "_42"]).max() for c in ["auc"] + FC)
    print(f"reproduksi NM plasmode rep 0 gamma 0 ({len(m)} pipeline): selisih maks {d:.2e} [{time.time() - t0:.0f}s]", flush=True); assert len(m) == 3 and d < 1e-6


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "--check"; torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False; t0 = time.time()
    if mode == "--check":
        check(); sys.exit()
    json.dump(dict(prereg_sha256=sha(os.path.join(HERE, "PRAREGISTRASI_putaran52.md")), code_sha256=sha(__file__), common_sha256=sha(os.path.join(HERE, "round51_common.py"))), open(f"{OUT}/meta_nm.json", "w"), indent=1)
    what = sys.argv[2]; sh = sys.argv.index("--shard") if "--shard" in sys.argv else None; si, sn = (int(sys.argv[sh + 1]), int(sys.argv[sh + 2])) if sh else (0, 1)
    if what == "real":
        for j_, (r, k, tr, te) in enumerate(JOBS):
            if j_ % sn != si:
                continue
            path = f"{OUT}/nm_ash/r{r}_k{k}.json"
            if os.path.exists(path):
                continue
            t1 = time.time(); res = run_fold(r, k, tr, te); json.dump(res, open(path + ".tmp", "w")); os.replace(path + ".tmp", path); print(f"lipatan r{r}k{k} {(time.time() - t1) / 60:.1f} menit", flush=True)
    else:
        nrep = 20; path = f"{OUT}/nm_plasmode_s{si}.csv"; M_, N38 = G.nm_setup(); dev = torch.device("cuda")
        done = set(map(tuple, pd.read_csv(path, usecols=["rep", "gamma"]).drop_duplicates().values.tolist())) if os.path.exists(path) else set(); idx = 0
        for rep in range(nrep):
            for g in (0.0, 1.0):
                idx += 1
                if (idx - 1) % sn != si or (rep, g) in done:
                    continue
                pd.DataFrame(nm_unit_ash(rep, g, M_, N38, dev)).to_csv(path, mode="a", header=not os.path.exists(path), index=False); print(f"NM-ash rep {rep} gamma {g} [{time.time() - t0:.0f}s]", flush=True)
    print(f"selesai {time.time() - t0:.0f}s", flush=True)
