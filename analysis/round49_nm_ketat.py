#!/usr/bin/env python3
"""Round 49A, NM part (GPU): NM-V4 with strict nesting: the V4 coefficients (segment-weighted) and the standardization of the age/sex summary are fitted
only on the training patients, without the 20% validation patients used to choose the configuration and for early stopping. See PRAREGISTRASI_putaran49.md.
One GPU process at a time. Usage: python round49_nm_ketat.py --check | --run   (resumable; one JSON per fold in hasil49/nm_ketat/)"""
import os, sys, json, time, hashlib
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, "/home/miftah_f/RTX-5090/tuep_fresh/code")
import numpy as np, torch
from sklearn.metrics import roc_auc_score
import round30_nm as M
nm, tf, XSEG, PID, Y, AS, DES, N, POS, JOBS, SUBJ = M.nm, M.tf, M.XSEG, M.PID, M.Y, M.AS, M.DES, M.N, M.POS, M.JOBS, M.SUBJ
OUT = os.path.join(HERE, "hasil49"); os.makedirs(f"{OUT}/nm_ketat", exist_ok=True)


def tokens_fit(X, pid, fit_p):
    """Same as round30_nm.tokens(variant='V4') but all statistics from the `fit_p` patients only."""
    fm = np.isin(pid, fit_p); Dseg = DES[pid]
    B = np.linalg.lstsq(Dseg[fm], X[fm], rcond=None)[0]; sd = (X[fm] - Dseg[fm] @ B).std(0, ddof=np.linalg.matrix_rank(Dseg[fm]))
    T = (X - Dseg @ B) / sd; toks = [T[pid == i].astype(np.float32) for i in range(N)]
    summ = np.column_stack([np.vstack([np.median(t, 0) for t in toks]), (AS - AS[fit_p].mean(0)) / AS[fit_p].std(0)]).astype(np.float32)
    return toks, summ


def run_fold(r, k, tr, te, strict=True):
    ylab = {s: int(Y[POS[s]]) for s in SUBJ}; tr_p, va_p = tf.split_val(list(SUBJ[tr]), ylab, 100 * r + k)
    tr_i, va_i = np.array([POS[s] for s in tr_p]), np.array([POS[s] for s in va_p])
    assert not (set(tr_i) | set(va_i)) & set(te) and set(tr_i) | set(va_i) == set(tr) and not set(tr_i) & set(va_i)
    toks, summ = tokens_fit(XSEG, PID, tr_i if strict else tr); t0 = time.time()
    g = {f"{d}_{L}": nm.train_one(toks, summ, tr_i, va_i, d, L, 1000 * r + 10 * k, True, nm.MAX_EPOCHS)[1] for d, L in nm.GRID}
    bd, bL = map(int, max(g, key=g.get).split("_")); preds, eps = [], []
    for s in range(nm.SEEDS["F"]):
        m, va, ep = nm.train_one(toks, summ, tr_i, va_i, bd, bL, 7919 * (s + 1) + 1000 * r + 10 * k, True, nm.MAX_EPOCHS); preds.append(nm.predict(m, toks, summ, te, bL)); eps.append(ep)
    return dict(r=r, k=k, variant="V4-ketat" if strict else "V4-longgar", grid_val_auc=g, chosen=[bd, bL], epochs=eps, test={SUBJ[i]: float(p) for i, p in zip(te, np.mean(preds, 0))},
                seconds=time.time() - t0, test_auc=float(roc_auc_score(Y[te], np.mean(preds, 0))))


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "--check"
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    json.dump(dict(prereg_sha256=sha(os.path.join(HERE, "PRAREGISTRASI_putaran49.md")), code_sha256=sha(__file__)), open(f"{OUT}/meta_nm.json", "w"), indent=1)
    if mode == "--check":                                   # loose mode reproduces fold r0k0 of round 30; the strict-version test AUC is NOT printed
        r, k, tr, te = JOBS[0]; old = json.load(open(os.path.join(HERE, "hasil30", "nm_v4", f"r{r}_k{k}.json"))); res = run_fold(r, k, tr, te, strict=False)
        d = max(abs(res["test"][s] - old["test"][s]) for s in old["test"]); print("longgar r0k0: AUC lama", old["test_auc"], "baru", res["test_auc"], "selisih prediksi maks", f"{d:.2e}", "dipilih", old["chosen"], res["chosen"], flush=True)
        assert d < 5e-3; print("checks passed"); sys.exit()
    sh = sys.argv.index("--shard") if "--shard" in sys.argv else None; si, sn = (int(sys.argv[sh + 1]), int(sys.argv[sh + 2])) if sh else (0, 1)   # fold split across processes (execution only)
    for j_, (r, k, tr, te) in enumerate(JOBS):
        if j_ % sn != si:
            continue
        path = f"{OUT}/nm_ketat/r{r}_k{k}.json"
        if os.path.exists(path):
            continue
        t0 = time.time(); res = run_fold(r, k, tr, te); json.dump(res, open(path, "w")); print(f"fold r{r}k{k} {(time.time() - t0) / 60:.1f}min", flush=True)
    if len([f for f in os.listdir(f"{OUT}/nm_ketat") if f.endswith(".json")]) == len(JOBS):
        open(f"{OUT}/NM_SELESAI", "w").write(time.ctime())
    print("SHARD DONE", flush=True)
