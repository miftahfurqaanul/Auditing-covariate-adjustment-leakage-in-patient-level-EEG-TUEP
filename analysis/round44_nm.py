#!/usr/bin/env python3
"""Round 44 (GPU): NM-V4 WITHOUT age/sex and with design D2 (age^2 term), 50 canonical folds; NM-V4 recipe of round 30 ((d, L) grid by inner validation, 5 seeds). See PRAREGISTRASI_putaran44.md.
One GPU process at a time. Usage: python round44_nm.py --check | --run   (resumable per fold)"""
import os, sys, json, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
os.environ["PYTHONWARNINGS"] = "ignore"
import numpy as np, torch
assert ("--check" in sys.argv) or torch.zeros(1, device="cuda").is_cuda                                             # initialize CUDA before the old module hides the GPU
import round30_nm as M
os.environ.pop("CUDA_VISIBLE_DEVICES", None)
from sklearn.metrics import roc_auc_score
nm, tf = M.nm, M.tf; OUT = os.path.join(HERE, "hasil44", "nm_clean"); os.makedirs(OUT, exist_ok=True)
Y, SUBJ, POS, N, JOBS, XSEG, PID, AS = M.Y, M.SUBJ, M.POS, M.N, M.JOBS, M.XSEG, M.PID, M.AS
AGE = np.asarray(M.r4.AGE, float); D2 = np.column_stack([M.DES, ((AGE - AGE.mean()) / AGE.std()) ** 2]); M.DES = D2             # tokens() memakai DES global modul round30_nm


def run_fold_clean(r, k, tr, te, max_epochs=nm.MAX_EPOCHS):
    ylab = {s: int(Y[POS[s]]) for s in SUBJ}; tr_p, va_p = tf.split_val(list(SUBJ[tr]), ylab, 100 * r + k)
    tr_i, va_i = np.array([POS[s] for s in tr_p]), np.array([POS[s] for s in va_p]); assert not (set(tr_i) | set(va_i)) & set(te) and set(tr_i) | set(va_i) == set(tr)
    toks, summ = M.tokens(XSEG, PID, tr, "V4", no_as=True); t0 = time.time()
    g = {f"{d}_{L}": nm.train_one(toks, summ, tr_i, va_i, d, L, 1000 * r + 10 * k, True, max_epochs)[1] for d, L in nm.GRID}
    bd, bL = map(int, max(g, key=g.get).split("_")); preds, eps = [], []
    for s in range(nm.SEEDS["F"]):
        m, va, ep = nm.train_one(toks, summ, tr_i, va_i, bd, bL, 7919 * (s + 1) + 1000 * r + 10 * k, True, max_epochs); preds.append(nm.predict(m, toks, summ, te, bL)); eps.append(ep)
    return dict(r=r, k=k, variant="V4-noAS-D2", grid_val_auc=g, chosen=[bd, bL], epochs=eps, test={SUBJ[i]: float(p) for i, p in zip(te, np.mean(preds, 0))}, seconds=time.time() - t0, test_auc=float(roc_auc_score(Y[te], np.mean(preds, 0))))


def assemble():
    P = np.full((10, N), np.nan)
    for r, k, tr, te in JOBS:
        d = json.load(open(f"{OUT}/r{r}_k{k}.json"))["test"]; P[r, te] = [d[s] for s in SUBJ[te]]
    assert not np.isnan(P).any(); np.savez(os.path.join(HERE, "hasil44", "E_nm_prediksi.npz"), NM_clean=P); print("ALL FOLDS DONE; NM_clean matrix saved", flush=True)


def checks():
    r, k, tr, te = JOBS[0]; toks, summ = M.tokens(XSEG, PID, tr, "V4", no_as=True); assert np.all(summ[:, 114:] == 0) and D2.shape == (N, 8)
    Dseg = D2[PID]; fm = np.isin(PID, tr); B = np.linalg.lstsq(Dseg[fm], XSEG[fm], rcond=None)[0]; sd = (XSEG[fm] - Dseg[fm] @ B).std(0, ddof=np.linalg.matrix_rank(Dseg[fm])); T = (XSEG - Dseg @ B) / sd
    worst = max(np.abs(toks[i] - T[PID == i].astype(np.float32)).max() for i in range(N)); assert worst < 1e-4, worst
    X2 = XSEG.copy(); X2[np.isin(PID, te)] = X2[np.isin(PID, te)] * 5 + 3; toks2, _ = M.tokens(X2, PID, tr, "V4", no_as=True); assert all(np.array_equal(toks[i], toks2[i]) for i in tr)
    print("3 checks passed (D2 token without AS equals the second implementation; AS summary zero; test data do not change training tokens)", flush=True)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "--check"; torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    if mode == "--check":
        checks()
    elif mode == "--assemble":
        assemble()
    else:
        checks(); shard = (int(sys.argv[sys.argv.index("--shard") + 1]), int(sys.argv[sys.argv.index("--shard") + 2])) if "--shard" in sys.argv else (0, 1)
        for idx, (r, k, tr, te) in enumerate(JOBS):
            if idx % shard[1] != shard[0]:
                continue
            path = f"{OUT}/r{r}_k{k}.json"
            if os.path.exists(path):
                continue
            t0 = time.time(); res = run_fold_clean(r, k, tr, te); tmp = path + ".tmp"; json.dump(res, open(tmp, "w")); os.replace(tmp, path); print(f"fold r{r}k{k} {(time.time() - t0) / 60:.1f}min: test AUC {res['test_auc']:.3f}; chosen {res['chosen']}", flush=True)
        print(f"SHARD {shard[0]}/{shard[1]} DONE", flush=True)
