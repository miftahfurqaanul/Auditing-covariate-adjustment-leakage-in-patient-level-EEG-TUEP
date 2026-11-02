#!/usr/bin/env python3
"""Round 30, NM part (GPU): NormMamba-F with the V4 normative layer (covariate regression on the segments of ALL training patients), reduced version of the leakage test,
and reproduction of one V0 fold. See PRAREGISTRASI_putaran30.md. One GPU process at a time.
Usage: python round30_nm.py --repro | --dry-run | --run | --null [n_replicates]"""
import os, sys, json, time, hashlib
HERE = os.path.dirname(os.path.abspath(__file__)); CODE = "/home/miftah_f/RTX-5090/tuep_fresh/code"; sys.path.insert(0, CODE)
import numpy as np, pandas as pd, torch
from sklearn.metrics import roc_auc_score
import normmamba as nm, round4 as r4, tang_fixed as tf

OUT = os.path.join(HERE, "hasil30"); os.makedirs(f"{OUT}/nm_v4", exist_ok=True)
XSEG, PID, Y, AS, DES, N, POS, JOBS, DEV = nm.XSEG, nm.PID, nm.Y, nm.AS, nm.DES, nm.N, nm.POS, nm.JOBS, nm.DEV
SUBJ = r4.SUBJ


def tokens(X, pid, tr, variant, no_as=False):
    """variant 'V0': coefficients from the training control segments (old layer); 'V4': from the segments of all training patients."""
    fit_p = tr[Y[tr] == 0] if variant == "V0" else tr
    fm = np.isin(pid, fit_p); Dseg = DES[pid]
    B = np.linalg.lstsq(Dseg[fm], X[fm], rcond=None)[0]; sd = (X[fm] - Dseg[fm] @ B).std(0, ddof=np.linalg.matrix_rank(Dseg[fm]))
    T = (X - Dseg @ B) / sd
    toks = [T[pid == i].astype(np.float32) for i in range(N)]
    summ = np.column_stack([np.vstack([np.median(t, 0) for t in toks]), (AS - AS[tr].mean(0)) / AS[tr].std(0)]).astype(np.float32)
    if no_as:
        summ[:, 114:] = 0.0
    return toks, summ


def run_fold_v4(r, k, tr, te, variant="V4", max_epochs=nm.MAX_EPOCHS):
    ylab = {s: int(Y[POS[s]]) for s in SUBJ}; tr_p, va_p = tf.split_val(list(SUBJ[tr]), ylab, 100 * r + k)
    tr_i, va_i = np.array([POS[s] for s in tr_p]), np.array([POS[s] for s in va_p])
    assert not (set(tr_i) | set(va_i)) & set(te) and set(tr_i) | set(va_i) == set(tr)
    toks, summ = tokens(XSEG, PID, tr, variant); t0 = time.time()
    g = {f"{d}_{L}": nm.train_one(toks, summ, tr_i, va_i, d, L, 1000 * r + 10 * k, True, max_epochs)[1] for d, L in nm.GRID}
    bd, bL = map(int, max(g, key=g.get).split("_")); preds, eps = [], []
    for s in range(nm.SEEDS["F"]):
        m, va, ep = nm.train_one(toks, summ, tr_i, va_i, bd, bL, 7919 * (s + 1) + 1000 * r + 10 * k, True, max_epochs); preds.append(nm.predict(m, toks, summ, te, bL)); eps.append(ep)
    return dict(r=r, k=k, variant=variant, grid_val_auc=g, chosen=[bd, bL], epochs=eps, test={SUBJ[i]: float(p) for i, p in zip(te, np.mean(preds, 0))},
                seconds=time.time() - t0, test_auc=float(roc_auc_score(Y[te], np.mean(preds, 0))))


def shuffled_segments(rng):
    pi = rng.permutation(N); X2 = np.vstack([XSEG[PID == pi[i]] for i in range(N)]); pid2 = np.concatenate([np.full(int((PID == pi[i]).sum()), i) for i in range(N)])
    return X2, pid2


def null_fold(X, pid, r, k, tr, te, variant, seed_extra=0, max_epochs=20):
    ylab = {s: int(Y[POS[s]]) for s in SUBJ}; tr_p, va_p = tf.split_val(list(SUBJ[tr]), ylab, 100 * r + k)
    tr_i, va_i = np.array([POS[s] for s in tr_p]), np.array([POS[s] for s in va_p])
    toks, summ = tokens(X, pid, tr, variant, no_as=True)
    m, va, ep = nm.train_one(toks, summ, tr_i, va_i, 32, 32, 1000 * r + 10 * k + seed_extra, True, max_epochs)
    return nm.predict(m, toks, summ, te, 32)


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "--dry-run"
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    print(f"device {nm.DEV}; segments {len(XSEG)}; patients {N}", flush=True)
    json.dump(dict(prereg_sha256=sha(os.path.join(HERE, "PRAREGISTRASI_putaran30.md")), code_sha256=sha(__file__)), open(f"{OUT}/meta_nm.json", "w"), indent=1)
    if mode == "--dry-run":
        r, k, tr, te = JOBS[0]; t0 = time.time(); toks, summ = tokens(XSEG, PID, tr, "V4"); c = tr[Y[tr] == 0]
        print("V4 tokens done", f"{time.time() - t0:.0f} s"); ylab = {s: int(Y[POS[s]]) for s in SUBJ}; tr_p, va_p = tf.split_val(list(SUBJ[tr]), ylab, 0)
        tr_i, va_i = np.array([POS[s] for s in tr_p]), np.array([POS[s] for s in va_p])
        for d, L in nm.GRID:
            t1 = time.time(); m, va, ep = nm.train_one(toks, summ, tr_i, va_i, d, L, 0, True, max_epochs=3); print(f"d={d} L={L}: 3 epoch {time.time() - t1:.1f} s (val {va:.3f})", flush=True)
        t1 = time.time(); X2, pid2 = shuffled_segments(np.random.default_rng(30500)); print("pengacakan segmen", f"{time.time() - t1:.1f} s", X2.shape)
        t1 = time.time(); p = null_fold(X2, pid2, 0, 0, tr, te, "V4"); print("one null fold (V4)", f"{time.time() - t1:.1f} s", "AUC", round(roc_auc_score(Y[te], p), 3))
        t1 = time.time(); p = null_fold(X2, pid2, 0, 0, tr, te, "V0"); print("one null fold (V0)", f"{time.time() - t1:.1f} s", "AUC", round(roc_auc_score(Y[te], p), 3))
        sys.exit()
    if mode == "--repro":                                   # reproduce one V0 fold against the JSON of round 16
        r, k, tr, te = JOBS[0]; old = json.load(open(f"{nm.OUT}/r{r}_k{k}.json"))["F"]; res = run_fold_v4(r, k, tr, te, "V0")
        d = np.array([res["test"][s] - old["test"][s] for s in SUBJ[te]]); print("AUC lama", old["test_auc"], "baru", res["test_auc"], "selisih prediksi maks", np.abs(d).max(), "dipilih lama", old["chosen"], "baru", res["chosen"], flush=True)
        json.dump(dict(old_auc=old["test_auc"], new_auc=res["test_auc"], max_abs_diff=float(np.abs(d).max()), old_chosen=old["chosen"], new_chosen=res["chosen"]), open(f"{OUT}/nm_repro_V0_r0k0.json", "w")); sys.exit()
    if mode == "--run":
        for r, k, tr, te in JOBS:
            path = f"{OUT}/nm_v4/r{r}_k{k}.json"
            if os.path.exists(path):
                continue
            t0 = time.time(); res = run_fold_v4(r, k, tr, te); json.dump(res, open(path, "w"))
            print(f"fold r{r}k{k} {(time.time() - t0) / 60:.1f}min: test AUC {res['test_auc']:.3f}; chosen {res['chosen']}", flush=True)
        print("ALL FOLDS DONE", flush=True); sys.exit()
    if mode == "--null":
        nrep = int(sys.argv[2]) if len(sys.argv) > 2 else 20; rows = []; path = f"{OUT}/nm_null_replikasi.csv"
        if os.path.exists(path):
            rows = pd.read_csv(path).to_dict("records")
        done = {x["rep"] for x in rows}
        for rep in range(nrep):
            if rep in done:
                continue
            X2, pid2 = shuffled_segments(np.random.default_rng(30500 + rep)); row = dict(rep=rep); t0 = time.time()
            for variant in ("V0", "V4"):
                P = np.full(N, np.nan)
                for r, k, tr, te in JOBS[:5]:
                    P[te] = null_fold(X2, pid2, r, k, tr, te, variant)
                row[variant] = float(roc_auc_score(Y, P))
            rows.append(row); pd.DataFrame(rows).to_csv(path, index=False); print(f"rep {rep}: V0 {row['V0']:.3f} V4 {row['V4']:.3f} ({time.time() - t0:.0f} s)", flush=True)
        D = pd.DataFrame(rows); print(D[["V0", "V4"]].describe().round(4).to_string()); print("NULL DONE", flush=True)
