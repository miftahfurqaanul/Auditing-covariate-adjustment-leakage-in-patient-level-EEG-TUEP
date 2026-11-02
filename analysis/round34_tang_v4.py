#!/usr/bin/env python3
"""Round 34 (tier 4b, GPU): Tang-F (Tang et al. 2024; order-2 PS -> Bi-LSTM + attention) retrained with and without V4 covariate adjustment. See PRAREGISTRASI_putaran34.md.
Variant F  = Tang-F of round7 exactly (199 patients, 50 canonical folds, seed 2000 + 10r + k, split_val 100r + k, 100 epochs, best epoch = patient validation AUC), only saving the validation scores.
Variant V4 = same, but D_patient @ B is subtracted from every per-second feature (506-d); B = OLS of the per-second features on the I2 design [intercept, age, sex, era] fitted on ALL
            training patients (training + validation, no labels). B is computed exactly from per-patient sums (without duplicating data). The built-in standardization (mu, sd of the training sequences) is kept.
One GPU process at a time. Usage: python round34_tang_v4.py --check | --run | --null [n_replicates=20]   (all resumable; evaluation: round34_eval_tang.py, CPU)"""
import os, sys, json, time, copy, hashlib
HERE = os.path.dirname(os.path.abspath(__file__)); CODE = "/home/miftah_f/RTX-5090/tuep_fresh/code"; sys.path.insert(0, CODE)
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.metrics import roc_auc_score
import round4 as r4, tang_fixed as tf, tang_train as tt, threshold_select as ts

OUT = os.path.join(HERE, "hasil34"); [os.makedirs(f"{OUT}/{v}", exist_ok=True) for v in ("F", "V4")]
SUBJ, D, ERA = r4.SUBJ, r4.DESIGN["I2"], np.asarray(r4.YEAR_G); POS = {s: i for i, s in enumerate(SUBJ)}


def load_data():
    X, meta = tf.load(); y = meta.label.values; subj = meta.subj.values; ylab = meta.groupby("subj").label.first()
    owner = np.array([POS[s] for s in subj]); assert (ylab.loc[[s for s in SUBJ if s in ylab.index]].values == r4.Y[[POS[s] for s in SUBJ if s in ylab.index]]).all()
    S1 = np.zeros((len(SUBJ), 506)); cnt = np.zeros(len(SUBJ)); np.add.at(S1, owner, X.sum(1, dtype=np.float64)); np.add.at(cnt, owner, X.shape[1])
    return X, y, subj, ylab, owner, S1, cnt


def coef(S1, cnt, pool):
    """B (7 x 506) exact: (sum n_p d d')^-1 sum d S1' over the `pool` patients (indices)."""
    Dp = D[pool]; A = (Dp * cnt[pool][:, None]).T @ Dp; return np.linalg.lstsq(A, Dp.T @ S1[pool], rcond=None)[0]


def train_fold(Xg, y, subj, tr_p, va_p, te_p, ylab, seed, dev, Off=None, epochs=tt.EPOCHS):
    """Copy of tang_fixed.train_fold; Off (n_seq x 506, on the GPU) is subtracted from every second before standardization when given. Saves the validation scores of the patients."""
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    tr, va, tst = (np.where(np.isin(subj, s))[0] for s in (tr_p, va_p, te_p))
    raw = (lambda i: Xg[torch.as_tensor(i, device=dev)]) if Off is None else (lambda i: Xg[torch.as_tensor(i, device=dev)] - Off[torch.as_tensor(i, device=dev)][:, None, :])
    mu_t, sd_t = tt.standardise_gpu(raw, tr, dev); get = lambda i: (raw(i) - mu_t) / sd_t
    n1 = y[tr].sum(); n0 = len(tr) - n1; w = torch.tensor([len(tr) / (2 * n0), len(tr) / (2 * n1)], dtype=torch.float32, device=dev)
    m = tt.TangNet().to(dev); opt = torch.optim.Adam(m.parameters(), lr=tt.LR); lossf = nn.CrossEntropyLoss(weight=w)
    yt = torch.from_numpy(y).long().to(dev); best = (-1, None, -1)

    def predict(ids):
        m.eval(); out = []
        with torch.no_grad():
            for i in range(0, len(ids), 512):
                out.append(torch.softmax(m(get(ids[i:i + 512])), 1)[:, 1])
        m.train(); return torch.cat(out).cpu().numpy()
    for ep in range(epochs):
        m.train(); order = rng.permutation(tr)
        for i in range(0, len(order), tt.BATCH):
            b = np.sort(order[i:i + tt.BATCH]); loss = lossf(m(get(b)), yt[b]); opt.zero_grad(); loss.backward(); opt.step()
        ps, pv = tf.patient_scores(predict(va), subj[va]); auc = roc_auc_score(ylab.loc[ps].values, pv)
        if auc > best[0]:
            best = (auc, copy.deepcopy(m.state_dict()), ep + 1)
    m.load_state_dict(best[1]); ps, pv = tf.patient_scores(predict(va), subj[va]); thr, vbac = ts.best_threshold(ylab.loc[ps].values, pv); pt, ptv = tf.patient_scores(predict(tst), subj[tst])
    return dict(val_auc=best[0], best_epoch=best[2], val_threshold=float(thr), val_bac=float(vbac), test={s: float(v) for s, v in zip(pt, ptv)}, val={s: float(v) for s, v in zip(ps, pv)})


def fold_patients(F, r, k, ylab):
    te_p = sorted(F[(F.repeat == r) & (F.fold == k)].subject); trv = sorted(set(ylab.index) - set(te_p)); tr_p, va_p = tf.split_val(trv, ylab, 100 * r + k)
    assert not (set(tr_p) & set(va_p)) and not (set(tr_p + va_p) & set(te_p)) and set(tr_p + va_p + te_p) == set(ylab.index); return te_p, tr_p, va_p


def offsets(S1, cnt, owner, tr_p, va_p, dev):
    pool = np.array([POS[s] for s in tr_p + va_p]); B = coef(S1, cnt, pool); return torch.from_numpy((D[owner] @ B).astype(np.float32)).to(dev)


def checks(X, y, subj, ylab, owner, S1, cnt):
    """CPU, 2 epochs: (1) exact B = direct OLS over all seconds; (2) training residuals orthogonal to the design; (3) changing the test X does not change B; (4) copy of train_fold (Off=None) = tang_fixed.train_fold (CPU, small subset 1 epoch, identical)."""
    F = tf.folds_199(); te_p, tr_p, va_p = fold_patients(F, 0, 0, ylab); pool = np.array([POS[s] for s in tr_p + va_p]); B = coef(S1, cnt, pool); assert np.linalg.matrix_rank(D[pool]) == D.shape[1]
    sel = np.isin(owner, pool[:40]); Xs = X[sel].reshape(-1, 506).astype(np.float64); Ds = np.repeat(D[owner[sel]], X.shape[1], axis=0)
    B2 = np.linalg.lstsq(Ds, Xs, rcond=None)[0]; Bs = coef(S1, cnt, pool[:40]); assert np.abs(Bs - B2).max() < 1e-6 * max(1, np.abs(B2).max()), np.abs(Bs - B2).max()
    sel = np.isin(owner, pool); R = X[sel].reshape(-1, 506).astype(np.float64) - np.repeat((D[owner[sel]] @ B), X.shape[1], axis=0); Dd = np.repeat(D[owner[sel]], X.shape[1], axis=0)
    assert np.abs(Dd.T @ R).max() / len(R) < 1e-5, np.abs(Dd.T @ R).max() / len(R)
    S1b = S1.copy(); ts_idx = np.array([POS[s] for s in te_p]); S1b[ts_idx] = S1b[ts_idx] * 5 + 3; assert np.array_equal(coef(S1b, cnt, pool), B)
    torch.set_num_threads(4); dev = torch.device("cpu"); Xg = torch.from_numpy(X).to(dev); sub = lambda ps, n: [s for c in (0, 1) for s in [q for q in ps if ylab[q] == c][:n]]
    trs, vas, tes = sub(tr_p, 15), sub(va_p, 5), sub(te_p, 5)                                  # small stratified subset so that the CPU check is fast
    a = tf.train_fold(Xg, y, subj, trs, vas, tes, ylab, 2000, dev, epochs=1); b = train_fold(Xg, y, subj, trs, vas, tes, ylab, 2000, dev, epochs=1)
    assert a["best_epoch"] == b["best_epoch"] and all(abs(a["test"][s] - b["test"][s]) < 1e-6 for s in a["test"]), "train_fold copy differs from tang_fixed.train_fold"
    Off = offsets(S1, cnt, owner, trs, vas, dev); c = train_fold(Xg, y, subj, trs, vas, tes, ylab, 2000, dev, Off=Off, epochs=1); assert all(np.isfinite(list(c["test"].values()))) and set(c["val"]) == set(vas)
    print("4 checks passed (exact B, orthogonal residuals, test X does not change B, identical train_fold copy); test AUC (small subset, 1 epoch, code test only): F",
          round(roc_auc_score(ylab.loc[list(a['test'])].values, list(a['test'].values())), 3), "V4", round(roc_auc_score(ylab.loc[list(c['test'])].values, list(c['test'].values())), 3), flush=True)


def run(X, y, subj, ylab, owner, S1, cnt):
    dev = torch.device("cuda"); Xg = torch.from_numpy(X).to(dev); F = tf.folds_199(); t0 = time.time(); old = "/home/miftah_f/RTX-5090/tuep_fresh/data/tang/fixed"
    for r in range(10):
        for k in range(5):
            te_p, tr_p, va_p = fold_patients(F, r, k, ylab); seed = 2000 + 10 * r + k
            for var in ("F", "V4"):
                path = f"{OUT}/{var}/r{r}_f{k}.json"
                if os.path.exists(path):
                    continue
                t1 = time.time(); Off = None if var == "F" else offsets(S1, cnt, owner, tr_p, va_p, dev); res = train_fold(Xg, y, subj, tr_p, va_p, te_p, ylab, seed, dev, Off=Off)
                pt = pd.Series(res["test"]); auc = roc_auc_score(ylab.loc[pt.index].values, pt.values); msg = ""
                if var == "F" and os.path.exists(f"{old}/r{r}_f{k}.json"):
                    o = json.load(open(f"{old}/r{r}_f{k}.json")); msg = f" | round7 AUC {roc_auc_score(ylab.loc[list(o['test'])].values, list(o['test'].values())):.3f}"
                print(f"{var} r{r} f{k}: {time.time() - t1:.0f}s, epoch {res['best_epoch']}, val AUC {res['val_auc']:.3f}, test AUC {auc:.3f}{msg}  [total {time.time() - t0:.0f}s]", flush=True)
                json.dump(dict(res, train_patients=len(tr_p), val_patients=len(va_p), variant=var, seed=seed), open(path, "w"))


def null(X, y, subj, ylab, owner, S1, cnt, n):
    """Reduced leakage test (as for NM): repetition 0 (5 folds), the sequences of each patient are replaced by those of another patient (H1 permuted globally, H2a permuted within era); the new owner carries its own\n    label, age, sex, and era; pooled AUC over 199 test patients. Variants F and V4."""
    dev = torch.device("cuda"); F = tf.folds_199(); pats = [s for s in SUBJ if s in ylab.index]; pi = np.array([POS[s] for s in pats]); seqs = {p: np.where(owner == p)[0] for p in pi}
    path = f"{OUT}/tang_null_replikasi.csv"; done = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame(columns=["rep", "scenario", "variant", "auc"]); t0 = time.time()
    for rep in range(n):
        for scen in ("H1", "H2a"):
            rng = np.random.default_rng((34000 if scen == "H1" else 34300) + rep); perm = np.arange(len(pi))
            if scen == "H1":
                perm = rng.permutation(len(pi))
            else:
                for e in np.unique(ERA[pi]):
                    ii = np.where(ERA[pi] == e)[0]; perm[ii] = rng.permutation(ii)
            idx = np.concatenate([seqs[pi[perm[j]]] for j in range(len(pi))]); own_n = np.concatenate([np.full(len(seqs[pi[perm[j]]]), pi[j]) for j in range(len(pi))])
            subj_n = SUBJ[own_n]; y_n = r4.Y[own_n]; Xg = torch.from_numpy(X[idx]).to(dev); S1n = np.zeros_like(S1); cn = np.zeros_like(cnt); S1n[pi] = S1[pi[perm]]; cn[pi] = cnt[pi[perm]]
            for var in ("F", "V4"):
                if ((done.rep == rep) & (done.scenario == scen) & (done.variant == var)).any():
                    continue
                sc = {}
                for k in range(5):
                    te_p, tr_p, va_p = fold_patients(F, 0, k, ylab); Off = None if var == "F" else offsets(S1n, cn, own_n, tr_p, va_p, dev)
                    sc.update(train_fold(Xg, y_n, subj_n, tr_p, va_p, te_p, ylab, 2000 + k, dev, Off=Off)["test"])
                auc = roc_auc_score(ylab.loc[list(sc)].values, list(sc.values())); done = pd.concat([done, pd.DataFrame([dict(rep=rep, scenario=scen, variant=var, auc=auc)])]); done.to_csv(path, index=False)
                print(f"null rep {rep} {scen} {var}: AUC {auc:.3f} [{time.time() - t0:.0f}s]", flush=True)
            del Xg; torch.cuda.empty_cache()
    print(done.groupby(["scenario", "variant"]).auc.agg(["count", "mean", "std"]).round(4).to_string(), flush=True)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "--check"; t0 = time.time(); X, y, subj, ylab, owner, S1, cnt = load_data()
    print(f"{len(X)} sequences, {len(ylab)} patients; design {D.shape}; cuda {torch.cuda.is_available()}", flush=True)
    if mode == "--check":
        checks(X, y, subj, ylab, owner, S1, cnt); print(f"done {time.time() - t0:.0f}s")
    elif mode == "--run":
        checks(X, y, subj, ylab, owner, S1, cnt); run(X, y, subj, ylab, owner, S1, cnt)
        json.dump(dict(code_sha256=hashlib.sha256(open(__file__, "rb").read()).hexdigest(), seconds=time.time() - t0), open(f"{OUT}/meta_tang.json", "w")); print(f"done {time.time() - t0:.0f}s")
    elif mode == "--null":
        null(X, y, subj, ylab, owner, S1, cnt, int(sys.argv[2]) if len(sys.argv) > 2 else 20)
