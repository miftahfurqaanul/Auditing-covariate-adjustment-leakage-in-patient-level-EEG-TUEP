#!/usr/bin/env python3
"""Round 42 (GPU): Tang plasmode (Bi-LSTM + attention, per-second 506-d features, 199 patients) and NormMamba (segment tokens), condition H1. See PRAREGISTRASI_putaran42.md.
One GPU process at a time. Usage: python round42_gpu_nn.py --check | --run <Tang|NM> [n_rep]   (resumable)"""
import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
os.environ["PYTHONWARNINGS"] = "ignore"; os.environ.pop("CUDA_VISIBLE_DEVICES", None)
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, torch
torch.zeros(1, device='cuda'); assert torch.cuda.device_count() >= 1, 'GPU not visible'       # initialize CUDA BEFORE the old module sets CUDA_VISIBLE_DEVICES="" on import
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import round42_common as C

OUT = os.path.join(HERE, "hasil42"); os.makedirs(OUT, exist_ok=True)
N, D = C.N, C.D; FAMS = {"Tang": dict(gammas=[0.0, 1.0], nrep=20), "NM": dict(gammas=[0.0, 1.0], nrep=20)}
PIPES_TANG = [(t, a) for t in ("raw", "symlog", "rank") for a in ("none", "pat")]; PIPES_NM = [("raw", "N0"), ("raw", "V4"), ("rank", "V4")]


# ------------------------------------------------------------------ Tang
def tang_setup():
    import round34_tang_v4 as T
    X, y_real, subj, ylab_real, owner, S1, cnt = T.load_data(); pats = [s for s in T.SUBJ if s in ylab_real.index]; pi = np.array([T.POS[s] for s in pats]); seqs = {p: np.where(owner == p)[0] for p in pi}
    return T, X, subj, owner, pi, seqs


def gpu_rank_fit(Xg, rows, seed, nsub=100000, chunk=100):
    """Quantiles (1001 points) per feature from a subsample of seconds in rows `rows` (training+validation sequences)."""
    flat = Xg[torch.as_tensor(rows, device=Xg.device)].reshape(-1, Xg.shape[2]); g = torch.Generator(device="cpu").manual_seed(seed); ix = torch.randperm(flat.shape[0], generator=g)[:nsub].to(Xg.device); sub = flat[ix]
    lv = torch.linspace(0, 1, 1001, device=Xg.device); return torch.cat([torch.quantile(sub[:, j:j + chunk].float(), lv, dim=0) for j in range(0, sub.shape[1], chunk)], dim=1)


def gpu_rank_apply(Xg, Qs, chunk=600):
    out = torch.empty_like(Xg); Qt = Qs.T.contiguous()
    for i in range(0, Xg.shape[0], chunk):
        x = Xg[i:i + chunk]; n, s, d = x.shape; xt = x.reshape(-1, d).T.contiguous(); pos = torch.searchsorted(Qt, xt).clamp(1, 1000)
        lo = torch.gather(Qt, 1, pos - 1); hi = torch.gather(Qt, 1, pos); fr = ((xt - lo) / (hi - lo).clamp_min(1e-12)).clamp(0, 1); u = ((pos - 1).float() + fr) / 1000
        out[i:i + chunk] = torch.special.ndtri(u.clamp(1e-3, 1 - 1e-3)).T.reshape(n, s, d)
    return out


def tang_transform(Xg, kind, rows, seed):
    if kind == "raw":
        return Xg
    if kind == "symlog":
        return torch.sign(Xg) * torch.log1p(torch.abs(Xg))
    return gpu_rank_apply(Xg, gpu_rank_fit(Xg, rows, seed))


def tang_unit(rep, gamma, T, X, subj, owner, pi, seqs, dev, Xsh_cache):
    import pandas as pd
    y = C.gen_labels(rep, gamma); auc_cov = float(roc_auc_score(y, C.ETA)); rng = np.random.default_rng(C.kseed("plasmode_perm_tang", rep)); perm = rng.permutation(len(pi))
    if rep not in Xsh_cache:
        Xsh_cache.clear(); idx = np.concatenate([seqs[pi[perm[j]]] for j in range(len(pi))]); own_n = np.concatenate([np.full(len(seqs[pi[perm[j]]]), pi[j]) for j in range(len(pi))]); Xsh_cache[rep] = (torch.from_numpy(X[idx]).to(dev), own_n)
    Xg0, own_n = Xsh_cache[rep]; subj_n = T.SUBJ[own_n]; y_n = y[own_n]; ylab = pd.Series(y[pi], index=T.SUBJ[pi]); skf = StratifiedKFold(5, shuffle=True, random_state=C.kseed("plasmode_cv", rep, gamma)); folds = list(skf.split(np.zeros(len(pi)), y[pi]))
    P = {p: {} for p in PIPES_TANG}; fold_auc = {p: [] for p in PIPES_TANG}
    for fi, (a_, b_) in enumerate(folds):
        te_p = sorted(T.SUBJ[pi[b_]]); trv = sorted(set(ylab.index) - set(te_p)); tr_p, va_p = T.tf.split_val(trv, ylab, C.kseed("plasmode_val", rep, gamma, fi)); pool = np.array([T.POS[s] for s in tr_p + va_p]); rows = np.where(np.isin(subj_n, tr_p + va_p))[0]; seed = C.kseed("plasmode_tang", rep, gamma, fi) % 100000
        cache = {}
        for (t, a) in PIPES_TANG:
            if t not in cache:
                cache[t] = tang_transform(Xg0, t, rows, seed)
            Xt = cache[t]; Off = None
            if a == "pat":
                S = np.zeros((N, Xt.shape[2])); cn = np.zeros(N); sm = Xt.sum(1, dtype=torch.float64).cpu().numpy(); np.add.at(S, own_n, sm); np.add.at(cn, own_n, Xt.shape[1]); M = S / np.maximum(cn, 1)[:, None]
                B = np.linalg.lstsq(D[pool], M[pool], rcond=None)[0]; Off = torch.from_numpy((D[own_n] @ B).astype(np.float32)).to(dev)
            res = T.train_fold(Xt, y_n, subj_n, tr_p, va_p, te_p, ylab, seed, dev, Off=Off)
            P[(t, a)].update(res["test"]); sc = pd.Series(res["test"]); fold_auc[(t, a)].append(float(roc_auc_score(ylab.loc[sc.index].values, sc.values)))
        del cache; torch.cuda.empty_cache()
    rows_out = []
    for (t, a), sc in P.items():
        s = pd.Series(sc); rows_out.append(dict(fam="Tang", rep=rep, cond="H1", gamma=gamma, transform=t, adjust=a, auc=float(roc_auc_score(ylab.loc[s.index].values, s.values)), **{f"fold_auc_{i}": v for i, v in enumerate(fold_auc[(t, a)])}, auc_cov=auc_cov))
    return rows_out


# ------------------------------------------------------------------ NM
def nm_setup():
    import round30_nm as M, round38_nm_paired as N38
    return M, N38


def nm_rank(X, trmask):
    from scipy.stats import norm
    Qs = np.quantile(X[trmask], C.Q, axis=0); out = np.empty(X.shape, float)
    for j in range(X.shape[1]):
        out[:, j] = norm.ppf(np.clip(np.interp(X[:, j], Qs[:, j], C.Q), 1e-3, 1 - 1e-3))
    return out


def nm_unit(rep, gamma, M, N38, dev):
    nm = M.nm; y = C.gen_labels(rep, gamma); auc_cov = float(roc_auc_score(y, C.ETA)); pi = C.gen_perm(rep, "H1"); XSEG, PID = M.XSEG, M.PID
    X2 = np.vstack([XSEG[PID == pi[i]] for i in range(N)]); pid2 = np.concatenate([np.full(int((PID == pi[i]).sum()), i) for i in range(N)]); folds = C.folds_for(rep, gamma, y)
    nm.Y = y.astype(M.Y.dtype if hasattr(M.Y, "dtype") else int); ylab = {s: int(y[i]) for i, s in enumerate(C.SUBJ)}                                           # synthetic labels for train_one and the validation AUC
    P = {p: np.full(N, np.nan) for p in PIPES_NM}
    for fi, (tr, te) in enumerate(folds):
        tr_p, va_p = N38.tf.split_val(list(C.SUBJ[tr]), ylab, C.kseed("plasmode_val", rep, gamma, fi)); tr_i, va_i = np.array([M.POS[s] for s in tr_p]), np.array([M.POS[s] for s in va_p]); seed = C.kseed("plasmode_nm", rep, gamma, fi) % 100000
        for (t, v) in PIPES_NM:
            Xt = nm_rank(X2, np.isin(pid2, tr)) if t == "rank" else X2
            toks, summ = N38.tokens_v(Xt, pid2, tr, "N0") if v == "N0" else M.tokens(Xt, pid2, tr, "V4", no_as=True)
            m, va, ep = nm.train_one(toks, summ, tr_i, va_i, 32, 32, seed, True, 20); P[(t, v)][te] = nm.predict(m, toks, summ, te, 32)
    rows = []
    for (t, v), p in P.items():
        fa = C.fold_aucs(y, p, folds); rows.append(dict(fam="NM", rep=rep, cond="H1", gamma=gamma, transform=t, adjust=v, auc=float(roc_auc_score(y, p)), **{f"fold_auc_{i}": x for i, x in enumerate(fa)}, auc_cov=auc_cov))
    return rows


def checks():
    n = 0; dev = torch.device("cuda")
    x = torch.randn(300, 60, 506, device=dev).exp(); Q = gpu_rank_fit(x, np.arange(200), 1); z = gpu_rank_apply(x, Q)
    ref = C.make_transform("rank", None) if False else None
    zc = z[:200].reshape(-1, 506).double().cpu().numpy(); assert abs(zc.mean()) < 0.05 and abs(zc.std() - 1) < 0.1; n += 1                              # rank on the fit data: ~N(0,1)
    # second implementation (numpy interp) on the same quantile subsample
    Qn = Q.double().cpu().numpy(); xn = x[:5].reshape(-1, 506).double().cpu().numpy(); from scipy.stats import norm
    for j in (0, 100, 505):
        u = np.clip(np.interp(xn[:, j], Qn[:, j], np.linspace(0, 1, 1001)), 1e-3, 1 - 1e-3); assert np.abs(norm.ppf(u) - z[:5].reshape(-1, 506)[:, j].double().cpu().numpy()).max() < 2e-3
    n += 1
    x2 = x.clone(); x2[200:] = x2[200:] * 5 + 3; assert torch.allclose(gpu_rank_apply(x2, gpu_rank_fit(x2, np.arange(200), 1))[:200], z[:200], atol=1e-5); n += 1                       # test data do not change the fit
    s = torch.sign(x) * torch.log1p(torch.abs(x)); assert torch.allclose(tang_transform(x, "symlog", None, 0), s); n += 1
    print(f"{n} GPU checks passed", flush=True)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "--check"; t0 = time.time(); os.environ.pop("CUDA_VISIBLE_DEVICES", None)             # old module (round28_analisis) hides the GPU on import; CUDA not initialized yet
    if mode == "--check":
        checks()
    elif mode == "--run":
        fam = sys.argv[2]; nrep = int(sys.argv[3]) if len(sys.argv) > 3 else FAMS[fam]["nrep"]; path = f"{OUT}/{fam}.csv"; dev = torch.device("cuda"); torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
        done = set(map(tuple, pd.read_csv(path, usecols=["rep", "gamma"]).drop_duplicates().values.tolist())) if os.path.exists(path) else set()
        if fam == "Tang":
            T, X, subj, owner, pi, seqs = tang_setup(); cache = {}
        else:
            M, N38 = nm_setup()
        for rep in range(nrep):
            for g in FAMS[fam]["gammas"]:
                if (rep, g) in done:
                    continue
                rows = tang_unit(rep, g, T, X, subj, owner, pi, seqs, dev, cache) if fam == "Tang" else nm_unit(rep, g, M, N38, dev)
                pd.DataFrame(rows).to_csv(path, mode="a", header=not os.path.exists(path), index=False); print(f"{fam} rep {rep} gamma {g}: {len(rows)} pipelines [{time.time() - t0:.0f}s]", flush=True)
    print(f"done {time.time() - t0:.0f}s", flush=True)
