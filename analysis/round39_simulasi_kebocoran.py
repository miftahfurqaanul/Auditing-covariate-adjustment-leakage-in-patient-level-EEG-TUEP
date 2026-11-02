#!/usr/bin/env python3
"""Round 39: leakage map of confound adjustment on clustered data (windows within patients); controlled simulation with ZERO disease signal. See PRAREGISTRASI_putaran39.md.
Each replicate generates one dataset; all configurations (transform x adjustment) are run on the same data and folds so that the paired differences are valid.
Usage: python round39_simulasi_kebocoran.py --check | --dry-run | --run <stage> <chain cpu|gpu> [n_rep]   (resumable; CPU: CUDA_VISIBLE_DEVICES="" automatically for the cpu chain)"""
import os, sys, json, time, zlib, itertools, warnings
HERE = os.path.dirname(os.path.abspath(__file__)); CHAIN = None
for i, a in enumerate(sys.argv):
    if a == "--run" and len(sys.argv) > i + 2:
        CHAIN = sys.argv[i + 2]
if CHAIN != "gpu":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ["PYTHONWARNINGS"] = "ignore"; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy import stats
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

OUT = os.path.join(HERE, "hasil39"); os.makedirs(OUT, exist_ok=True)
CPU_CLF = ["LRpat", "RF", "ET"]; GPU_CLF = ["XGB", "MLP", "LSTM"]
TREE = {"RF", "ET", "XGB"}                                             # invariance to monotone transforms: the 'none' baseline is the same for all transforms
ERA_REAL1 = np.array([.25, .16, .20, .24, .15]); ERA_REAL0 = np.array([.06, .16, .22, .24, .32])       # era | epilepsy, non-epilepsy (TUEP cohort table)
CONFIGS = [("raw", "none"), ("raw", "win"), ("raw", "pat"), ("symlog", "none"), ("symlog", "win"), ("symlog", "pat"), ("rank", "none"), ("rank", "pat")]           # tahap A (pilot)
def configs_b(dist, imb, chain):
    base = [("raw", "none"), ("raw", "pat"), ("symlog", "none"), ("symlog", "pat"), ("symlogc", "none"), ("symlogc", "pat"), ("rank", "none"), ("rank", "pat")]
    if dist == "lognormal":
        base += [("log", "none"), ("log", "pat")]
    if imb:
        base += [("raw", "win")]
    if chain == "gpu":
        base = [c for c in base if c[0] in ("raw", "symlogc", "rank", "log") or c == ("symlog", "pat")]
    return base


def kseed(*parts):
    return zlib.crc32("|".join(map(str, parts)).encode()) & 0x7fffffff


# ------------------------------------------------------------------ pembangkit data
def make_data(seed, n=200, m=50, p=50, dist="normal", scen="H1", s=1.0, imb=False, sig=1.0, cfrac=0.01, cscale=100.0):
    """Zero disease signal: features do not depend on the label. The label is related to the covariates (age, era) with strength s (s = 1: as in TUEP). H2a: features depend on the covariates."""
    rng = np.random.default_rng(seed); y = np.r_[np.zeros(n // 2), np.ones(n - n // 2)]; rng.shuffle(y)
    age = 55 + 15 * (rng.standard_normal(n) - 0.45 * s * (y - 0.5) * 2 * 0.5); sex = rng.binomial(1, 0.545, n).astype(float)
    base = (ERA_REAL1 + ERA_REAL0) / 2; p1 = (1 - s) * base + s * ERA_REAL1; p0 = (1 - s) * base + s * ERA_REAL0
    era = np.array([rng.choice(5, p=(p1 if yy == 1 else p0)) for yy in y]); keep = [0, 1, 2, 4]                       # acuan = era 3 (2012)
    D = np.column_stack([np.ones(n), (age - age.mean()) / age.std(), sex] + [(era == k).astype(float) for k in keep])
    mu = rng.standard_normal((n, p))
    if scen == "H2a":
        mu = mu + D[:, 1:] @ (0.3 * rng.standard_normal((D.shape[1] - 1, p)))
    mp = np.full(n, m) if not imb else np.clip(np.round(m * np.exp(0.6 * rng.standard_normal(n))), max(5, m // 4), 4 * m).astype(int)
    pid = np.repeat(np.arange(n), mp); N = len(pid); e = rng.standard_normal((N, p))
    if dist == "normal":
        X = mu[pid] + e
    elif dist == "lognormal":
        X = np.exp(sig * (mu[pid] + e))
    elif dist == "t3":
        X = mu[pid] + rng.standard_t(3, (N, p)) / np.sqrt(3)
    elif dist == "contam":
        scale = np.where(rng.random(N) < cfrac, cscale, 1.0); X = mu[pid] + e * scale[:, None]
    elif dist == "chi2":
        X = mu[pid] + (rng.chisquare(3, (N, p)) - 3) / np.sqrt(6)
    else:
        raise ValueError(dist)
    return dict(X=X, pid=pid, y=y, D=D, n=n, p=p)


# ------------------------------------------------------------------ transforms and adjustment (fitted only on training data)
def transform(X, trw, kind):
    if kind == "raw":
        return X
    if kind == "symlog":
        return np.sign(X) * np.log1p(np.abs(X))
    if kind == "symlogc":                                                # scale c = median |x| per feature from the training windows
        c = np.median(np.abs(X[trw]), axis=0) + 1e-12; return np.sign(X) * np.log1p(np.abs(X) / c)
    if kind == "log":                                                    # positive data only (lognormal)
        return np.log(np.maximum(X, 1e-300))
    if kind == "rank":                                                   # rank-based inverse normal from the training quantiles
        q = np.linspace(0, 1, 1001); Q = np.quantile(X[trw], q, axis=0); out = np.empty_like(X)
        for j in range(X.shape[1]):
            u = np.interp(X[:, j], Q[:, j], q); out[:, j] = stats.norm.ppf(np.clip(u, 1e-3, 1 - 1e-3))
        return out
    raise ValueError(kind)


def adjust(T, pid, D, tr_p, trw, kind):
    if kind == "none":
        return T
    if kind == "win":
        B = np.linalg.lstsq(D[pid[trw]], T[trw], rcond=None)[0]
    else:                                                                # pat: OLS on the per-patient mean (unweighted)
        n = D.shape[0]; S = np.zeros((n, T.shape[1])); np.add.at(S, pid, T); M = S / np.bincount(pid, minlength=n)[:, None]; B = np.linalg.lstsq(D[tr_p], M[tr_p], rcond=None)[0]
    return T - D[pid] @ B


# ------------------------------------------------------------------ pengklasifikasi
def pat_scores(prob, pid, te_p):
    return pd.Series(prob).groupby(pid).mean().reindex(te_p).values


def clf_cpu(name, Z, pid, y, tr_p, te_p, seed):
    trw = np.isin(pid, tr_p); tew = np.isin(pid, te_p)
    if name == "LRpat":
        n = len(y); S = np.zeros((n, Z.shape[1])); np.add.at(S, pid, Z); M = S / np.bincount(pid, minlength=n)[:, None]
        m = make_pipeline(StandardScaler(), LogisticRegressionCV(Cs=np.logspace(-3, 1, 9), cv=5, scoring="roc_auc", class_weight="balanced", max_iter=2000)); return m.fit(M[tr_p], y[tr_p]).predict_proba(M[te_p])[:, 1]
    cls = RandomForestClassifier if name == "RF" else ExtraTreesClassifier
    m = cls(n_estimators=100, class_weight="balanced", n_jobs=1, random_state=seed).fit(Z[trw], y[pid[trw]]); return pat_scores(m.predict_proba(Z[tew])[:, 1], pid[tew], te_p)


def clf_gpu(name, Z, pid, y, tr_p, te_p, seed, dev):
    import torch, torch.nn as nn
    trw = np.isin(pid, tr_p); tew = np.isin(pid, te_p); mu, sd = Z[trw].mean(0), Z[trw].std(0) + 1e-8; torch.manual_seed(seed); rng = np.random.default_rng(seed)
    if name == "XGB":
        import xgboost as xgb
        m = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, tree_method="hist", device="cuda", random_state=seed, n_jobs=1, verbosity=0).fit(Z[trw], y[pid[trw]]); return pat_scores(m.predict_proba(Z[tew])[:, 1], pid[tew], te_p)
    Zs = torch.tensor(((Z - mu) / sd).clip(-20, 20), dtype=torch.float32, device=dev); yw = torch.tensor(y[pid], dtype=torch.float32, device=dev); lossf = nn.BCEWithLogitsLoss()
    if name == "MLP":
        net = nn.Sequential(nn.Linear(Z.shape[1], 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1)).to(dev); opt = torch.optim.Adam(net.parameters(), lr=1e-3); idx = np.where(trw)[0]
        for ep in range(15):
            net.train(); perm = rng.permutation(idx)
            for i in range(0, len(perm), 256):
                b = torch.as_tensor(perm[i:i + 256], device=dev); loss = lossf(net(Zs[b]).squeeze(1), yw[b]); opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            pr = torch.sigmoid(net(Zs[torch.as_tensor(np.where(tew)[0], device=dev)]).squeeze(1)).cpu().numpy()
        return pat_scores(pr, pid[tew], te_p)
    # bidirectional LSTM + attention over the sequence of windows per patient
    class Net(nn.Module):
        def __init__(s_, d):
            super().__init__(); s_.l = nn.LSTM(d, 32, batch_first=True, bidirectional=True); s_.a = nn.Linear(64, 1); s_.f = nn.Linear(64, 1)
        def forward(s_, x):
            h, _ = s_.l(x); w = torch.softmax(s_.a(h), 1); return s_.f((w * h).sum(1)).squeeze(1)
    starts = np.r_[0, np.cumsum(np.bincount(pid))[:-1]]; cnt = np.bincount(pid); L = int(min(30, cnt[np.r_[tr_p, te_p]].min())); net = Net(Z.shape[1]).to(dev); opt = torch.optim.Adam(net.parameters(), lr=2e-3); ar = np.arange(L)
    for ep in range(15):
        net.train(); order = rng.permutation(tr_p)
        for i in range(0, len(order), 32):
            ps = order[i:i + 32]; st = np.array([starts[q] + rng.integers(0, cnt[q] - L + 1) for q in ps]); x = Zs[torch.as_tensor(st[:, None] + ar, device=dev)]; yb = torch.tensor(y[ps], dtype=torch.float32, device=dev)
            loss = lossf(net(x), yb); opt.zero_grad(); loss.backward(); opt.step()
    net.eval(); out = []
    with torch.no_grad():
        for q in te_p:
            nch = max(1, cnt[q] // L); st = starts[q] + np.arange(nch) * L; x = Zs[torch.as_tensor(st[:, None] + ar, device=dev)]; out.append(torch.sigmoid(net(x)).mean().item())
    return np.array(out)


# ------------------------------------------------------------------ one replicate
def run_rep(cell, rep, clfs, configs, dev=None):
    cell = dict(dict(sig=1.0, cfrac=0.01, cscale=100.0), **cell); key = (cell["dist"], cell["scen"], cell["n"], cell["m"], cell["p"], cell["imb"], cell["s"])
    if (cell["sig"], cell["cfrac"], cell["cscale"]) != (1.0, 0.01, 100.0):
        key = key + (cell["sig"], cell["cfrac"], cell["cscale"])
    seed = kseed(*key, rep)
    data = make_data(seed, **{k: cell[k] for k in ("n", "m", "p", "dist", "scen", "s", "imb", "sig", "cfrac", "cscale")}); X, pid, y, D = data["X"], data["pid"], data["y"], data["D"]; n = data["n"]
    kurt = float(np.median(stats.kurtosis(X, axis=0))); folds = list(StratifiedKFold(5, shuffle=True, random_state=seed).split(np.zeros(n), y)); res = []
    meta = np.zeros(n)
    for tr, te in folds:
        meta[te] = LogisticRegression(max_iter=500).fit(D[tr][:, 1:], y[tr]).predict_proba(D[te][:, 1:])[:, 1]
    meta_auc = roc_auc_score(y, meta)
    for clf in clfs:
        want = [c for c in configs if not (clf in TREE and c[1] == "none" and c[0] != "raw")]; P = {c: np.zeros(n) for c in want}
        for fi, (tr, te) in enumerate(folds):
            trw = np.isin(pid, tr); cache = {}
            for t, a in want:
                if t not in cache:
                    cache[t] = transform(X, trw, t)
                Z = adjust(cache[t], pid, D, tr, trw, a); fs = kseed(seed, clf, fi)
                P[(t, a)][te] = clf_cpu(clf, Z, pid, y, tr, te, fs) if clf in CPU_CLF else clf_gpu(clf, Z, pid, y, tr, te, fs, dev)
        for c in want:
            res.append(dict(rep=rep, clf=clf, transform=c[0], adjust=c[1], auc=float(roc_auc_score(y, P[c])), kurtosis=kurt, meta_auc=float(meta_auc), **cell))
        if clf in TREE:
            for c in configs:
                if c not in want:
                    res.append(dict(rep=rep, clf=clf, transform=c[0], adjust="none", auc=float(roc_auc_score(y, P[("raw", "none")])), kurtosis=kurt, meta_auc=float(meta_auc), **cell))
    return res


# ------------------------------------------------------------------ definisi tahap
def cells_pilot():
    base = dict(n=200, m=50, p=50, imb=False, s=1.0); out = []
    for d_, sc in itertools.product(["normal", "lognormal", "t3", "contam", "chi2"], ["H1", "H2a"]):
        out.append(dict(base, dist=d_, scen=sc))
    out += [dict(base, dist="normal", scen="H1", s=0.0), dict(base, dist="chi2", scen="H1", s=0.0)]          # control: no label-covariate association
    return out


def cells_b():
    base = dict(n=200, m=50, p=50, imb=False, s=1.0, sig=1.0, cfrac=0.01, cscale=100.0); out = []
    for dist in ("lognormal", "contam"):
        b = dict(base, dist=dist, scen="H1"); out.append(b)                                              # basic cells (identical to the pilot; now 100 replicates)
        out += [dict(b, n=100), dict(b, n=500), dict(b, m=10), dict(b, m=200), dict(b, p=10), dict(b, p=200), dict(b, imb=True), dict(b, s=0.5), dict(b, s=0.0)]
        if dist == "lognormal":
            out += [dict(b, sig=v) for v in (0.25, 0.5, 0.75, 1.5, 2.0)]
        else:
            out += [dict(b, cfrac=f, cscale=c) for f, c in ((0.003, 100.0), (0.01, 10.0), (0.03, 100.0), (0.01, 1000.0))]
        for h in [dict(b, scen="H2a")] + ([dict(b, scen="H2a", sig=v) for v in (0.5, 1.5, 2.0)] if dist == "lognormal" else [dict(b, scen="H2a", cfrac=f, cscale=c) for f, c in ((0.01, 10.0), (0.01, 1000.0))]):
            out.append(h)
    return out


def stage_cells(stage):
    if stage == "pilot":
        return cells_pilot()
    if stage == "B":
        return cells_b()
    if stage == "C":
        b = dict(n=200, m=50, p=50, imb=False, s=1.0, sig=1.0, cfrac=0.01, cscale=100.0, scen="H1"); return [dict(b, dist=d_) for d_ in ("normal", "t3", "chi2")]
    raise ValueError(stage)


def write_rows(path, rows):
    df = pd.DataFrame(rows); df.to_csv(path, mode="a", header=not os.path.exists(path), index=False)


def done_set(path):
    if not os.path.exists(path):
        return set()
    d = pd.read_csv(path, usecols=["dist", "scen", "n", "m", "p", "imb", "s", "sig", "cfrac", "cscale", "rep", "clf"]).drop_duplicates(); return set(map(tuple, d.values.tolist()))


def checks():
    n_ok = 0
    d = make_data(7, n=60, m=20, p=8, dist="lognormal", scen="H2a"); X, pid, y, D = d["X"], d["pid"], d["y"], d["D"]; tr = np.arange(0, 40); trw = np.isin(pid, tr)
    Tt = transform(X, trw, "symlog"); Z = adjust(Tt, pid, D, tr, trw, "win"); assert np.abs(D[pid[trw]].T @ Z[trw]).max() / trw.sum() < 1e-8; n_ok += 1                                          # window-weighted residual orthogonal
    Zp = adjust(Tt, pid, D, tr, trw, "pat"); S = np.zeros((60, 8)); np.add.at(S, pid, Zp); M = S / np.bincount(pid)[:, None]; assert np.abs(D[tr].T @ M[tr]).max() / len(tr) < 1e-8; n_ok += 1    # patient-level residual orthogonal
    X2 = X.copy(); te_w = ~trw; X2[te_w] = X2[te_w] * 5 + 3; Zr = adjust(transform(X, trw, "rank"), pid, D, tr, trw, "pat"); Zr2 = adjust(transform(X2, trw, "rank"), pid, D, tr, trw, "pat"); assert np.allclose(Zr[trw], Zr2[trw]); n_ok += 1   # test data do not change the training features
    y2 = y.copy(); y2[40:] = 1 - y2[40:]; a = clf_cpu("RF", Zr, pid, y, tr, np.arange(40, 60), 1); b = clf_cpu("RF", Zr, pid, y2, tr, np.arange(40, 60), 1); assert np.allclose(a, b); n_ok += 1                       # test labels are not used
    a = clf_cpu("LRpat", Zr, pid, y, tr, np.arange(40, 60), 1); b = clf_cpu("LRpat", Zr, pid, y2, tr, np.arange(40, 60), 1); assert np.allclose(a, b); n_ok += 1
    c = [make_data(3, n=40, m=10, p=5, dist=x) for x in ("normal", "lognormal", "t3", "contam", "chi2")]; k = [float(np.median(stats.kurtosis(z["X"], axis=0))) for z in c]; assert k[1] > 3 and k[3] > 20 and abs(k[0]) < 1; n_ok += 1
    print(f"{n_ok} checks passed; median kurtosis per distribution: " + ", ".join(f"{a}={b:.1f}" for a, b in zip(("normal", "lognormal", "t3", "contam", "chi2"), k)), flush=True)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "--check"; t0 = time.time()
    if mode == "--check":
        checks()
    elif mode == "--dry-run":
        checks(); cell = dict(n=200, m=50, p=50, imb=False, s=1.0, dist="contam", scen="H1")
        for clf in CPU_CLF:
            t1 = time.time(); r = run_rep(cell, 0, [clf], CONFIGS); a = pd.DataFrame(r); print(clf, f"{time.time() - t1:.0f}s", a[["transform", "adjust", "auc"]].round(3).values.tolist()[:4], flush=True)
        if os.environ.get("DRY_GPU") == "1":
            import torch
            dev = torch.device("cuda")
            for clf in GPU_CLF:
                t1 = time.time(); r = run_rep(cell, 0, [clf], [("raw", "none"), ("raw", "pat"), ("symlog", "pat")], dev); print(clf, f"{time.time() - t1:.0f}s", [round(x["auc"], 3) for x in r], flush=True)
    elif mode == "--run":
        stage, chain = sys.argv[2], sys.argv[3]; nrep = int(sys.argv[4]) if len(sys.argv) > 4 else 20; clfs = (CPU_CLF if stage == "pilot" else ["LRpat", "RF"]) if chain == "cpu" else GPU_CLF; path = f"{OUT}/{stage}_{chain}.csv"; done = done_set(path)
        cells = stage_cells(stage); cfgp = CONFIGS if chain == "cpu" else [("raw", "none"), ("raw", "pat"), ("symlog", "none"), ("symlog", "pat")]
        if chain == "gpu":
            import torch
            dev = torch.device("cuda")
        for ci, cell in enumerate(cells):
            if chain == "gpu" and cell["scen"] != "H1":
                continue
            cell = dict(dict(sig=1.0, cfrac=0.01, cscale=100.0), **cell); configs = cfgp if stage == "pilot" else configs_b(cell["dist"], cell["imb"], chain)
            heavy = stage != "pilot" and chain == "cpu" and (cell["m"] == 200 or cell["p"] == 200 or cell["n"] == 500); nrc = max(nrep // 2, 30) if heavy else nrep
            todo = [(r, c) for r in range(nrc) for c in clfs if (cell["dist"], cell["scen"], cell["n"], cell["m"], cell["p"], cell["imb"], cell["s"], cell["sig"], cell["cfrac"], cell["cscale"], r, c) not in done]
            if not todo:
                continue
            if chain == "cpu":
                rows = Parallel(n_jobs=int(os.environ.get("NJOBS", 12)))(delayed(run_rep)(cell, r, [c], configs) for r, c in todo)
                for rr in rows:
                    write_rows(path, rr)
            else:
                for r, c in todo:
                    write_rows(path, run_rep(cell, r, [c], configs, dev))
            print(f"cell {ci + 1}/{len(cells)} {cell['dist']} {cell['scen']} s={cell['s']}: done [{time.time() - t0:.0f}s]", flush=True)
    print(f"done {time.time() - t0:.0f}s")
