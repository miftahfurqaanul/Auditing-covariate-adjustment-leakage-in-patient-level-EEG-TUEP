#!/usr/bin/env python3
"""Putaran 51 (E2 sekunder, GPU): plasmode Tang (Bi-LSTM + atensi) kondisi H1 dengan penyesuaian ash. Salinan round42_gpu_nn.tang_unit; satu-satunya perubahan:
jenis penyesuaian 'ash' (koefisien round51_common.ash_coef) di samping 'pat' (OLS). Benih, permutasi, lipatan, dan pelatihan sama dengan putaran 42. Lihat PRAREGISTRASI_putaran51.md.
Pemakaian: python round51_tang.py --check | --run [n_rep]   (resumable; hasil51/tuep_Tang.csv)"""
import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
os.environ["PYTHONWARNINGS"] = "ignore"; os.environ.pop("CUDA_VISIBLE_DEVICES", None)
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, torch
torch.zeros(1, device="cuda"); assert torch.cuda.device_count() >= 1, "GPU tidak terlihat"
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import round42_gpu_nn as G
import round51_common as K
C, N, D = G.C, G.N, G.D
OUT = os.path.join(HERE, "hasil51"); os.makedirs(OUT, exist_ok=True)
PIPES = [("raw", "none"), ("raw", "pat"), ("rank", "pat"), ("raw", "ash"), ("rank", "ash")]; GAMMAS = [0.0, 1.0]


def unit(rep, gamma, T, X, subj, owner, pi, seqs, dev, Xsh_cache, pipes=PIPES):
    y = C.gen_labels(rep, gamma); auc_cov = float(roc_auc_score(y, C.ETA)); rng = np.random.default_rng(C.kseed("plasmode_perm_tang", rep)); perm = rng.permutation(len(pi))
    if rep not in Xsh_cache:
        Xsh_cache.clear(); idx = np.concatenate([seqs[pi[perm[j]]] for j in range(len(pi))]); own_n = np.concatenate([np.full(len(seqs[pi[perm[j]]]), pi[j]) for j in range(len(pi))]); Xsh_cache[rep] = (torch.from_numpy(X[idx]).to(dev), own_n)
    Xg0, own_n = Xsh_cache[rep]; subj_n = T.SUBJ[own_n]; y_n = y[own_n]; ylab = pd.Series(y[pi], index=T.SUBJ[pi]); folds = list(StratifiedKFold(5, shuffle=True, random_state=C.kseed("plasmode_cv", rep, gamma)).split(np.zeros(len(pi)), y[pi]))
    P = {p: {} for p in pipes}; fold_auc = {p: [] for p in pipes}
    for fi, (a_, b_) in enumerate(folds):
        te_p = sorted(T.SUBJ[pi[b_]]); trv = sorted(set(ylab.index) - set(te_p)); tr_p, va_p = T.tf.split_val(trv, ylab, C.kseed("plasmode_val", rep, gamma, fi)); pool = np.array([T.POS[s] for s in tr_p + va_p])
        rows = np.where(np.isin(subj_n, tr_p + va_p))[0]; seed = C.kseed("plasmode_tang", rep, gamma, fi) % 100000; cache = {}
        for (t, a) in pipes:
            if t not in cache:
                cache[t] = G.tang_transform(Xg0, t, rows, seed)
            Xt = cache[t]; Off = None
            if a in ("pat", "ash"):
                S = np.zeros((N, Xt.shape[2])); cn = np.zeros(N); sm = Xt.sum(1, dtype=torch.float64).cpu().numpy(); np.add.at(S, own_n, sm); np.add.at(cn, own_n, Xt.shape[1]); M = S / np.maximum(cn, 1)[:, None]
                B = np.linalg.lstsq(D[pool], M[pool], rcond=None)[0] if a == "pat" else K.ash_coef(D[pool], M[pool])[0]; Off = torch.from_numpy((D[own_n] @ B).astype(np.float32)).to(dev)
            res = T.train_fold(Xt, y_n, subj_n, tr_p, va_p, te_p, ylab, seed, dev, Off=Off)
            P[(t, a)].update(res["test"]); sc = pd.Series(res["test"]); fold_auc[(t, a)].append(float(roc_auc_score(ylab.loc[sc.index].values, sc.values)))
        del cache; torch.cuda.empty_cache()
    out = []
    for (t, a), sc in P.items():
        s = pd.Series(sc); out.append(dict(cohort="tuep", fam="Tang", rep=rep, cond="H1", gamma=gamma, transform=t, adjust=a, auc=float(roc_auc_score(ylab.loc[s.index].values, s.values)), **{f"fold_auc_{i}": v for i, v in enumerate(fold_auc[(t, a)])}, auc_cov=auc_cov))
    return out


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "--check"; t0 = time.time(); os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    dev = torch.device("cuda"); torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    T, X, subj, owner, pi, seqs = G.tang_setup(); cache = {}
    if mode == "--check":                                                  # reproduksi rep 0, gamma 0, raw-none dan raw-pat terhadap hasil42/Tang.csv (GPU: dicatat; ambang 1e-6)
        new = pd.DataFrame(unit(0, 0.0, T, X, subj, owner, pi, seqs, dev, cache, pipes=[("raw", "none"), ("raw", "pat")])); old = pd.read_csv(os.path.join(HERE, "hasil42", "Tang.csv")); old = old[(old.rep == 0) & (old.gamma == 0.0)]
        m = new.merge(old, on=["transform", "adjust"], suffixes=("", "_42")); FC = [f"fold_auc_{i}" for i in range(5)]; d = max(np.abs(m[c] - m[c + "_42"]).max() for c in ["auc"] + FC)
        print(f"reproduksi Tang rep 0 gamma 0 ({len(m)} pipeline): selisih maks {d:.2e} [{time.time() - t0:.0f}s]", flush=True); assert len(m) == 2 and d < 1e-6
    elif mode == "--run":
        nrep = int(sys.argv[2]) if len(sys.argv) > 2 else 20; path = f"{OUT}/tuep_Tang.csv"
        done = set(map(tuple, pd.read_csv(path, usecols=["rep", "gamma"]).drop_duplicates().values.tolist())) if os.path.exists(path) else set()
        for rep in range(nrep):
            for g in GAMMAS:
                if (rep, g) in done:
                    continue
                pd.DataFrame(unit(rep, g, T, X, subj, owner, pi, seqs, dev, cache)).to_csv(path, mode="a", header=not os.path.exists(path), index=False); print(f"Tang rep {rep} gamma {g} [{time.time() - t0:.0f}s]", flush=True)
    print(f"selesai {time.time() - t0:.0f}s", flush=True)
