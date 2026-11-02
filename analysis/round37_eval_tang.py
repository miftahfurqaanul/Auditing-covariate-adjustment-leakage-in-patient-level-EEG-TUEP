#!/usr/bin/env python3
"""Round 35, evaluation (CPU; V4t): Tang-F vs Tang-Lp (GPU result of round34_tang_v4.py) and comparison with I2-adj@199, C1@199, Tang-F round7. See PRAREGISTRASI_putaran34.md.
Variant +AS (stacking): logistic regression on the VALIDATION patients of each fold (n = 32) with features [score logit, age, sex] applied to the test patients; only training/validation data are used.
Usage: CUDA_VISIBLE_DEVICES="" python round34_eval_tang.py [--smoke]   (--smoke uses the Tang-F JSON of round7 as F and V4, only to test the code)"""
import os, sys, json, glob, hashlib
os.environ["CUDA_VISIBLE_DEVICES"] = ""; os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ["PYTHONWARNINGS"] = "ignore"
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, "/home/miftah_f/RTX-5090/tuep_fresh/code")
import numpy as np, pandas as pd
from scipy import stats
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import round28_analisis as A, round29_sensitivitas as S, round7 as r7

SMOKE = "--smoke" in sys.argv; OLD = "/home/miftah_f/RTX-5090/tuep_fresh/data/tang/fixed"
SRCDIR = {"F": OLD, "V4": OLD} if SMOKE else {"F": os.path.join(HERE, "hasil37", "L"), "V4": os.path.join(HERE, "hasil37", "Lp")}
OUT = os.path.join(HERE, "hasil37" + ("/smoke" if SMOKE else "")); os.makedirs(OUT, exist_ok=True)
Y, N, SUBJ, ERA, DEV, AS, BP, D, ek = A.Y, A.N, A.c17.SUBJ, A.ERA, A.DEV, A.c17.AS, A.BP, A.D_I2, A.ek
YD = np.char.add(np.char.add(ERA, "|"), DEV); J199 = r7.J199; POS = {s: i for i, s in enumerate(SUBJ)}; EXC = [POS[s] for s in r7.tf.EXCLUDE]; KEEP = np.setdiff1d(np.arange(N), EXC)
z7 = np.load("/home/miftah_f/RTX-5090/tuep_fresh/reports/round7_fold_aucs.npz", allow_pickle=True)


def load(var, stack=False):
    P = np.full((10, N), np.nan)
    for r, k, tr, te in J199:
        d = json.load(open(f"{SRCDIR[var]}/r{r}_f{k}.json")); assert set(d["test"]) == set(SUBJ[te]), (var, r, k)
        if not stack:
            P[r, te] = [d["test"][s] for s in SUBJ[te]]
        else:
            val = d["val"] if "val" in d else {s: d["test"][s] for s in list(d["test"])[:32]}            # smoke: fake validation (only tests the code)
            vi = np.array([POS[s] for s in val]); lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
            mu, sd = A.c17.AS[vi, 0].mean(), A.c17.AS[vi, 0].std() or 1.0; f = lambda idx, sc: np.column_stack([lg(sc), (AS[idx, 0] - mu) / sd, AS[idx, 1]])
            m = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000).fit(f(vi, np.array(list(val.values()))), Y[vi]); P[r, te] = m.predict_proba(f(te, np.array([d["test"][s] for s in SUBJ[te]])))[:, 1]
    assert not np.isnan(P[:, KEEP]).any(); return P


def fa(P):
    return np.array([roc_auc_score(Y[te], P[r, te]) for r, _, _, te in J199])


def pooled(P):
    return float(np.mean([roc_auc_score(Y[KEEP], P[r, KEEP]) for r in range(10)]))


def fit_v4(X, tr):
    B = np.linalg.lstsq(D[tr], X[tr], rcond=None)[0]; sd = (X[tr] - D[tr] @ B).std(0, ddof=np.linalg.matrix_rank(D[tr])); return (X - D @ B) / sd


def lr4(X, adj, tr, te):
    F = np.hstack([fit_v4(X, tr) if adj else X, AS]); return ek.nested_lr().fit(F[tr], Y[tr]).predict_proba(F[te])[:, 1]


def nb(a, n1=160, n2=40):
    m = a.mean(); h = stats.t.ppf(0.975, len(a) - 1) * a.std(ddof=1) * np.sqrt(1 / len(a) + n2 / n1); return 100 * m, 100 * (m - h), 100 * (m + h)


if __name__ == "__main__":
    P = {"Tang-L": load("F"), "Tang-Lp": load("V4"), "Tang-L+AS": load("F", True), "Tang-Lp+AS": load("V4", True)}
    for adj, nm in ((True, "I2-adj@199"), (False, "C1@199")):
        out = Parallel(n_jobs=12)(delayed(lr4)(BP, adj, tr, te) for _, _, tr, te in J199); M = np.full((10, N), np.nan)
        for (r, _, _, te), o in zip(J199, out):
            M[r, te] = o
        P[nm] = M
    FA = {k: fa(m) for k, m in P.items()}; FA["Tang-F round7"] = np.asarray(z7["Tang-F@199"], float)
    d = np.abs(FA["Tang-L"] - FA["Tang-F round7"]); print(f"reproduction of Tang-F: per-fold AUC difference from round7 mean {d.mean():.3f}, max {d.max():.3f}; mean AUC {100 * FA['Tang-L'].mean():.2f} vs {100 * FA['Tang-F round7'].mean():.2f}", flush=True)
    # (Tang-L uses symlog features, so it is not expected to equal Tang-F of round7; there is no reproduction assert)
    rows = []
    for k, a in FA.items():
        mn, lo, hi = nb(a); rows.append(dict(model=k, auc_fold=mn, lo=lo, hi=hi, sd=100 * a.std(ddof=1), auc_pooled=100 * pooled(P[k]) if k in P else np.nan,
                                              strata_year_x_device=100 * A.r9.strat_auc(Y[KEEP], np.nanmean(P[k], 0)[KEEP], YD[KEEP]) if k in P else np.nan))
    M1 = pd.DataFrame(rows); M1.to_csv(f"{OUT}/tang_log_metrik.csv", index=False); print(M1.round(2).to_string(index=False), flush=True)
    pairs = [("Tang-Lp", "Tang-L", "primary"), ("I2-adj@199", "Tang-Lp", "primary"), ("Tang-Lp+AS", "Tang-L+AS", "primary"),
             ("I2-adj@199", "Tang-L", "descriptive"), ("I2-adj@199", "Tang-Lp+AS", "descriptive"), ("Tang-Lp", "C1@199", "descriptive"), ("Tang-Lp+AS", "C1@199", "descriptive"),
             ("Tang-L+AS", "Tang-L", "descriptive"), ("Tang-L", "Tang-F round7", "descriptive")]
    rows = []
    for a, b, fam in pairs:
        dd, lo, hi, p = S.nb_diff(FA[a], FA[b]); rows.append(dict(test=f"{a} - {b}", diff=dd, lo=lo, hi=hi, p=p, folds_better=int((FA[a] > FA[b]).sum()), family=fam))
    T = pd.DataFrame(rows); pm = T.family == "primary"; T["p_holm"] = np.nan; T.loc[pm, "p_holm"] = A.holm(T.loc[pm, "p"].values); T["q_bh"] = stats.false_discovery_control(np.nan_to_num(T.p.values, nan=1.0))
    T.to_csv(f"{OUT}/tang_log_selisih.csv", index=False); print(T.round(4).to_string(index=False), flush=True)
    np.savez(f"{OUT}/tang_log_prediksi.npz", **{k.replace("+", "_AS_").replace("-", "_").replace("@", "_"): v for k, v in P.items()})
    nf = os.path.join(HERE, "hasil37", "__tidak_ada__.csv")
    if os.path.exists(nf) and not SMOKE:
        R = pd.read_csv(nf); g = R.groupby(["scenario", "variant"]).auc.agg(n="count", mean="mean", sd="std").reset_index(); g["se"] = g.sd / np.sqrt(g.n); g["p95_rata2"] = g["mean"] + 1.645 * g.se
        g["keputusan"] = np.where((g["mean"] <= 0.52) & (g.p95_rata2 <= 0.54), "tidak ada bukti kebocoran", np.where(g["mean"] > 0.55, "kebocoran", "pemanfaatan kecil/tidak pasti")); g.to_csv(f"{OUT}/tang_log_null_ringkasan.csv", index=False); print(g.round(4).to_string(index=False), flush=True)
    json.dump(dict(smoke=SMOKE, code_sha256=hashlib.sha256(open(__file__, "rb").read()).hexdigest()), open(f"{OUT}/meta_eval_tang_log.json", "w")); print("done")
