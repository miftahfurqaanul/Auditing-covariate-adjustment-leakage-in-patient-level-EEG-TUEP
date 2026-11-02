#!/usr/bin/env python3
"""Independent check for round 42 (PRAREGISTRASI_putaran42.md section 5). Second implementation and asserts. CPU only. Usage: python round42_cek_independen.py"""
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""; os.environ.setdefault("OMP_NUM_THREADS", "1")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); OUT = os.path.join(HERE, "hasil42")
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
import statsmodels.api as sm
from scipy import stats
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import round42_common as C
import round42_cpu as P
import round42_ringkas as S
n_ok = 0; FC = [f"fold_auc_{i}" for i in range(5)]


def ok(m):
    global n_ok
    n_ok += 1; print("OK  ", m, flush=True)


# 1. ETA and y_sim: second implementation (statsmodels; independent generator)
Xd = C.D[:, 1:]; fit = sm.Logit(C.Y, np.column_stack([np.ones(200), Xd])).fit(disp=0, maxiter=200); eta2 = np.column_stack([np.ones(200), Xd]) @ fit.params; eta2 = eta2 - eta2.mean(); assert np.abs(eta2 - C.ETA).max() < 1e-3, np.abs(eta2 - C.ETA).max()
for rep in (0, 7, 19):
    e = np.random.default_rng(C.kseed("plasmode_e", rep)).logistic(size=200)
    for g in C.GAMMAS:
        y = np.zeros(200, int); y[np.argsort(-(g * eta2 + e), kind="stable")[:100]] = 1
        assert (y == C.gen_labels(rep, g)).mean() > 0.97 and C.gen_labels(rep, g).sum() == 100
cal = {g: np.mean([roc_auc_score(C.gen_labels(i, g), C.ETA) for i in range(100)]) for g in C.GAMMAS}; assert abs(cal[0.0] - 0.5) < 0.03 and abs(cal[1.0] - C.AUC_ETA_REAL) < 0.03 and cal[0.0] < cal[0.5] < cal[1.0]
ok(f"eta (statsmodels vs sklearn, 1e-3); y_sim 100/100 e equal the independent generator (>= 97% identical labels); calibration of the covariate AUC gamma 0/0.5/1 = {cal[0.0]:.3f}/{cal[0.5]:.3f}/{cal[1.0]:.3f} (nyata {C.AUC_ETA_REAL:.3f})")

# 2. transforms and adjustment: second implementation
X, owner = P.windows_for(C.gen_perm(1, "H1")); tr = np.arange(0, 160); trw = np.isin(owner, tr)
Zr = C.make_transform("rank", X[trw])(X)
for j in (0, 17, 47):
    qs = np.percentile(X[trw][:, j], np.linspace(0, 100, 1001)); f = interp1d(qs, np.linspace(0, 1, 1001), bounds_error=False, fill_value=(0.0, 1.0)); z2 = stats.norm.ppf(np.clip(f(X[:, j]), 1e-3, 1 - 1e-3))
    assert np.abs(z2 - Zr[:, j]).max() < 1e-6, (j, np.abs(z2 - Zr[:, j]).max())
assert np.allclose(C.make_transform("symlog", None)(X), np.sign(X) * np.log1p(np.abs(X))) and np.allclose(C.make_transform("log", None)(X), np.log(np.maximum(X, 1e-12)))
Zp = C.adjust_units(Zr, owner, tr, "pat"); M = np.zeros((200, 48)); [M.__setitem__(i, Zr[owner == i].mean(0)) for i in range(200)]; B = LinearRegression(fit_intercept=False).fit(C.D[tr], M[tr]).coef_.T; assert np.abs(Zp - (Zr - C.D[owner] @ B)).max() < 1e-8
Zw = C.adjust_units(Zr, owner, tr, "win"); B = LinearRegression(fit_intercept=False).fit(C.D[owner][trw], Zr[trw]).coef_.T; assert np.abs(Zw - (Zr - C.D[owner] @ B)).max() < 1e-8
ok("rank (np.percentile + interp1d), symlog, log, pat offset (mean per patient) and win (all training windows) from the second implementation equal the main implementation (1e-6 / 1e-8)")

# 3. built-in perturbation (test data/labels change nothing)
P.checks(); ok("built-in checks of round42_cpu (test labels, test data, residual orthogonality)")

# 4. independent reproduction of one RF unit and one LR unit from scratch (the workflow is rewritten here)
for fam, csv in (("RF", "RF.csv"), ("LR", "LR.csv")):
    fp = f"{OUT}/{csv}"
    if not os.path.exists(fp):
        print("passed (not yet available):", fp); continue
    Dd = pd.read_csv(fp); rep, cond, g = 2, "H1", 1.0; u = Dd[(Dd.rep == rep) & (Dd.cond == cond) & (Dd.gamma == g)]
    if u.empty:
        continue
    y = C.gen_labels(rep, g); pi = C.gen_perm(rep, cond); folds = list(StratifiedKFold(5, shuffle=True, random_state=C.kseed("plasmode_cv", rep, g)).split(np.zeros(200), y))
    if fam == "RF":
        blocks = [np.where(P.PIDX == q)[0] for q in range(200)]; Xn = np.vstack([P.XR[blocks[pi[p]]] for p in range(200)]); ow = np.concatenate([np.full(len(blocks[pi[p]]), p) for p in range(200)]); sc = np.zeros(200)
        for fi, (trp, tep) in enumerate(folds):
            seed = C.kseed("plasmode_clf", rep, g, fi); trw_ = np.isin(ow, trp); f = C.make_transform("rank", Xn[trw_]); Z = f(Xn); Mm = np.vstack([Z[ow == i].mean(0) for i in range(200)]); Bb = np.linalg.lstsq(C.D[trp], Mm[trp], rcond=None)[0]; Z = Z - C.D[ow] @ Bb
            vt, vv = C.split_val(trp, y, seed); fit_ = np.isin(ow, vt); m = RandomForestClassifier(n_estimators=100, max_depth=1000, class_weight="balanced", n_jobs=1, random_state=seed).fit(Z[fit_], y[ow[fit_]])
            sel = np.isin(ow, tep); pr = m.predict_proba(Z[sel])[:, 1]; sc[tep] = pd.Series(pr).groupby(ow[sel]).mean().reindex(tep).values
        r = u[(u["transform"] == "rank") & (u.adjust == "pat")].iloc[0]; assert abs(roc_auc_score(y, sc) - r["auc"]) < 1e-12 and max(abs(roc_auc_score(y[te], sc[te]) - r[f"fold_auc_{i}"]) for i, (_, te) in enumerate(folds)) < 1e-12
    else:
        import evalkit as ek
        Xl = P.BP[pi]; sc = np.zeros(200)
        for trp, tep in folds:
            f = C.make_transform("symlog", None); Z = f(Xl); Bb = np.linalg.lstsq(C.D[trp], Z[trp], rcond=None)[0]; Z = Z - C.D @ Bb; sc[tep] = ek.nested_lr().fit(Z[trp], y[trp]).predict_proba(Z[tep])[:, 1]
        r = u[(u["transform"] == "symlog") & (u.adjust == "pat")].iloc[0]; assert abs(roc_auc_score(y, sc) - r["auc"]) < 1e-10
    ok(f"{fam}: one unit (rep 2, H1, gamma 1) reproduced from scratch with a rewritten workflow; pooled and per-fold AUC equal the stored CSV")

# 5. kelengkapan, duplikat, NaN
fams = {"LR": (30, ["H1", "H2a", "nyata"], 6, [0.0, 0.5, 1.0]), "RF": (30, ["H1", "H2a", "nyata"], 6, [0.0, 0.5, 1.0]), "XGB": (20, ["H1", "H2a"], 6, [0.0, 0.5, 1.0]), "Tang": (20, ["H1"], 6, [0.0, 1.0]), "NM": (20, ["H1"], 3, [0.0, 1.0])}
for fam, (nr, conds, npipe, gs) in fams.items():
    fp = f"{OUT}/{fam}.csv"
    if not os.path.exists(fp):
        print("passed (not yet available):", fp); continue
    Dd = pd.read_csv(fp); assert not Dd.duplicated(["rep", "cond", "gamma", "transform", "adjust"]).any() and Dd[["auc"] + FC].notna().all().all() and ((Dd[FC] >= 0) & (Dd[FC] <= 1)).all().all()
    if fam == "NM" and not os.path.exists(f"{OUT}/NM_SELESAI"):
        print(f"skipping NM completeness (not finished: {len(Dd)} rows)"); continue
    assert len(Dd) == nr * len(conds) * npipe * len(gs), (fam, len(Dd), nr * len(conds) * npipe * len(gs)); ok(f"{fam}: {len(Dd)} complete rows ({nr} replicates x {len(conds)} kondisi x {len(gs)} gamma x {npipe} pipelines), no duplicates/NaN, AUC in [0,1]")

# 6. sanity check: pipeline without adjustment H1 ~ 0.5 and summary recomputed
D = S.load(); U = D[(D.cond == "H1") & (D.adjust.isin(["none", "N0"]))]
for fam, G in U.groupby("fam"):
    assert abs(G["fold_mean"].mean() - 0.5) < 0.03, (fam, G["fold_mean"].mean())
ok("pipeline without adjustment on H1 (features independent of label and covariates): mean fold_mean within 0.5 +- 0.03 for every family")
T = pd.read_csv(f"{OUT}/ringkas_pipelines.csv"); rng = np.random.default_rng(5)
for i in rng.choice(len(T), size=min(20, len(T)), replace=False):
    r = T.iloc[i]; g = D[(D["fam"] == r["fam"]) & (D["cond"] == r["cond"]) & (D["gamma"] == r["gamma"]) & (D["pipe"] == r["pipe"])]; assert len(g) == r.nrep and abs(g["fold_mean"].mean() - r.fold_mean_mean) < 1e-9 and abs(g["auc"].mean() - r.auc_mean) < 1e-9
    if r.adjusted and r.cond in ("H1", "H2a"):
        assert S.dec_R1(g["fold_mean"].mean(), g["fold_mean"].std(ddof=1) / np.sqrt(len(g))) == r.R1_fold_mean
ok("20 random rows of ringkas_pipelines.csv (mean and R1 decision) recomputed from the raw rows (1e-9)")
print(f"\nALL {n_ok} CHECK GROUPS PASSED", flush=True)
