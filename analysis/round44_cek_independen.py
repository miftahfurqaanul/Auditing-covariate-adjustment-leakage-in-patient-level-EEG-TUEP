#!/usr/bin/env python3
"""Independent check for round 44 (PRAREGISTRASI_putaran44.md section 5). Second implementation and asserts. CPU only. Usage: python round44_cek_independen.py"""
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""; os.environ.setdefault("OMP_NUM_THREADS", "1")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, "/home/miftah_f/RTX-5090/tuep_fresh/code"); OUT = os.path.join(HERE, "hasil44")
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
import statsmodels.api as sm
from scipy import stats
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score
import round44_cpu as F
import round42_common as C42
import round28_analisis as A
n_ok = 0; Y, J, SUBJ, N = F.Y, F.J, F.SUBJ, F.N


def ok(m):
    global n_ok
    n_ok += 1; print("OK  ", m, flush=True)


# 1. reproduksi (H44-0)
Z = np.load(f"{OUT}/A_prediksi.npz"); I2ADJ = np.load(os.path.join(HERE, "hasil29/B_prediksi_data_asli.npz"))["V4_AS"]; P3ADJ = np.load(os.path.join(HERE, "hasil30/p3_v5_prediksi_data_asli.npz"))["P3_V5_AS"]
assert np.abs(A.c17.fold_aucs(J, Z["I2__asis"]) - A.c17.fold_aucs(J, I2ADJ)).max() < 1e-9 and np.abs(A.c17.fold_aucs(J, Z["P3__asis"]) - A.c17.fold_aucs(J, P3ADJ)).max() < 1e-9
ok("H44-0: variant `asis` reproduces I2-adj (hasil29) and P3-adj (hasil30) on the per-fold AUC (1e-9)")

# 2. post_features: second implementation (interp1d + np.percentile, LinearRegression, design D2 reassembled)
AGE = np.asarray(A.AGE, float); D2b = np.column_stack([A.D_I2, ((AGE - AGE.mean()) / AGE.std()) ** 2]); assert np.allclose(D2b, F.D2) and D2b.shape == (200, 8)
r, k, tr, te = J[3]; X = A.BP.astype(float)
for v, (rank, dn, uas) in F.VARS.items():
    f1 = F.post_features(X, tr, rank, dn, uas); Zc = X.copy()
    if rank:
        zz = np.empty(Zc.shape)
        for j in range(Zc.shape[1]):
            q = np.percentile(Zc[tr, j], np.linspace(0, 100, 1001)); fq = interp1d(q, np.linspace(0, 1, 1001), bounds_error=False, fill_value=(0.0, 1.0)); zz[:, j] = stats.norm.ppf(np.clip(fq(Zc[:, j]), 1e-3, 1 - 1e-3))
        Zc = zz
    Dm = D2b if dn == "D2" else A.D_I2; B = LinearRegression(fit_intercept=False).fit(Dm[tr], Zc[tr]).coef_.T; R = Zc - Dm @ B; R = R / R[tr].std(0, ddof=np.linalg.matrix_rank(Dm[tr]))
    R = np.hstack([R, F.AS]) if uas else R; assert np.abs(R - f1).max() < 1e-6, (v, np.abs(R - f1).max())
ok("post_features (rank-normal, V4 on D1/D2, AS) from the second implementation (np.percentile + interp1d, LinearRegression) equals the main implementation for 5 variants (1e-6)")

# 3. built-in perturbation
F.checks(); ok("built-in checks of round44_cpu (test data/labels, residual orthogonality D1/D2, IPW weights)")

# 4. IPW: second implementation (statsmodels Logit)
tr_ = J[10][2]; m = sm.Logit(Y[tr_], F.D1[tr_]).fit(disp=0, maxiter=200); p1 = 1 / (1 + np.exp(-(F.D1 @ m.params))); w = np.where(Y == 1, 1 / p1, 1 / (1 - p1)); lo, hi = np.percentile(w[tr_], [5, 95]); w = np.clip(w, lo, hi); w = w / w[tr_].mean()
assert np.abs(w[tr_] - F.ipw_weights(Y, tr_)[tr_]).max() < 1e-2 and abs(w[tr_].mean() - 1) < 1e-12
ok("IPW weights (statsmodels Logit, clipping at the 5-95 training percentiles, mean 1) equal the main implementation (tolerance 1e-2: sklearn lbfgs stops at the default tol; measured difference 3e-3, log-likelihood difference 2e-5)")

# 5. F2-Clean assembly and model-level statistics
T = pd.read_csv(f"{OUT}/ringkas_modelos.csv").set_index("model"); I2c, P3c = Z["I2__rank-noAS-D2"], Z["P3__rank-noAS-D2"]; f2nn = (I2c + P3c) / 2
fa = np.array([roc_auc_score(Y[te], f2nn[r, te]) for r, k, tr, te in J]); fb = np.array([roc_auc_score(Y[te], np.load(os.path.join(HERE, "hasil30/eval_prediksi_v4.npz"))["F2_adj"][r, te]) for r, k, tr, te in J]); d = fa - fb; se = np.sqrt(d.var(ddof=1) * (1 / 50 + 40 / 160))
assert abs(100 * fa.mean() - T.loc["F2-Clean-noNM", "auc_fold"]) < 1e-9 and abs(2 * stats.t.sf(abs(d.mean() / se), 49) - T.loc["F2-Clean-noNM", "p_F2-adj"]) < 1e-12
s = f2nn.mean(0); assert abs(np.corrcoef(s, AGE)[0, 1] ** 2 - T.loc["F2-Clean-noNM", "r2_age"]) < 1e-12
ok("F2-Clean-noNM reassembled from I2 and P3: per-fold AUC, difference vs F2-adj (Nadeau-Bengio p) and age r^2 recomputed (1e-9 / 1e-12)")

# 6. strata: brute-force AUC_k and first bootstrap
import round9 as r9, round41_strata_kaliper as S41
z = np.load(f"{OUT}/bruto_strata.npz", allow_pickle=True); names = list(z["names"]); g = r9.STRATA["year x device"]; j = names.index("F2-Clean-noNM")
def brute(y, sc, gg, age, k):
    num = den = 0
    for i in range(len(y)):
        if y[i] != 1:
            continue
        for q in range(len(y)):
            if y[q] == 0 and gg[i] == gg[q] and abs(age[i] - age[q]) <= k:
                num += 1.0 * (sc[i] > sc[q]) + 0.5 * (sc[i] == sc[q]); den += 1
    return num / den, den
for i, kk in enumerate(S41.KS):
    a, dn = brute(Y, f2nn.mean(0), g, AGE, kk); assert dn == z["npairs"][i] and abs(a - z["point"][i, j]) < 1e-12
rng = np.random.default_rng(44200); groups = [np.where(g == s_)[0] for s_ in np.unique(g)]; idx = np.concatenate([rng.choice(v, len(v)) for v in groups])
for i, kk in enumerate(S41.KS):
    a, dn = brute(Y[idx], f2nn.mean(0)[idx], g[idx], AGE[idx], kk); assert abs(a - z["boot"][0, i, j]) < 1e-12
ok("AUC_k within year x device strata (k = 5, 10, 15, infinity) and the first bootstrap for F2-Clean-noNM from the brute-force loop equal the stored result (1e-12)")

# 7. 30 windows: number of selected windows and median BP30
if os.path.exists(f"{OUT}/C_prediksi.npz"):
    import eegbench_fixed_eval as ebf
    w_ = np.load("/home/miftah_f/RTX-5090/tuep_fresh/data/features_bandpower_windows.npz", allow_pickle=True); Xw = w_["X"]; e = np.load("/home/miftah_f/RTX-5090/tuep_fresh/data/emb_labram_windows.npz", allow_pickle=True)
    first = ebf.first[e["rec"]]; subj = np.asarray(ebf.subj_r)[e["rec"]]; sub = F.subset_positions(); tot = 0; short = 0
    for p in range(N):
        idxw = np.where(first & (subj == p if False else subj == SUBJ[p]))[0]; pos = sub(len(idxw), p); assert len(np.unique(pos)) == min(30, len(idxw)) and pos.max() < len(idxw); tot += len(pos); short += len(idxw) < 30
    assert short == 2 and tot == 30 * 198 + sum(min(30, len(np.where(first & (subj == SUBJ[p]))[0])) for p in range(N) if len(np.where(first & (subj == SUBJ[p]))[0]) < 30)
    ok(f"30 windows: each patient has min(30, n) unique windows (total {tot}); exactly 2 patients with < 30 windows")

# 8. IPW-RF plasmode: one unit reproduced from scratch
fp = f"{OUT}/D_ipw_rf_plasmode.csv"
if os.path.exists(fp):
    D = pd.read_csv(fp); assert not D.duplicated(["rep", "cond", "gamma"]).any() and D[[f"fold_auc_{i}" for i in range(5)]].notna().all().all()
    row = D[(D.rep == 2) & (D.cond == "H1") & (D.gamma == 1.0)]
    if len(row):
        u = F.unit_ipw_rf(2, "H1", 1.0); assert abs(u["auc"] - row["auc"].iloc[0]) < 1e-12 and max(abs(u[f"fold_auc_{i}"] - row[f"fold_auc_{i}"].iloc[0]) for i in range(5)) < 1e-12
    ok(f"IPW-RF plasmode: {len(D)} complete units without duplicates/NaN; one unit (rep 2, H1, gamma 1) reproduced (1e-12)")

# 9. summary from raw data
for mname in ("I2__rank-noAS-D2", "P3__noAS", "F2-Clean-noNM-30"):
    P = Z[mname] if mname in Z.files else ((np.load(f"{OUT}/C_prediksi.npz")["I2-30"] + np.load(f"{OUT}/C_prediksi.npz")["P3-30"]) / 2 if os.path.exists(f"{OUT}/C_prediksi.npz") else None)
    if P is None:
        continue
    fa = np.array([roc_auc_score(Y[te], P[r, te]) for r, k, tr, te in J]); assert abs(100 * fa.mean() - T.loc[mname, "auc_fold"]) < 1e-9
ok("per-fold AUC of three rows of ringkas_modelos.csv recomputed with sklearn (1e-9)")

# 10. NM without AS (D2) and complete F2-Clean
import json as _json
if os.path.exists(f"{OUT}/E_nm_prediksi.npz"):
    NMc = np.load(f"{OUT}/E_nm_prediksi.npz")["NM_clean"]; POSs = {s_: i for i, s_ in enumerate(SUBJ)}
    for r, k, tr, te in J:
        d = _json.load(open(f"{OUT}/nm_clean/r{r}_k{k}.json")); assert d["variant"] == "V4-noAS-D2" and set(d["test"]) == set(SUBJ[te]) and len(d["epochs"]) == 5
        pr = np.array([d["test"][s_] for s_ in SUBJ[te]]); assert np.allclose(NMc[r, te], pr, atol=0, rtol=0) and abs(roc_auc_score(Y[te], pr) - d["test_auc"]) < 1e-12
    ok("NM without AS (D2): 50 JSON files (variant V4-noAS-D2; patients = test patients of each fold; 5 seeds; test AUC recomputed 1e-12) equal the matrix E_nm_prediksi")
    FCl = (Z["I2__rank-noAS-D2"] + Z["P3__rank-noAS-D2"] + NMc) / 3; fa = np.array([roc_auc_score(Y[te], FCl[r, te]) for r, k, tr, te in J]); F2a = np.load(os.path.join(HERE, "hasil30/eval_prediksi_v4.npz")); res = {}
    for nm_, ref in (("F2-adj", F2a["F2_adj"]), ("C1", F2a["C1"]), ("Floor", F2a["Floor"])):
        fb = np.array([roc_auc_score(Y[te], ref[r, te]) for r, k, tr, te in J]); d = fa - fb; se = np.sqrt(d.var(ddof=1) * (1 / 50 + 40 / 160)); res[nm_] = (d.mean(), 2 * stats.t.sf(abs(d.mean() / se), 49))
    Pm = pd.read_csv(f"{OUT}/ringkas_primer.csv")
    for i_, nm_ in enumerate(("F2-adj", "C1", "Floor")):
        assert abs(Pm.iloc[i_].nilai - res[nm_][0]) < 1e-12 and abs(Pm.iloc[i_].p - res[nm_][1]) < 1e-12, nm_
    s_ = FCl.mean(0); assert abs(np.corrcoef(s_, AGE)[0, 1] ** 2 - T.loc["F2-Clean", "r2_age"]) < 1e-12 and abs(100 * fa.mean() - T.loc["F2-Clean", "auc_fold"]) < 1e-9
    zz = np.load(f"{OUT}/bruto_strata.npz", allow_pickle=True); nmz = list(zz["names"]); a_, dn = brute(Y, s_, g, AGE, 10.0); assert abs(a_ - zz["point"][1, nmz.index("F2-Clean")]) < 1e-12 and dn == zz["npairs"][1]
    ok("complete F2-Clean reassembled (I2, P3, NM): per-fold AUC, T1-T3 (difference and Nadeau-Bengio p, 1e-12), age r^2, and T4 (AUC_10 with a brute-force loop, 1e-12) equal the summary")
print(f"\nALL {n_ok} CHECK GROUPS PASSED", flush=True)
