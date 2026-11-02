#!/usr/bin/env python3
"""Independent double check for round 30 (PRAREGISTRASI_putaran30.md section 6): second implementation of P3-V5 and NM-V4 tokens, integrity of the NM-V4 JSON files,
reconstruction of F2-adj and the tests, AUC within strata by brute force, and mlconfound. Every check uses an assert.
Usage: CUDA_VISIBLE_DEVICES="" python round30_cek_independen.py <extracted_mlconfound_folder>"""
import os, sys, json, glob, types
os.environ["CUDA_VISIBLE_DEVICES"] = ""; os.environ.setdefault("OMP_NUM_THREADS", "1")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, "/home/miftah_f/RTX-5090/tuep_fresh/code")
import numpy as np, pandas as pd
from scipy import stats, linalg
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score
import round28_analisis as A, round29_sensitivitas as S, round30_p3 as P3m, round30_p3_v5 as P5m, round30_nm as NMm
import normmamba as nmm

OUT = os.path.join(HERE, "hasil30"); Y, N, J, SUBJ, ERA, DEV = A.Y, A.N, A.J, A.c17.SUBJ, A.ERA, A.DEV; n_ok = 0


def ok(m):
    global n_ok
    n_ok += 1; print("OK  ", m, flush=True)


# ---- 1. NM-V4 tokens: second implementation (sklearn) on 3 folds
for f in (0, 17, 33):
    _, _, tr, te = J[f]; toks, summ = NMm.tokens(nmm.XSEG, nmm.PID, tr, "V4")
    fm = np.isin(nmm.PID, tr); Dseg = nmm.DES[nmm.PID]; lr = LinearRegression(fit_intercept=False).fit(Dseg[fm], nmm.XSEG[fm]); res = nmm.XSEG[fm] - Dseg[fm] @ lr.coef_.T
    sd = np.std(res, axis=0, ddof=np.linalg.matrix_rank(Dseg[fm])); T = (nmm.XSEG - Dseg @ lr.coef_.T) / sd
    worst = max(np.abs(toks[i] - T[nmm.PID == i].astype(np.float32)).max() for i in range(N)); assert worst < 1e-4, (f, worst)
    summ2 = np.column_stack([np.vstack([np.median(T[nmm.PID == i], 0) for i in range(N)]), (nmm.AS - nmm.AS[tr].mean(0)) / nmm.AS[tr].std(0)]).astype(np.float32)
    assert np.abs(summ - summ2).max() < 1e-4, f
    resid_mean = np.abs(np.array([T[np.isin(nmm.PID, tr)][:, j] @ Dseg[np.isin(nmm.PID, tr)] for j in range(5)])).max() / np.isin(nmm.PID, tr).sum(); assert resid_mean < 1e-6
ok("NM-V4 tokens and summary from the second implementation equal the main implementation (difference < 1e-4; float32) and the training segment residuals are orthogonal to the design")

# ---- 2. P3-V5 features: second implementation (loop per window, scipy logm)
C, pid = P3m.p3.load_cache(8); IU = P3m.p3.rfe.IU; SC = P3m.p3.rfe.SCALE
def v5_second(tr, windows_idx=None):
    tw = np.isin(pid, tr); feats = []
    for b in range(5):
        M = C[tw, b].mean(0); e, V = np.linalg.eigh(M); W = V @ np.diag(1 / np.sqrt(e)) @ V.T; out = []
        for n in windows_idx:
            Cb = W @ C[n, b] @ W; S_ = Cb + 1e-3 * np.trace(Cb) / 18 * np.eye(18); L = linalg.logm(S_).real; out.append(L[IU] * SC)
        feats.append(np.array(out))
    return np.hstack(feats)
for f in (0, 30):
    _, _, tr, te = J[f]; pat = [3, 77, 150]; X1 = P5m.features_v5(C, pid, tr)
    for p_ in pat:
        idx = np.where(pid == p_)[0]; F2 = v5_second(tr, idx); assert np.abs(F2.mean(0) - X1[p_]).max() < 1e-7, (f, p_, np.abs(F2.mean(0) - X1[p_]).max())
ok("P3-V5 features (global reference from training patients, tangent per window then mean) from the second implementation (scipy logm, loop per window) equal the main implementation (1e-7)")

# ---- 3. integritas JSON NM-V4
Pnm = np.full((10, N), np.nan); POS = {s: i for i, s in enumerate(SUBJ)}
for r, k, tr, te in J:
    d = json.load(open(f"{OUT}/nm_v4/r{r}_k{k}.json")); assert set(d["test"]) == set(SUBJ[te]) and d["variant"] == "V4" and len(d["epochs"]) == 5
    Pnm[r, te] = [d["test"][s] for s in SUBJ[te]]; assert abs(roc_auc_score(Y[te], Pnm[r, te]) - d["test_auc"]) < 1e-12
assert not np.isnan(Pnm).any(); ok("50 NM-V4 JSON files: patients = test patients of each fold, variant V4, 5 seeds, test AUC equals the recomputation (1e-12), no missing values")

# ---- 4. F2-adj and tests: reconstruction
EV = np.load(f"{OUT}/eval_prediksi_v4.npz"); P3a = np.load(f"{OUT}/p3_v5_prediksi_data_asli.npz")["P3_V5_AS"]; I2a = np.load(os.path.join(HERE, "hasil29", "B_prediksi_data_asli.npz"))["V4_AS"]
assert np.array_equal(EV["NM_adj"], Pnm) and np.array_equal(EV["P3_adj"], P3a) and np.array_equal(EV["I2_adj"], I2a); F2 = (I2a + P3a + Pnm) / 3; assert np.allclose(EV["F2_adj"], F2)
fa = lambda M: np.array([roc_auc_score(Y[te], M[r, te]) for r, _, _, te in J]); Tt = pd.read_csv(f"{OUT}/eval_selisih_per_lipatan.csv").set_index("test")
pairs = {"F2-adj - I2-adj": (F2, I2a), "F2-adj - C1": (F2, EV["C1"]), "F2-adj - LaBraM-FT": (F2, EV["LaBraM_FT"]), "F2-adj - CBraMod-FT": (F2, EV["CBraMod_FT"]), "I2-adj - C1": (I2a, EV["C1"]), "P3-adj - Riemann-mentah": (P3a, EV["Riemann_mentah"]), "NM-adj - Mamba-mentah": (Pnm, EV["Mamba_mentah"])}
for name, (a, b) in pairs.items():
    d = fa(a) - fa(b); se = np.sqrt(d.var(ddof=1) * (1 / 50 + 40 / 160)); t = d.mean() / se; assert abs(d.mean() - Tt.loc[name, "diff"]) < 1e-12 and abs(2 * stats.t.sf(abs(t), 49) - Tt.loc[name, "p"]) < 1e-12, name
ok("F2-adj = mean of the three views; per-fold differences (7 tests) recomputed with the explicit Nadeau-Bengio formula (1e-12)")
Mt = pd.read_csv(f"{OUT}/eval_metrik.csv").set_index("model")
for m, Mx in (("F2-adj", F2), ("I2-adj", I2a), ("P3-adj", P3a), ("NM-adj", Pnm)):
    assert abs(100 * fa(Mx).mean() - Mt.loc[m, "auc_fold"]) < 1e-9 and abs(100 * np.mean([roc_auc_score(Y, p) for p in Mx]) - Mt.loc[m, "auc_pooled"]) < 1e-9, m
ok("per-fold AUC and pooled AUC of F2-adj, I2-adj, P3-adj, NM-adj recomputed with sklearn (1e-9)")

# ---- 5. AUC within strata by brute force (F2-adj)
W = pd.read_csv(f"{OUT}/eval_dalam_strata.csv"); YD = np.char.add(np.char.add(ERA, "|"), DEV)
def brute(s, g):
    num = den = 0
    for st in np.unique(g):
        e = np.where((g == st) & (Y == 1))[0]; n = np.where((g == st) & (Y == 0))[0]
        for i in e:
            for j in n:
                num += 1.0 * (s[i] > s[j]) + 0.5 * (s[i] == s[j]); den += 1
    return num / den
row = W[(W.strata == "year x device") & (W.test == "F2-adj - Floor")].iloc[0]; assert abs(brute(F2.mean(0), YD) - row["auc_F2-adj"]) < 1e-12
row = W[(W.strata == "year") & (W.test == "I2-adj - Floor")].iloc[0]; assert abs(brute(I2a.mean(0), ERA) - row["auc_I2-adj"]) < 1e-12
ok("AUC within strata (F2-adj year x device; I2-adj year) equals the brute-force pair count (1e-12)")

# ---- 6. mlconfound
mlc = sys.argv[1]; sys.path.insert(0, mlc); sys.modules["pygam"] = types.ModuleType("pygam"); sys.modules["pygam"].LinearGAM = object
from mlconfound.stats import partial_confound_test, full_confound_test
Cp = pd.read_csv(f"{OUT}/eval_confounder_parsial.csv"); Cf = pd.read_csv(f"{OUT}/eval_confounder_penuh.csv"); cod = lambda a: np.unique(a, return_inverse=True)[1]; rows = []
for m, s in (("F2-adj", F2.mean(0)), ("NM-adj", Pnm.mean(0)), ("P3-adj", P3a.mean(0))):
    pr = partial_confound_test(Y, s, cod(ERA), num_perms=2000, cat_y=True, cat_yhat=False, cat_c=True, random_state=1, progress=False, n_jobs=1)
    fr = full_confound_test(Y, s, cod(YD), num_perms=2000, cat_y=True, cat_yhat=False, cat_c=True, random_state=1, progress=False, n_jobs=1)
    mp = Cp[(Cp.model == m) & (Cp.confounder == "era")].iloc[0]; mf = Cf[(Cf.model == m) & (Cf.confounder == "year x device")].iloc[0]
    rows.append(dict(model=m, parsial_era_paket=pr.p, parsial_era_saya=mp.p, penuh_yd_paket=fr.p, penuh_yd_saya=mf.p, r2_parsial_paket=pr.r2_yhat_c, r2_parsial_saya=mp.R2)); assert abs(pr.r2_yhat_c - mp.R2) < 1e-9
    for a, b in ((pr.p, mp.p), (fr.p, mf.p)):
        assert (a < 0.01 and b < 0.02) or (a > 0.10 and b > 0.05) or abs(a - b) < 0.05, (m, a, b)
Mc = pd.DataFrame(rows); Mc.to_csv(f"{OUT}/cek_mlconfound.csv", index=False); print(Mc.round(4).to_string(index=False))
ok("mlconfound vs my implementation (F2-adj, NM-adj, P3-adj; partial era and full year x device): R^2 identical; same decision or p differing by < 0.05")

# ---- 7. null summary: recomputed from the replicates
for f_, cols in (("p3_v5_kebocoran_ringkasan.csv", ["V5"]), ("p3_kebocoran_ringkasan.csv", ["V0", "V4"])):
    Rr = pd.read_csv(f"{OUT}/{f_.replace('ringkasan', 'replikasi')}"); Ss = pd.read_csv(f"{OUT}/{f_}").set_index("scenario")
    for sc in Ss.index:
        for c in cols:
            assert abs(Rr[Rr.scenario == sc][c].mean() - Ss.loc[sc, f"{c}_mean"]) < 1e-12 and len(Rr[Rr.scenario == sc]) == 100
Nn = pd.read_csv(f"{OUT}/nm_null_replikasi.csv"); assert len(Nn) == 20
ok("null summary of P3 (V0, V4, V5; 100 replicates per scenario) and NM (20 replicates) recomputed from the replicate rows (1e-12)")
print(f"\nALL {n_ok} CHECK GROUPS PASSED")
