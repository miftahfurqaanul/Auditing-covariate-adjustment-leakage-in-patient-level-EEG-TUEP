#!/usr/bin/env python3
"""Independent double check for round 29 (PRAREGISTRASI_putaran29.md section 4). A second implementation of V1-V4 with explicit loops and
sklearn.LinearRegression, reconstruction of statistics by other means, and the mlconfound package. Every check uses an assert.
Usage: python round29_cek_independen.py <extracted_mlconfound_folder>"""
import os, sys, types
os.environ["CUDA_VISIBLE_DEVICES"] = ""; os.environ.setdefault("OMP_NUM_THREADS", "1")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score
import round29_sensitivitas as S
A = S.A; OUT = S.OUT; Y, N, J, ERA, DEV, BP, AS, ek, c17 = A.Y, A.N, A.J, A.ERA, A.DEV, A.BP, S.AS, S.ek, S.c17
n_ok = 0


def ok(m):
    global n_ok
    n_ok += 1; print("OK  ", m, flush=True)


# ---- second implementation (explicit loop; B via sklearn.LinearRegression without an extra intercept)
def norm_z(X, D, fit_idx, apply_idx):
    lr = LinearRegression(fit_intercept=False).fit(D[fit_idx], X[fit_idx]); B = lr.coef_.T
    res = X[fit_idx] - D[fit_idx] @ B; sd = np.std(res, axis=0, ddof=np.linalg.matrix_rank(D[fit_idx]))
    return (X[apply_idx] - D[apply_idx] @ B) / sd


def second(X, f, tr, te, v, with_as):
    ctrl = [i for i in tr if Y[i] == 0]; F = np.zeros((N, X.shape[1]))
    if v == "C1":
        F, trfit = X.copy(), tr
    elif v in ("V0", "V3"):
        D = S.D_FULL if v == "V0" else S.D_NOERA; F[:] = norm_z(X, D, ctrl, np.arange(N)); trfit = tr
    elif v == "V4":
        F[:] = norm_z(X, S.D_FULL, list(tr), np.arange(N)); trfit = tr
    elif v == "V1":
        rng = np.random.default_rng(29000 + f); R = []
        for e in sorted(set(ERA[ctrl])):
            idx = np.array([i for i in ctrl if ERA[i] == e]); idx = idx[rng.permutation(len(idx))]; R += list(idx[: -(-len(idx) // 2)])
        F[:] = norm_z(X, S.D_FULL, sorted(R), np.arange(N)); trfit = np.array([i for i in tr if i not in set(R)])
    elif v == "V2":
        rng = np.random.default_rng(29000 + f); g = {}
        for k in (0, 1):
            idx = np.array([i for i in tr if Y[i] == k]); idx = idx[rng.permutation(len(idx))]
            for j, i in enumerate(idx):
                g[i] = j % 5
        F[te] = norm_z(X, S.D_FULL, ctrl, te)
        for k in range(5):
            mem = np.array([i for i in tr if g[i] == k]); ck = [i for i in ctrl if g[i] != k]; F[mem] = norm_z(X, S.D_FULL, ck, mem)
        trfit = tr
    if with_as:
        F = np.hstack([F, AS])
    return ek.nested_lr().fit(F[trfit], Y[trfit]).predict_proba(F[te])[:, 1]


def second_feat(X, f, tr, te, v):
    G = {}
    src_globals = globals()
    return None


# (a) features: second implementation compared with the main features (V1-V4 on 3 folds)
def second_features(X, f, tr, te, v):
    ctrl = [i for i in tr if Y[i] == 0]; F = np.zeros((N, X.shape[1]))
    if v in ("V0", "V3"):
        F[:] = norm_z(X, S.D_FULL if v == "V0" else S.D_NOERA, ctrl, np.arange(N)); trfit = tr
    elif v == "V4":
        F[:] = norm_z(X, S.D_FULL, list(tr), np.arange(N)); trfit = tr
    elif v == "V1":
        rng = np.random.default_rng(29000 + f); R = []
        for e in sorted(set(ERA[ctrl])):
            idx = np.array([i for i in ctrl if ERA[i] == e]); idx = idx[rng.permutation(len(idx))]; R += list(idx[: -(-len(idx) // 2)])
        F[:] = norm_z(X, S.D_FULL, sorted(R), np.arange(N)); trfit = np.array([i for i in tr if i not in set(R)])
    elif v == "V2":
        rng = np.random.default_rng(29000 + f); g = {}
        for k in (0, 1):
            idx = np.array([i for i in tr if Y[i] == k]); idx = idx[rng.permutation(len(idx))]
            for j, i in enumerate(idx):
                g[i] = j % 5
        F[te] = norm_z(X, S.D_FULL, ctrl, te)
        for k in range(5):
            mem = np.array([i for i in tr if g[i] == k]); ck = [i for i in ctrl if g[i] != k]; F[mem] = norm_z(X, S.D_FULL, ck, mem)
        trfit = tr
    return F, trfit
worst = 0.0
for f in (0, 11, 30, 49):
    _, _, tr, te = J[f]
    for v in ("V0", "V1", "V2", "V3", "V4"):
        F2, t2 = second_features(BP, f, tr, te, v); F1, t1 = S.features(BP, f, tr, te, v)
        assert np.array_equal(np.asarray(t1), np.asarray(t2)), (f, v); worst = max(worst, np.abs(F1 - F2).max()); assert np.abs(F1 - F2).max() < 1e-5, (f, v, np.abs(F1 - F2).max())
ok(f"V0-V4 features from the second implementation equal the main implementation on 4 folds (max difference {worst:.1e}; identical training indices)")
# (b) predictions of all 50 folds with the second implementation vs. the stored matrix: pooled AUC and per-fold AUC
Zs = np.load(f"{OUT}/B_prediksi_data_asli.npz")
def second_all(v, wa):
    def one(f, tr, te):
        F, trfit = second_features(BP, f, tr, te, v) if v != "C1" else (BP.copy(), tr)
        if wa:
            F = np.hstack([F, AS])
        return ek.nested_lr().fit(F[trfit], Y[trfit]).predict_proba(F[te])[:, 1]
    return A.Parallel(n_jobs=12)(A.delayed(one)(f, tr, te) for f, (_, _, tr, te) in enumerate(J))
for v in S.VARIANTS:
    for wa, tag in ((False, "noAS"), (True, "AS")):
        P2 = c17.to_matrix(J, second_all(v, wa)); P1 = Zs[f"{v}_{tag}"]
        assert abs(c17.pooled_auc(P2) - c17.pooled_auc(P1)) < 3e-3, (v, tag, c17.pooled_auc(P2), c17.pooled_auc(P1))
        assert np.abs(c17.fold_aucs(J, P2).mean() - c17.fold_aucs(J, P1).mean()) < 3e-3, (v, tag)
ok("all 50 folds x 6 pipelines x (without, with age-sex) from the second implementation: pooled AUC and per-fold AUC equal the main result within 3e-3 (solver tolerance; a relative perturbation of 1e-15 in the features changes probabilities by up to about 1.5e-3)")

# ---- replicate 0 H1 and H2a from the CSV
Rn = pd.read_csv(f"{OUT}/A_kebocoran_replikasi.csv")
for name, groups, seed0 in (("H1_all", [np.arange(N)], 29100), ("H2a_era", [np.where(ERA == e)[0] for e in np.unique(ERA)], 29300)):
    Xs = A.shuffle_cols(BP, groups, np.random.default_rng(seed0)); row = Rn[(Rn.scenario == name) & (Rn.rep == 0)].iloc[0]
    for v in S.VARIANTS:
        def _p(f, tr, te, v=v):
            F, trfit = second_features(Xs, f, tr, te, v) if v != "C1" else (Xs.copy(), tr)
            return ek.nested_lr().fit(F[trfit], Y[trfit]).predict_proba(F[te])[:, 1]
        out = A.Parallel(n_jobs=12)(A.delayed(_p)(f, tr, te) for f, (_, _, tr, te) in enumerate(J))
        assert abs(c17.pooled_auc(c17.to_matrix(J, out)) - row[v]) < 2e-3, (name, v, c17.pooled_auc(c17.to_matrix(J, out)), row[v])   # same as above: solver tolerance
    ok(f"{name}: replicate 0 recomputed with the second implementation for 6 pipelines (pooled AUC within 2e-3; solver error)")

# ---- per-fold difference: other implementation (scipy)
Z = np.load(f"{OUT}/B_prediksi_data_asli.npz"); D1 = pd.read_csv(f"{OUT}/B_selisih_per_lipatan.csv")
for _, r in D1.iterrows():
    v, ref = r.test.split(" - "); a = np.array([roc_auc_score(Y[te], Z[f"{v}_{r.input}"][rr, te]) for rr, _, _, te in J]); b = np.array([roc_auc_score(Y[te], Z[f"{ref}_{r.input}"][rr, te]) for rr, _, _, te in J])
    d = a - b; se = np.sqrt(d.var(ddof=1) * (1 / 50 + 40 / 160)); t = d.mean() / se; assert abs(d.mean() - r["diff"]) < 1e-12 and abs(2 * stats.t.sf(abs(t), 49) - r.p) < 1e-12, r.to_dict()
ok(f"per-fold difference: {len(D1)} tests recomputed from the stored predictions with the explicit Nadeau-Bengio formula (1e-12)")

# ---- AUC within strata: compute pairs by brute force
W = pd.read_csv(f"{OUT}/B_dalam_strata.csv"); YD = np.char.add(np.char.add(ERA, "|"), DEV)
def brute(s, g):
    num = den = 0
    for st in np.unique(g):
        e = np.where((g == st) & (Y == 1))[0]; n = np.where((g == st) & (Y == 0))[0]
        for i in e:
            for j in n:
                num += 1.0 * (s[i] > s[j]) + 0.5 * (s[i] == s[j]); den += 1
    return num / den
for v in ("V0", "V1", "V2", "V3", "V4"):
    s = Z[f"{v}_AS"].mean(0); row = W[(W.strata == "year x device") & (W.test == f"{v} - Floor")].iloc[0]
    assert abs(brute(s, YD) - row[f"auc_{v}"]) < 1e-12, v
ok("AUC within year x device strata for V0-V4 (with age-sex) equals the brute-force pair count (1e-12)")

# ---- null proportion and summary statistics
N1 = pd.read_csv(f"{OUT}/B_data_asli_vs_nol.csv"); Ss = pd.read_csv(f"{OUT}/A_kebocoran_ringkasan.csv").set_index("scenario")
for _, r in N1.iterrows():
    g = Rn[Rn.scenario == r.scenario]; suf = {"pooled": "", "within_era": "_wera", "within_year_x_device": "_wyd"}[r.metric]
    assert abs((g[r.variant + suf] - g["C1" + suf]).mean() - r.null_mean) < 1e-12
for sc in Ss.index:
    for v in S.VARIANTS:
        assert abs(Rn[Rn.scenario == sc][v].mean() - Ss.loc[sc, f"{v}_mean"]) < 1e-12 and abs(np.percentile(Rn[Rn.scenario == sc][v], 95) - Ss.loc[sc, f"{v}_p95"]) < 1e-12
ok("leakage summary and comparison with the null distribution recomputed from the replicates (1e-12)")

# ---- mlconfound on V1/V2 (with age-sex): partial (era) and full
mlc = sys.argv[1]; sys.path.insert(0, mlc); sys.modules["pygam"] = types.ModuleType("pygam"); sys.modules["pygam"].LinearGAM = object
from mlconfound.stats import partial_confound_test, full_confound_test
Cp = pd.read_csv(f"{OUT}/C_confounder_parsial.csv"); Cf = pd.read_csv(f"{OUT}/C_confounder_penuh.csv"); cod = lambda a: np.unique(a, return_inverse=True)[1]; rows = []
for v in ("V0", "V1", "V2", "V4"):
    s = Z[f"{v}_AS"].mean(0)
    pr = partial_confound_test(Y, s, cod(ERA), num_perms=2000, cat_y=True, cat_yhat=False, cat_c=True, random_state=1, progress=False, n_jobs=1)
    fr = full_confound_test(Y, s, cod(YD), num_perms=2000, cat_y=True, cat_yhat=False, cat_c=True, random_state=1, progress=False, n_jobs=1)
    mp = Cp[(Cp.model == f"{v}_AS") & (Cp.confounder == "era")].iloc[0]; mf = Cf[(Cf.model == f"{v}_AS") & (Cf.confounder == "year x device")].iloc[0]
    rows.append(dict(model=f"{v}_AS", parsial_era_paket=pr.p, parsial_era_saya=mp.p, penuh_yd_paket=fr.p, penuh_yd_saya=mf.p, r2_parsial_paket=pr.r2_yhat_c, r2_parsial_saya=mp.R2))
    assert abs(pr.r2_yhat_c - mp.R2) < 1e-9
    for a, b in ((pr.p, mp.p), (fr.p, mf.p)):
        assert (a < 0.01 and b < 0.02) or (a > 0.10 and b > 0.05) or abs(a - b) < 0.05, (v, a, b)
M = pd.DataFrame(rows); M.to_csv(f"{OUT}/C_pembanding_mlconfound.csv", index=False); print(M.round(4).to_string(index=False))
ok("mlconfound vs my implementation (V0, V1, V2, V4; partial era and full year x device): R^2 identical; same decision or p differing by < 0.05")
print(f"\nALL {n_ok} CHECK GROUPS PASSED")
