#!/usr/bin/env python3
"""Independent double check for round 28 (see PRAREGISTRASI_analisis_tambahan.md section 6).
Uses separately written code (manual ranks, sklearn confusion_matrix, statsmodels, pandas, the mlconfound 0.21.3 package from PyPI,
and a second implementation of both pipelines). Every check uses an assert; there is no "OK" without a comparison.
Usage: python round28_cek_independen.py <extracted_mlconfound_folder>
"""
import os, sys, types
os.environ["CUDA_VISIBLE_DEVICES"] = ""; os.environ.setdefault("OMP_NUM_THREADS", "1")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from joblib import Parallel, delayed
import statsmodels.api as sm, statsmodels.formula.api as smf
import round28_analisis as A

OUT = A.OUT; Y, N, J, ERA, DEV, MONT, AGE, BP = A.Y, A.N, A.J, A.ERA, A.DEV, A.MONT, A.AGE, A.BP
n_ok = 0


def ok(msg):
    global n_ok
    n_ok += 1; print("OK  ", msg, flush=True)


P = A.load_models()

# ---- 1. korelasi: rank manual
K = pd.read_csv(f"{OUT}/1_korelasi_antar_pandangan.csv").set_index("pair")
S = {m: P[m].mean(0) for m in ("I2", "P3", "NM", "Floor")}
rho = lambda a, b: float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])
for pr in K.index:
    a, b = pr.split("-")
    assert abs(rho(S[a], S[b]) - K.loc[pr, "rho_all"]) < 1e-9, pr
    assert abs(rho(S[a][Y == 1], S[b][Y == 1]) - K.loc[pr, "rho_epilepsy"]) < 1e-9, pr
    assert abs(rho(S[a][Y == 0], S[b][Y == 0]) - K.loc[pr, "rho_nonepilepsy"]) < 1e-9, pr
    assert abs(np.mean([rho(P[a][r], P[b][r]) for r in range(10)]) - K.loc[pr, "rho_repeat_mean"]) < 1e-9, pr
    assert K.loc[pr, "lo"] < K.loc[pr, "rho_all"] < K.loc[pr, "hi"], pr
ok(f"Spearman: {len(K)} pairs x 4 sizes equal the manual rank computation (1e-9)")

# ---- 2. net benefit: confusion_matrix
NB = pd.read_csv(f"{OUT}/2_net_benefit.csv"); NBP = pd.read_csv(f"{OUT}/2_net_benefit_prevalensi.csv")
def nb_manual(m, pt, thr=None):
    vals = []
    for r in range(10):
        pred = (P[m][r] >= (pt if thr is None else thr)).astype(int); tn, fp, fn, tp = confusion_matrix(Y, pred, labels=[0, 1]).ravel()
        vals.append((tp - fp * pt / (1 - pt)) / N)
    return float(np.mean(vals))
for m in ("F2", "I2", "Floor"):
    for _, row in NB.iterrows():
        assert abs(nb_manual(m, row.pt) - row[f"NB_{m}"]) < 1e-12, (m, row.pt)
for _, row in NB.iterrows():
    assert abs(row.treat_all - (0.5 - 0.5 * row.pt / (1 - row.pt))) < 1e-12; assert row["NB_F2_lo"] <= row["NB_F2"] <= row["NB_F2_hi"]
ok("net benefit (prevalence 0.5): 3 models x 17 thresholds equal confusion_matrix (1e-12); treat-all equals the formula; point inside the bootstrap CI")
for _, row in NBP.iterrows():
    pi, pt = row.prevalence, row.pt; odds_t = pt / (1 - pt) * (1 - pi) / pi; thr = odds_t / (1 + odds_t); vals = []
    for r in range(10):
        pred = (P["F2"][r] >= thr).astype(int); tn, fp, fn, tp = confusion_matrix(Y, pred, labels=[0, 1]).ravel(); sens, spec = tp / (tp + fn), tn / (tn + fp)
        vals.append(sens * pi - (1 - spec) * (1 - pi) * pt / (1 - pt))
    assert abs(np.mean(vals) - row.NB_F2) < 1e-12, (pi, pt)
ok("net benefit with adjusted prevalence (F2; 3 prevalences x 17 thresholds) equals the second implementation (1e-12)")

# ---- 3. strata: pandas
SS = pd.read_csv(f"{OUT}/3_strata_ringkasan.csv").set_index("strata"); SC = pd.read_csv(f"{OUT}/3_strata_sel.csv")
age3 = np.where(AGE <= 40, "<=40", np.where(AGE <= 60, "41-60", ">60"))
df = pd.DataFrame(dict(y=Y, era=ERA, dev=DEV, mont=MONT, age3=age3)); df["yd"] = df.era + "|" + df.dev; df["ym"] = df.era + "|" + df.mont
for name, col in (("year", "era"), ("device (3 grup)", "dev"), ("year x device", "yd"), ("year x montage", "ym"), ("age3", "age3")):
    ct = pd.crosstab(df[col], df.y).reindex(columns=[0, 1], fill_value=0); pairs = (ct[0] * ct[1]).sum()
    assert pairs == SS.loc[name, "same_stratum_pairs"], name
    assert int(((ct[0] > 0) & (ct[1] > 0)).sum()) == SS.loc[name, "n_strata_both_classes"], name
    assert int(ct[((ct[0] > 0) & (ct[1] > 0))].sum().sum()) == SS.loc[name, "patients_in_both_class_strata"], name
    sub = SC[SC.strata == name].set_index("stratum")
    for s in ct.index:
        assert (sub.loc[s, "n_nonepilepsy"], sub.loc[s, "n_epilepsy"]) == (ct.loc[s, 0], ct.loc[s, 1]), (name, s)
ok("number of patients per stratum and pairs within strata equal the pandas crosstab (5 strata types)")

# ---- 4a. statistik R^2 lewat statsmodels
Tp = pd.read_csv(f"{OUT}/4_confounder_parsial.csv"); Tf = pd.read_csv(f"{OUT}/4_confounder_penuh.csv")
Sm = {m: P[m].mean(0) for m in A.MODELS}
for _, r in Tp.iterrows():
    c = {"era": ERA, "device": DEV, "montage": MONT, "age": AGE}[r.confounder]
    d = pd.DataFrame(dict(yh=Sm[r.model], c=c)); f = "yh ~ C(c)" if r.confounder != "age" else "yh ~ c"
    assert abs(smf.ols(f, d).fit().rsquared - r.R2) < 1e-9, (r.model, r.confounder)
for _, r in Tf.iterrows():
    d = pd.DataFrame(dict(yh=Sm[r.model], y=Y)); assert abs(smf.ols("yh ~ y", d).fit().rsquared - r.R2) < 1e-9, r.model
ok(f"statistik R^2: {len(Tp)} partial tests and {len(Tf)} full tests equal statsmodels OLS (1e-9)")
# Holm again
from statsmodels.stats.multitest import multipletests
for T in (Tp, Tf):
    pr = T[T.family == "primary"]; h = multipletests(pr.p.values, method="holm")[1]
    assert np.allclose(h, pr.p_holm.values, atol=1e-12)
ok("Holm within the primary family equals statsmodels.multipletests (1e-12)")

# ---- 4b. paket mlconfound (p-value)
mlc_dir = sys.argv[1]; sys.path.insert(0, mlc_dir)
sys.modules["pygam"] = types.ModuleType("pygam"); sys.modules["pygam"].LinearGAM = object
from mlconfound.stats import partial_confound_test, full_confound_test
def codes(a): return np.unique(a, return_inverse=True)[1]
NP = 2000; rows = []
for m in ("Floor", "C1", "I2", "F2"):
    for cn, c, cat in (("era", ERA, True), ("device", DEV, True), ("montage", MONT, True), ("age", AGE, False)):
        cc = codes(c) if cat else c
        res = partial_confound_test(Y, Sm[m], cc, num_perms=NP, cat_y=True, cat_yhat=False, cat_c=cat, cond_dist_method="linear", random_state=1, progress=False, n_jobs=1)
        mine = Tp[(Tp.model == m) & (Tp.confounder == cn)].iloc[0]; rows.append(dict(test="parsial", model=m, conf=cn, p_paket=res.p, p_saya=mine.p, r2_paket=res.r2_yhat_c, r2_saya=mine.R2))
for m in ("Floor", "I2", "F2"):
    for cn, c in (("era", ERA), ("device", DEV), ("year x device", np.char.add(np.char.add(ERA, "|"), DEV))):
        res = full_confound_test(Y, Sm[m], codes(c), num_perms=NP, cat_y=True, cat_yhat=False, cat_c=True, random_state=1, progress=False, n_jobs=1)
        mine = Tf[(Tf.model == m) & (Tf.confounder == cn)].iloc[0]; rows.append(dict(test="penuh", model=m, conf=cn, p_paket=res.p, p_saya=mine.p, r2_paket=res.r2_y_yhat, r2_saya=mine.R2))
C = pd.DataFrame(rows); C.to_csv(f"{OUT}/4_pembanding_mlconfound.csv", index=False); print(C.round(4).to_string(index=False))
disc = []
for _, r in C.iterrows():
    assert abs(r.r2_paket - r.r2_saya) < 1e-9, ("R2", r.to_dict())
    se = np.sqrt(max(r.p_saya, 1 / 10001) * (1 - r.p_saya) / 10000 + max(r.p_paket, 1 / NP) * (1 - r.p_paket) / NP) if 0 < r.p_paket < 1 else 0.0
    tol = 4 * se + 0.01 if r.p_saya > 0.001 or r.p_paket > 0.003 else 0.006     # very small p: both <= 0.006
    selaras = abs(r.p_paket - r.p_saya) <= tol
    if not selaras:
        disc.append(r.to_dict())
    if r.model in A.PRIMARY:                                                    # primary-family rows: must be aligned
        assert selaras, ("primary p not aligned", r.to_dict(), tol)
    if r.p_saya < 0.01: assert r.p_paket < 0.02, r.to_dict()                    # the decision at the extremes must be the same for all rows
    if r.p_saya > 0.10: assert r.p_paket > 0.05, r.to_dict()
C["selaras_MC"] = [abs(x.p_paket - x.p_saya) <= (4 * np.sqrt(max(x.p_saya, 1 / 10001) * (1 - x.p_saya) / 10000 + max(x.p_paket, 1 / NP) * (1 - x.p_paket) / NP) + 0.01 if 0 < x.p_paket < 1 else 0.006) if (x.p_saya > 0.001 or x.p_paket > 0.003) else abs(x.p_paket - x.p_saya) <= 0.006 for x in C.itertuples()]
C.to_csv(f"{OUT}/4_pembanding_mlconfound.csv", index=False)
print("rows not aligned (non-primary; different null: the package uses MCMC with estimated y|c, I used exact stratified permutation):", disc)
ok(f"mlconfound 0.21.3 vs my implementation: {len(C)} tests; R^2 identical (1e-9); primary rows (I2, F2) aligned within Monte Carlo error; decision at the extremes (p<0.01 / p>0.10) the same for all rows; {len(disc)} non-primary rows differ slightly (reported)")

# ---- 5. Hamdan: second pipeline implementation + permutation properties
H = pd.read_csv(f"{OUT}/5_hamdan_replikasi.csv"); ref = pd.read_json(f"{OUT}/5_hamdan_acuan_data_asli.json", typ="series")
CS = A.ek.CS; Dm = A.D_I2
def pipe(X, tr, te, use_z, with_as):
    if use_z:
        c = tr[Y[tr] == 0]; B = np.linalg.lstsq(Dm[c], X[c], rcond=None)[0]; sd = (X[c] - Dm[c] @ B).std(0, ddof=np.linalg.matrix_rank(Dm[c])); F = (X - Dm @ B) / sd
    else:
        F = X
    if with_as:
        F = np.hstack([F, A.c17.AS])
    mdl = make_pipeline(StandardScaler(), LogisticRegressionCV(Cs=CS, cv=5, scoring="roc_auc", class_weight="balanced", max_iter=5000)).fit(F[tr], Y[tr])
    return mdl.predict_proba(F[te])[:, 1]
def pooled(X, use_z, with_as):
    out = Parallel(n_jobs=12)(delayed(pipe)(X, tr, te, use_z, with_as) for _, _, tr, te in J)
    return A.c17.pooled_auc(A.c17.to_matrix(J, out))
for name, (groups, seed0) in A.hamdan_groups().items():
    Xs = A.shuffle_cols(BP, groups, np.random.default_rng(seed0))
    for g in groups:                                         # permutation property: columns are permuted only within groups
        assert np.allclose(np.sort(Xs[g], 0), np.sort(BP[g], 0))
    mask = np.zeros(N, bool); [mask.__setitem__(g, True) for g in groups]; assert mask.all()
    assert (Xs != BP).mean() > 0.9, "permutation moved too few values"
    row = H[(H.scenario == name) & (H.rep == 0)].iloc[0]
    for col, (uz, wa) in {"C1_noAS": (False, False), "I2_noAS": (True, False), "C1_AS": (False, True), "I2_AS": (True, True)}.items():
        assert abs(pooled(Xs, uz, wa) - row[col]) < 1e-9, (name, col)
    ok(f"Hamdan {name}: replicate 0 recomputed with the second implementation for 4 pipelines (1e-9); the permutation preserves the content of each group")
assert abs(pooled(BP, True, True) - 0.7121) < 5e-4 and abs(pooled(BP, False, True) - 0.66627) < 5e-4 and abs(pooled(BP, True, False) - ref["I2_noAS"]) < 1e-9
ok("real-data reference: I2 (with age/sex) 0.7121 and C1 0.66627 reproduced by the second implementation")
print(f"\nALL {n_ok} CHECK GROUPS PASSED")
