#!/usr/bin/env python3
"""Independent check for round 43 (PRAREGISTRASI_putaran43.md section 6). Second implementation and asserts. CPU only. Usage: python round43_cek_independen.py"""
import os, sys, re
os.environ["CUDA_VISIBLE_DEVICES"] = ""; os.environ.setdefault("OMP_NUM_THREADS", "1")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, "/home/miftah_f/RTX-5090/tuep_fresh/code"); OUT = os.path.join(HERE, "hasil43")
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from scipy import stats
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import round43_fm_kompresi as F
import round43_optimisme as O
import round28_analisis as A
n_ok = 0; Y, J, SUBJ, D, ERA = A.Y, A.J, A.c17.SUBJ, A.D_I2, A.ERA


def ok(m):
    global n_ok
    n_ok += 1; print("OK  ", m, flush=True)


# 1. patient matrix: first session from h5_path (smallest session per patient) and aggregation with a loop
w = np.load(f"{F.DATA}/features_bandpower_windows.npz", allow_pickle=True); X, names, h5 = w["X"], list(w["names"]), w["h5_path"]
sess = np.array([int(re.search(r"_s(\d+)_", p).group(1)) for p in h5]); pid = np.array([p.split("/")[2] for p in h5]); firstmin = np.zeros(len(h5), bool)
for p in np.unique(pid):
    m = pid == p; firstmin[m] = sess[m] == sess[m].min()
e = np.load(f"{F.DATA}/emb_labram_windows.npz", allow_pickle=True); import eegbench_fixed_eval as ebf
first = ebf.first[e["rec"]]; assert first.sum() == 26968 and (first == firstmin).all(); ok("first-session windows from ebf.first equal the independent determination from h5_path (smallest session per patient): 26,968 windows")
M = F.load_mats(); p_list = [3, 77, 150]
for i in p_list:
    idx = np.where(firstmin & (pid == SUBJ[i]))[0]; assert np.abs(np.median(X[idx], 0) - M["BP"][i]).max() < 1e-12 and np.abs(e["E"][idx].mean(0) - M["LaBraM"][i]).max() < 1e-6
    cols = {(b, r): [names.index(f"rel_{b}_{ch}") for ch in chs] for r, chs in F.REGIONS.items() for b in F.BANDS}; Rw = np.zeros((len(idx), 25)); j = 0
    for r in F.REGIONS:
        for b in F.BANDS:
            for t, wi in enumerate(idx):
                Rw[t, j] = np.mean([X[wi, c] for c in cols[(b, r)]])
            j += 1
    c50 = np.concatenate([np.median(Rw, 0), np.percentile(Rw, 75, axis=0) - np.percentile(Rw, 25, axis=0)]); assert np.abs(c50 - M["Comp50"][i]).max() < 1e-5
ok("BP (median), LaBraM (mean) and Comp50 (mean of channels within region, median and IQR across windows) for 3 patients recomputed with a loop (1e-12 for BP; 1e-6 LaBraM; 1e-5 Comp50 because of the float32 source)")

# 2. feats(): second implementation (SVD, np.percentile + interp1d, sklearn LinearRegression)
r, k, tr, te = J[5]
for key, use_pca in (("LaBraM", True), ("Comp50", False)):
    Xm = M[key]; f1 = F.feats(Xm, tr, use_pca, True); Z = Xm.copy()
    if use_pca:
        mu, sd0 = Z[tr].mean(0), Z[tr].std(0); Zs = (Z - mu) / sd0; U, S_, Vt = np.linalg.svd(Zs[tr], full_matrices=False); comp = Zs @ Vt[:50].T
    else:
        comp = Z
    zz = np.empty(comp.shape)
    for j in range(comp.shape[1]):
        q = np.percentile(comp[tr, j], np.linspace(0, 100, 1001)); fq = interp1d(q, np.linspace(0, 1, 1001), bounds_error=False, fill_value=(0.0, 1.0)); zz[:, j] = stats.norm.ppf(np.clip(fq(comp[:, j]), 1e-3, 1 - 1e-3))
    B = LinearRegression(fit_intercept=False).fit(D[tr], zz[tr]).coef_.T; R = zz - D @ B; R = R / R[tr].std(0, ddof=np.linalg.matrix_rank(D[tr]))
    AS = F.AS; Rf = np.hstack([R, (AS - AS[tr].mean(0)) / AS[tr].std(0)])
    if use_pca:                                                                                              # PCA component signs are arbitrary; align per column with the main features
        for j in range(50):
            if np.corrcoef(Rf[tr, j], f1[tr, j])[0, 1] < 0:
                Rf[:, j] = -Rf[:, j]
        assert np.abs(Rf[:, :50] - f1[:, :50]).max() < 5e-2, np.abs(Rf[:, :50] - f1[:, :50]).max()          # rank-normal is sign-sensitive through interpolation: tolerance 5e-2 on values; correlation checked strictly
        assert min(np.corrcoef(Rf[:, j], f1[:, j])[0, 1] for j in range(50)) > 0.995
    else:
        assert np.abs(Rf - f1).max() < 1e-6
ok("feats() (standardization + PCA50, rank-normal, V4, AS) from the second implementation (SVD, np.percentile + interp1d, LinearRegression) equals the main implementation (Comp50: 1e-6; PCA: per-component correlation > 0.995 and difference < 5e-2 after sign alignment)")

# 3. built-in perturbation
F.checks(); ok("built-in checks of round43 (test data do not change training features, residuals orthogonal, test labels not used; PCA and non-PCA)")

# 4. one outer fold reproduced from scratch: own inner CV, own LR
Z = np.load(f"{OUT}/A_prediksi.npz")
for tag, key, pca, uas in (("Comp50__AS", "Comp50", False, True), ("CBraMod-eq__AS", "CBraMod_L12", True, True)):
    f = 7; r, k, tr, te = J[f]; seed = F.kseed("e43", f); Xm = M[key]; sc = {C: [] for C in F.CS}
    for a, b in StratifiedKFold(5, shuffle=True, random_state=seed).split(tr, Y[tr]):
        ia, ib = tr[a], tr[b]; Fz = F.feats(Xm, ia, pca, uas)
        for C in F.CS:
            m = LogisticRegression(penalty="l2", C=C, class_weight="balanced", max_iter=5000).fit(Fz[ia], Y[ia]); sc[C].append(roc_auc_score(Y[ib], m.decision_function(Fz[ib])))
    best = min(F.CS, key=lambda C: (-np.mean(sc[C]), C)); Fz = F.feats(Xm, tr, pca, uas); pr = LogisticRegression(penalty="l2", C=best, class_weight="balanced", max_iter=5000).fit(Fz[tr], Y[tr]).predict_proba(Fz[te])[:, 1]
    assert abs(Z[tag + "__C"][f] - best) < 1e-12 and np.abs(Z[tag][r, te] - pr).max() < 1e-9
ok("fold 7 (Comp50+AS and CBraMod-eq+AS) refitted from scratch (inner CV, chosen C, LR); chosen C and predictions equal the stored result (1e-9)")

# 5. ringkasan A
T = pd.read_csv(f"{OUT}/ringkas_A.csv").set_index("model"); I2 = np.load(os.path.join(HERE, "hasil29/B_prediksi_data_asli.npz"))["V4_AS"]
for tag in ("Comp50__AS", "BENDR-eq__AS", "LaBraM-eq__noAS"):
    fa = np.array([roc_auc_score(Y[te], Z[tag][r, te]) for r, k, tr, te in J]); fi = np.array([roc_auc_score(Y[te], I2[r, te]) for r, k, tr, te in J]); d = fa - fi; se = np.sqrt(d.var(ddof=1) * (1 / 50 + 40 / 160)); p = 2 * stats.t.sf(abs(d.mean() / se), 49)
    assert abs(100 * fa.mean() - T.loc[tag, "auc_fold"]) < 1e-9 and abs(p - T.loc[tag, "p_I2-adj"]) < 1e-12
ok("per-fold AUC, difference from I2-adj and Nadeau-Bengio p for 3 models recomputed with sklearn (1e-9)")

# 6. part C
Cn = pd.read_csv(f"{OUT}/C_null_replikasi.csv"); assert len(Cn) == 200 and not Cn.duplicated(["model", "scen", "rep"]).any() and Cn[["fold_mean", "auc"]].notna().all().all(); rep = 3; rng = np.random.default_rng(43000 + rep); perm = rng.permutation(200)
fa = []
for f in range(5):
    r, k, tr, te = J[f]; pr, _ = F.fit_predict(M["Comp50"][perm], Y, tr, te, False, False, F.kseed("e43null", rep, f)); fa.append(roc_auc_score(Y[te], pr))
assert abs(np.mean(fa) - Cn[(Cn.model == "Comp50") & (Cn.scen == "H1") & (Cn.rep == rep)]["fold_mean"].iloc[0]) < 1e-12
ok("C: 200 complete rows (5 pipelines x 2 conditions x 20 replicates) without duplicates/NaN; one null (Comp50 H1 rep 3) reproduced with the permutation regenerated (1e-12)")

# 7. part D: first two bootstraps with a loop; max-T recomputed
main, allc, ref, _ = O.load_candidates(); names = list(main); P = np.stack([main[k] for k in names]); z = np.load(f"{OUT}/D_bruto_utama.npz", allow_pickle=True); assert list(z["names"]) == names
pos, neg = np.where(Y == 1)[0], np.where(Y == 0)[0]; rng = np.random.default_rng(43100)
for b in range(2):
    while True:
        idx = O.boot_idx(rng, pos, neg); oob = np.setdiff1d(np.arange(200), idx)
        if Y[oob].sum() >= 5 and (len(oob) - Y[oob].sum()) >= 5:
            break
    for kk in (0, 3, 14):
        ain = np.mean([roc_auc_score(Y[idx], P[kk, r][idx]) for r in range(10)]); aoo = np.mean([roc_auc_score(Y[oob], P[kk, r][oob]) for r in range(10)]); assert abs(ain - z["ain"][b, kk]) < 1e-12 and abs(aoo - z["aoob"][b, kk]) < 1e-12
full = np.array([np.mean([roc_auc_score(Y, P[kk, r]) for r in range(10)]) for kk in range(len(names))]); assert np.abs(full - z["full"]).max() < 1e-12
d = full - np.mean([roc_auc_score(Y, ref["Floor"][r]) for r in range(10)]); db = z["boots"] - z["refboot_Floor"][:, None]; se = db.std(0, ddof=1); t_obs = (d / se).max(); tb = ((db - d) / se).max(1); p = (1 + (tb >= t_obs).sum()) / (1 + len(tb)); assert abs(p - O.maxT(z, "Floor")["p"]) < 1e-12
ok("D: first two bootstraps (in-bag and OOB AUC, three candidates) and the full AUC of 15 candidates recomputed with sklearn (1e-12); max-T p (Floor reference) recomputed from the raw bootstrap matrix (1e-12)")

# 8. kalibrasi prosedur D di bawah nol sejati
if os.path.exists(f"{OUT}/D_perm.npy"):
    r = np.load(f"{OUT}/D_perm.npy"); assert abs(r[:, 2].mean() - 0.5) < 0.02 and (r[:, 3] < 0.05).mean() <= 0.10 and r[:, 1].mean() > 0
    ok(f"D (labels permuted, 40x): mean optimism {r[:, 1].mean():.4f} > 0, AUC terkoreksi pemenang {r[:, 2].mean():.4f} (0.5 +- 0.02), max-T p < 0.05 in {(r[:, 3] < 0.05).mean():.3f} (<= 0.10)")
print(f"\nALL {n_ok} CHECK GROUPS PASSED", flush=True)
