#!/usr/bin/env python3
"""Independent check for round 41 (PRAREGISTRASI_putaran41.md section 4). Second implementation: brute-force loop over pairs. Every check uses an assert.
Usage: CUDA_VISIBLE_DEVICES="" python round41_cek_independen.py"""
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, "/home/miftah_f/RTX-5090/tuep_fresh/code")
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
import round9 as r9, round4 as r4
OUT = os.path.join(HERE, "hasil41"); n_ok = 0
Y = r9.Y; N = len(Y); AGE = np.asarray(r4.AGE, float); KS = [5.0, 10.0, 15.0, np.inf]; KN = ["5", "10", "15", "tak hingga"]
Z = np.load(os.path.join(HERE, "hasil30", "eval_prediksi_v4.npz"))


def ok(m):
    global n_ok
    n_ok += 1; print("OK  ", m, flush=True)


def brute(y, s, g, age, k):
    num = den = 0
    for i in range(len(y)):
        if y[i] != 1:
            continue
        for j in range(len(y)):
            if y[j] == 0 and g[i] == g[j] and abs(age[i] - age[j]) <= k:
                num += 1.0 * (s[i] > s[j]) + 0.5 * (s[i] == s[j]); den += 1
    return (num / den if den else np.nan), den


# 1. AUC_k and number of pairs from the brute-force loop (F2_adj, Floor, Demo, C1) in both strata
for sname in ("year x device", "year"):
    g = r9.STRATA[sname]; z = np.load(f"{OUT}/bruto_{sname.replace(' ', '_')}.npz"); M = list(z["models"])
    for m in ("F2_adj", "Floor", "Demo", "C1"):
        s = np.nanmean(Z[m], 0)
        for i, k in enumerate(KS):
            a, d = brute(Y, s, g, AGE, k); assert d == z["npairs"][i] and abs(a - z["point"][i, M.index(m)]) < 1e-12, (sname, m, k, a, z["point"][i, M.index(m)])
ok("AUC_k and number of pairs (F2_adj, Floor, Demo, C1; k = 5, 10, 15, infinity; two strata) from the brute-force loop equal the vectorized implementation (1e-12)")

# 2. k = infinity equals round 30 (round9.strat_auc) for all models
g = r9.STRATA["year x device"]; z = np.load(f"{OUT}/bruto_year_x_device.npz"); M = list(z["models"])
for m in M:
    assert abs(r9.strat_auc(Y, np.nanmean(Z[m], 0), g) - z["point"][3, M.index(m)]) < 1e-12, m
ref = pd.read_csv(os.path.join(HERE, "hasil30", "eval_dalam_strata.csv")); r0 = ref[(ref.strata == "year x device") & (ref.test == "F2-adj - Floor")].iloc[0]; assert abs(r0["auc_F2-adj"] - z["point"][3, M.index("F2_adj")]) < 1e-12 and abs(r0["auc_Floor"] - z["point"][3, M.index("Floor")]) < 1e-12
ok("k = infinity reproduces round9.strat_auc for 9 models and the stored values of eval_dalam_strata.csv (F2-adj, metadata) (1e-12)")

# 3. first two bootstrap replicates with the second implementation
for sname, off in (("year x device", 0), ("year", 1)):
    g = r9.STRATA[sname]; z = np.load(f"{OUT}/bruto_{sname.replace(' ', '_')}.npz"); M = list(z["models"]); rng = np.random.default_rng(41000 + off); groups = [np.where(g == s)[0] for s in np.unique(g)]
    for b in range(2):
        idx = np.concatenate([rng.choice(v, len(v)) for v in groups]); yb, gb, ab = Y[idx], g[idx], AGE[idx]
        for m in ("F2_adj", "Floor"):
            s = np.nanmean(Z[m], 0)[idx]
            for i, k in enumerate(KS):
                a, d = brute(yb, s, gb, ab, k); assert d == z["boot_npairs"][b, i] and abs(a - z["boot"][b, i, M.index(m)]) < 1e-12, (sname, b, m, k)
ok("first two bootstrap replicates (two strata; F2_adj and metadata; all k) recomputed with the brute-force loop and the same generator (1e-12)")

# 4. the bootstrap stays within the cell
g = r9.STRATA["year x device"]; rng = np.random.default_rng(41000); groups = [np.where(g == s)[0] for s in np.unique(g)]; idx = np.concatenate([rng.choice(v, len(v)) for v in groups]); assert np.array_equal(np.sort(np.bincount(pd.factorize(g[idx])[0])), np.sort(np.bincount(pd.factorize(g)[0])))
ok("bootstrap resampling preserves the size of each year x device cell")

# 5. summary from the raw bootstrap
T = pd.read_csv(f"{OUT}/ringkas_auc.csv"); D = pd.read_csv(f"{OUT}/ringkas_selisih.csv"); P = pd.read_csv(f"{OUT}/ringkas_primer.csv"); z = np.load(f"{OUT}/bruto_year_x_device.npz"); M = list(z["models"])
for _, r in T[T.strata == "year x device"].sample(15, random_state=3).iterrows():
    i = KN.index(r.k); b = z["boot"][:, i, M.index(r.model)]; assert abs(np.percentile(b, 2.5) - r.lo) < 1e-9 and abs(np.percentile(b, 97.5) - r.hi) < 1e-9 and abs(z["point"][i, M.index(r.model)] - r.auc) < 1e-12
for _, r in P[P.uji.str.startswith("B_")].iterrows():
    i = KN.index(r.uji[2:]); d = z["boot"][:, i, M.index("F2_adj")] - z["boot"][:, i, M.index("Floor")]; p = max(min(1, 2 * min((d <= 0).mean(), (d >= 0).mean())), 1 / 2000); assert abs(p - r.p) < 1e-12
ok("CI, p, and differences of the primary family recomputed from the raw bootstrap (1e-9)")

# 6. holm
p = P.p.values; o = np.argsort(p); h = np.zeros(6); run = 0
for rk, i in enumerate(o):
    run = max(run, min(1, p[i] * (6 - rk))); h[i] = run
assert np.allclose(h, P.p_holm.values); ok("Holm correction over 6 tests recomputed")

# 7. permutation control
pm = np.load(f"{OUT}/bruto_permutasi.npz")["perm"]; assert pm.shape == (200, 4, 9) and np.all(np.abs(np.nanmean(pm[:, :, 0], 0) - 0.5) < 0.02)
s = np.nanmean(Z["F2_adj"], 0); yp = Y.copy(); rng = np.random.default_rng(41001); cells = [np.where(g == s_)[0] for s_ in np.unique(g)]
for v in cells:
    yp[v] = rng.permutation(Y[v])
a, d = brute(yp, s, g, AGE, np.inf); assert abs(a - pm[0, 3, 0]) < 1e-12; ok("label permutation control within cells: mean AUC_k of F2 within 0.5 +- 0.02 and the first permutation recomputed by brute force (1e-12)")

# 8. without strata and caliper = global AUC
a_all, d_all = brute(Y, np.nanmean(Z["F2_adj"], 0), np.zeros(N), AGE, np.inf); assert d_all == 10000 and abs(a_all - roc_auc_score(Y, np.nanmean(Z["F2_adj"], 0))) < 1e-12
ok("without strata and without caliper (10,000 pairs), the pair AUC equals roc_auc_score (1e-12)")
print(f"\nALL {n_ok} CHECK GROUPS PASSED")
