"""Round 24 R3 (reports/round24_prediction.md, commit 48092a0): spectral view (I2) with an age-dependent normative sd.
Mean = OLS on the I2 design fitted on the training controls (as I2). Log-variance per feature: OLS of log(res^2 + 1e-12) on [1, a, a^2]
(a = age standardised with the training-control mean/sd) on the training controls; sigma^2 = exp(prediction + 1.2704) (E[log chi2_1] = -1.2704);
floor sigma >= 0.2 x median training-control sigma of the feature. z + age/sex -> evalkit.nested_lr.
Control: ages permuted among the training controls for the variance model only (mean model unchanged), 20x. CPU only.
Usage: python round24_agesd.py --dry-run | --run"""
import round17_common as c17
import os, sys, time, argparse, numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score
import evalkit as ek, round4 as r4

Y, N, AS, BP = c17.Y, c17.N, c17.AS, c17.BP.astype(np.float64)
OUT = f"{ek.D}/round24"; os.makedirs(OUT, exist_ok=True); EULER = 1.2704; FLOOR = 0.2


def r3_z(tr, age_perm_seed=None):
    D = r4.DESIGN["I2"]; c = tr[Y[tr] == 0]; B = np.linalg.lstsq(D[c], BP[c], rcond=None)[0]; res = BP[c] - D[c] @ B
    age = r4.AGE.astype(float).copy()
    if age_perm_seed is not None:                                                # control: permute ages of training controls (variance model only)
        age[c] = np.random.default_rng(age_perm_seed).permutation(age[c])
    m, s = age[c].mean(), age[c].std(); a = (age - m) / s; V = np.column_stack([np.ones(N), a, a ** 2])
    G = np.linalg.lstsq(V[c], np.log(res ** 2 + 1e-12), rcond=None)[0]; sig = np.sqrt(np.exp(V @ G + EULER))
    sig = np.maximum(sig, FLOOR * np.median(sig[c], 0)); return (BP - D @ B) / sig, sig, c


def r3_predict(tr, te, seed=None):
    z, _, _ = r3_z(tr, seed); X = np.hstack([z, AS]); return ek.nested_lr().fit(X[tr], Y[tr]).predict_proba(X[te])[:, 1]


def checks():
    tr = c17.JOBS[0][2]; z, sig, c = r3_z(tr); zi = r4.normative_z(r4.DESIGN["I2"], tr)
    nfl = int((sig[c] <= FLOOR * np.median(sig[c], 0) + 1e-15).sum())
    print(f"   fold r0k0: training-control sigma range {sig[c].min():.3g}-{sig[c].max():.3g}; cells at floor {nfl}; |z| > 2 on training controls {100 * (np.abs(z[c]) > 2).mean():.2f}% (I2 {100 * (np.abs(zi[c]) > 2).mean():.2f}%)")
    ok = c17.pf(np.allclose((z * sig)[c].mean(0), 0, atol=1e-8), "R3 residuals of training controls have mean 0 (same mean model as I2)")
    rng = np.random.default_rng(0); B0 = BP.copy(); te = c17.JOBS[0][3]; BP[te] += rng.normal(0, 3, BP[te].shape); zp, _, _ = r3_z(tr); BP[:] = B0
    ok &= c17.pf(np.allclose(zp[tr], z[tr]), "leakage: perturbing test patients leaves R3 training features unchanged")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); g = ap.add_mutually_exclusive_group(required=True); g.add_argument("--dry-run", action="store_true"); g.add_argument("--run", action="store_true")
    ap.add_argument("--jobs", type=int, default=12); a = ap.parse_args(); c17.header("round24_agesd"); t0 = time.time()
    if not checks():
        sys.exit("STOP")
    J = c17.JOBS
    if a.dry_run:
        r, k, tr, te = J[0]; print(f"dry-run fold r{r}k{k}: R3 {roc_auc_score(Y[te], r3_predict(tr, te)):.4f} | I2 {roc_auc_score(Y[te], r4.fit_predict('I2', tr, te, 0)):.4f} | {time.time() - t0:.1f} s"); sys.exit()
    P = c17.to_matrix(J, Parallel(n_jobs=a.jobs)(delayed(r3_predict)(tr, te) for _, _, tr, te in J)); print(f"R3 AUC {c17.pooled_auc(P):.4f}", flush=True)
    perm = [c17.to_matrix(J, Parallel(n_jobs=a.jobs)(delayed(r3_predict)(tr, te, 24300 + i) for _, _, tr, te in J)) for i in range(c17.N_SHUF)]
    P169 = c17.to_matrix(c17.J169, Parallel(n_jobs=a.jobs)(delayed(r3_predict)(tr, te) for _, _, tr, te in c17.J169))
    np.savez(f"{OUT}/r3.npz", R3=P, R3_agepermuted=np.stack(perm), R3_169=P169)
    print(f"done in {(time.time() - t0) / 60:.1f} min")
