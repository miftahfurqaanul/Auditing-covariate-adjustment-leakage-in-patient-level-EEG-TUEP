"""Round 17 P4 (reports/round17_P1-P5_prediction.md Bagian 6; choices in round17_implementation_notes.md): anatomical region aggregation
of the I2 normative z. Regions frozen from the 10-20 layout (frontal, temporal, central, posterior); bands delta, theta, alpha, beta,
gamma, log total -> 24 region-z features + age/sex. Primary P4: group lasso unavailable -> frozen fallback evalkit.nested_lr (L2) on the
24 features. Secondary P4-ens: six band-specific nested_lr models averaged with equal weights. Ablation P4-tanpa-total (20 features).
Negative controls: training labels permuted 20x, without <=2009. Coefficient sign stability over 50 folds. CPU only."""
import round17_common as c17
import time, numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score
import evalkit as ek, round4 as r4, features as fe

Y, AS, N = c17.Y, c17.AS, c17.N
REGIONS = {"frontal": ["FP1", "FP2", "F7", "F3", "FZ", "F4", "F8"], "temporal": ["T3", "T4", "T5", "T6"],
           "central": ["C3", "CZ", "C4"], "posterior": ["P3", "PZ", "P4", "O1", "O2"]}
BANDS = ["delta", "theta", "alpha", "beta", "gamma", "total"]                       # column blocks of 19 in BP (band-major)
assert sorted(sum(REGIONS.values(), [])) == sorted(fe.CH)
NAMES = [f"{b}_{g}" for b in BANDS for g in REGIONS]
COLS = [[bi * 19 + fe.CH.index(ch) for ch in REGIONS[g]] for bi in range(6) for g in REGIONS]


def region_z(tr):
    z = r4.normative_z(r4.DESIGN["I2"], tr); return np.column_stack([z[:, c].mean(1) for c in COLS])


def fit_lr(X, tr, te, y=None):
    m = ek.nested_lr().fit(X[tr], Y[tr] if y is None else y); return m.predict_proba(X[te])[:, 1], m[-1].coef_[0]


def p4_predict(tr, te, variant, perm_seed=None):
    Z = region_z(tr); y = None if perm_seed is None else np.random.default_rng(perm_seed).permutation(Y[tr])
    if variant == "P4":
        return fit_lr(np.hstack([Z, AS]), tr, te, y)
    if variant == "P4_nototal":
        return fit_lr(np.hstack([Z[:, :20], AS]), tr, te, y)
    if variant == "P4_ens":
        return np.mean([fit_lr(np.hstack([Z[:, 4 * b:4 * b + 4], AS]), tr, te, y)[0] for b in range(6)], 0), None


if __name__ == "__main__":
    a = c17.args(); c17.header("round17_p4"); t0 = time.time()
    print("regions (frozen, 10-20 anatomy):", {g: len(v) for g, v in REGIONS.items()}, "| features:", len(NAMES), NAMES[:4], "...")
    tr0 = c17.JOBS[0][2]; z = r4.normative_z(r4.DESIGN["I2"], tr0); c = tr0[Y[tr0] == 0]
    c17.pf(np.abs(z[c].mean(0)).max() < 1e-8, f"normative z of training controls has mean 0 (max |mean| {np.abs(z[c].mean(0)).max():.1e})")
    Z0 = region_z(tr0); chk = np.allclose(Z0[:, NAMES.index("beta_temporal")], z[:, [57 + fe.CH.index(ch) for ch in ["T3", "T4", "T5", "T6"]]].mean(1))
    c17.pf(chk, "region feature beta_temporal equals the mean of rel_beta z over T3, T4, T5, T6 (independent column lookup)")
    if a.dry_run:
        r, k, tr, te = c17.JOBS[0]
        print(f"dry-run fold r{r}k{k}: " + " | ".join(f"{v} {roc_auc_score(Y[te], p4_predict(tr, te, v)[0]):.4f}" for v in ["P4", "P4_ens", "P4_nototal"])
              + f" | I2 {roc_auc_score(Y[te], r4.fit_predict('I2', tr, te, 0)):.4f} | {time.time() - t0:.1f} s"); raise SystemExit
    J = c17.JOBS; P = {"I2": c17.to_matrix(J, Parallel(n_jobs=a.jobs)(delayed(r4.fit_predict)("I2", tr, te, 0) for _, _, tr, te in J))}
    if not c17.pf(abs(c17.pooled_auc(P["I2"]) - 0.71210) < 1e-5, f"I2 == 0.71210 ({c17.pooled_auc(P['I2']):.5f})"):
        raise SystemExit("STOP: reproduction failed")
    coefs = None
    for v in ["P4", "P4_ens", "P4_nototal"]:
        out = Parallel(n_jobs=a.jobs)(delayed(p4_predict)(tr, te, v) for _, _, tr, te in J); P[v] = c17.to_matrix(J, [o[0] for o in out])
        print(f"{v} AUC {c17.pooled_auc(P[v]):.5f}", flush=True)
        if v == "P4":
            coefs = np.array([o[1][:24] for o in out])
    sgn = np.sign(coefs); cons = np.maximum((sgn > 0).mean(0), (sgn < 0).mean(0))
    S = pd.DataFrame({"feature": NAMES, "mean_std_coef": coefs.mean(0), "sign_consistency": cons}).sort_values("mean_std_coef", key=abs, ascending=False)
    S.to_csv(f"{c17.R}/round17_p4_coef_stability.csv", index=False)
    print("coefficient sign stability over 50 folds (standardised coefficients, top 10 by |mean|):"); print(S.head(10).round(3).to_string(index=False))
    print(f"features with sign consistency >= 90%: {(cons >= 0.9).sum()}/24; per band (sum |mean coef|): "
          + ", ".join(f"{b} {np.abs(coefs.mean(0)[4 * i:4 * i + 4]).sum():.3f}" for i, b in enumerate(BANDS)))
    perm = []
    for i in range(c17.N_SHUF):
        Pp = c17.to_matrix(J, [o[0] for o in Parallel(n_jobs=a.jobs)(delayed(p4_predict)(tr, te, "P4", 17100 + i) for _, _, tr, te in J)])
        perm.append(c17.pooled_auc(Pp)); print(f"   label-permutation {i:2d}: P4 AUC {perm[-1]:.4f}", flush=True)
    perm = np.array(perm); print(f"label-permutation null: mean {perm.mean():.4f} sd {perm.std(ddof=1):.4f} (expected about 0.5); max {perm.max():.4f}")
    J2 = c17.J169; P169 = {"P4": c17.to_matrix(J2, [o[0] for o in Parallel(n_jobs=a.jobs)(delayed(p4_predict)(tr, te, "P4") for _, _, tr, te in J2)]),
                           "I2": c17.to_matrix(J2, Parallel(n_jobs=a.jobs)(delayed(r4.fit_predict)("I2", tr, te, 0) for _, _, tr, te in J2))}
    d, p = ek.corrected_ttest(c17.fold_aucs(J2, P169["P4"]), c17.fold_aucs(J2, P169["I2"]), 135, 34)
    print(f"without <=2009 (169): P4 {c17.pooled_auc(P169['P4']):.4f} | I2 {c17.pooled_auc(P169['I2']):.4f} | dAUC {d:+.4f} p {p:.4f}")
    np.savez(f"{c17.OUT}/p4.npz", P4=P["P4"], P4_ens=P["P4_ens"], P4_nototal=P["P4_nototal"], I2=P["I2"], perm_auc=perm, P4_169=P169["P4"], I2_169=P169["I2"], coefs=coefs)
    print(f"done in {(time.time() - t0) / 60:.1f} min")
