"""Round 17 P1 (reports/round17_P1-P5_prediction.md Bagian 3; choices in round17_implementation_notes.md): heteroscedastic normative model
with FIXED era means. Mean = OLS on the I2 design fitted on the non-epilepsy training patients only (as I2); scale = residual sd per
year group from the same controls (no shrinkage), floored at 0.2 x the median group sd of that feature. z + age/sex -> evalkit.nested_lr.
Negative controls: year groups shuffled 20x (full procedure), without <=2009 (169 patients); ablation P1-muSaja = I2. CPU only."""
import round17_common as c17
import time, numpy as np
from joblib import Parallel, delayed
import evalkit as ek, round4 as r4

Y, BP, AS, N = c17.Y, c17.BP, c17.AS, c17.N
FLOOR = 0.2


def p1_z(tr, yg):
    """Returns z (N x 114), number of floored (group, feature) cells, and the training-control index."""
    D = c17.year_design(yg); c = tr[Y[tr] == 0]
    B = np.linalg.lstsq(D[c], BP[c], rcond=None)[0]; res = BP[c] - D[c] @ B; mu = D @ B
    glob = res.std(0, ddof=np.linalg.matrix_rank(D[c]))
    gs = {g: np.sqrt((res[yg[c] == g] ** 2).sum(0) / ((yg[c] == g).sum() - 1)) for g in np.unique(yg[c]) if (yg[c] == g).sum() >= 2}
    med = np.median(np.vstack(list(gs.values())), 0); sd = np.empty_like(BP); nfloor = 0
    for g in np.unique(yg):
        s = gs.get(g, glob); s2 = np.maximum(s, FLOOR * med); nfloor += int((s2 > s).sum()) if g in gs else 0; sd[yg == g] = s2
    return (BP - mu) / sd, nfloor, c


def p1_predict(tr, te, yg):
    z, nf, _ = p1_z(tr, yg); X = np.hstack([z, AS])
    return ek.nested_lr().fit(X[tr], Y[tr]).predict_proba(X[te])[:, 1], nf


def checks(tr):
    z, nf, c = p1_z(tr, r4.YEAR_G); yc = r4.YEAR_G[c]
    worst = max(np.abs(z[c][yc == g].mean(0)).max() for g in np.unique(yc))
    c17.pf(worst < 1e-8, f"P1 normative z of training controls has mean 0 within every era (max |mean| {worst:.1e}); floored cells {nf}")
    cov = (np.abs(z[c]) > 2).mean(); print(f"   coverage |z|>2 on training controls: {100 * cov:.2f}% (expected about 4-6%)")
    for g in np.unique(yc):
        print(f"   era {g}: training controls {int((yc == g).sum())}")


if __name__ == "__main__":
    a = c17.args(); c17.header("round17_p1"); t0 = time.time()
    if a.dry_run:
        r, k, tr, te = c17.JOBS[0]; checks(tr)
        p, nf = p1_predict(tr, te, r4.YEAR_G); pi = r4.fit_predict("I2", tr, te, 0)
        from sklearn.metrics import roc_auc_score
        print(f"dry-run fold r{r}k{k}: P1 AUC {roc_auc_score(Y[te], p):.4f} | I2 AUC {roc_auc_score(Y[te], pi):.4f} | {time.time() - t0:.1f} s"); raise SystemExit
    J = c17.JOBS; checks(J[0][2])
    P = {}
    for m in ("I2", "C1"):
        P[m] = c17.to_matrix(J, Parallel(n_jobs=a.jobs)(delayed(r4.fit_predict)(m, tr, te, 0) for _, _, tr, te in J))
    ok = c17.pf(abs(c17.pooled_auc(P["C1"]) - 0.66627) < 1e-5, f"C1 == 0.66627 ({c17.pooled_auc(P['C1']):.5f})") & \
         c17.pf(abs(c17.pooled_auc(P["I2"]) - 0.71210) < 1e-5, f"I2 == 0.71210 ({c17.pooled_auc(P['I2']):.5f})")
    if not ok:
        raise SystemExit("STOP: reproduction failed")
    out = Parallel(n_jobs=a.jobs)(delayed(p1_predict)(tr, te, r4.YEAR_G) for _, _, tr, te in J)
    P["P1"] = c17.to_matrix(J, [o[0] for o in out]); print(f"P1 AUC {c17.pooled_auc(P['P1']):.5f} | floored cells per fold: median {np.median([o[1] for o in out]):.0f}, max {max(o[1] for o in out)}")
    shuf = []
    for i in range(c17.N_SHUF):
        yg = c17.shuffled_year(i); Ps = c17.to_matrix(J, [o[0] for o in Parallel(n_jobs=a.jobs)(delayed(p1_predict)(tr, te, yg) for _, _, tr, te in J)])
        shuf.append(c17.pooled_auc(Ps)); print(f"   year-shuffle {i:2d}: P1 AUC {shuf[-1]:.4f}", flush=True)
    shuf = np.array(shuf); print(f"year-shuffle null: mean {shuf.mean():.4f} sd {shuf.std(ddof=1):.4f} p95 {np.percentile(shuf, 95):.4f}; "
                                 f"P1 > p95: {c17.pooled_auc(P['P1']) > np.percentile(shuf, 95)}; rank p = {(1 + (shuf >= c17.pooled_auc(P['P1'])).sum()) / (1 + len(shuf)):.3f}")
    J2 = c17.J169
    P169 = {"P1": c17.to_matrix(J2, [o[0] for o in Parallel(n_jobs=a.jobs)(delayed(p1_predict)(tr, te, r4.YEAR_G) for _, _, tr, te in J2)]),
            "I2": c17.to_matrix(J2, Parallel(n_jobs=a.jobs)(delayed(r4.fit_predict)("I2", tr, te, 0) for _, _, tr, te in J2))}
    d, p = ek.corrected_ttest(c17.fold_aucs(J2, P169["P1"]), c17.fold_aucs(J2, P169["I2"]), 135, 34)
    print(f"without <=2009 (169): P1 {c17.pooled_auc(P169['P1']):.4f} | I2 {c17.pooled_auc(P169['I2']):.4f} (round 10: 0.6966) | dAUC {d:+.4f} p {p:.4f}")
    np.savez(f"{c17.OUT}/p1.npz", P1=P["P1"], I2=P["I2"], C1=P["C1"], shuffle_auc=shuf, P1_169=P169["P1"], I2_169=P169["I2"])
    print(f"done in {(time.time() - t0) / 60:.1f} min")
