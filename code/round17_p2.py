"""Round 17 P2 (reports/round17_P1-P5_prediction.md Bagian 4; choices in round17_implementation_notes.md): extreme-deviation (tail) features.
Segment z = normmamba.make_tokens normative formula (OLS on the I2 design over all 4-s segments of the non-epilepsy training patients).
Five frozen tail features per patient (p_extreme |z|>2.6, p_extreme2 |z|>2.0, top10, nchan_extreme, iqr_z) are appended to I2
(I2 z + age/sex) -> evalkit.nested_lr. Ablation P2ekor-saja (tails + age/sex). Negative controls: within-patient segment shuffle
(order-invariant by construction), year shuffle 20x, without <=2009, without relative beta+gamma. CPU only."""
import round17_common as c17
import time, numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score
import evalkit as ek, round4 as r4

Y, BP, AS, N = c17.Y, c17.BP, c17.AS, c17.N
Z_ = np.load(f"{ek.D}/seg4s_features.npz", allow_pickle=True); assert list(Z_["subjects"]) == list(r4.SUBJ)
XSEG, PID, NSEG = Z_["X"].astype(np.float64), Z_["pid"], Z_["nseg"]
assert np.all(np.diff(PID) >= 0) and np.array_equal(np.bincount(PID, minlength=N), NSEG)   # contiguous blocks per patient
STARTS = np.concatenate([[0], np.cumsum(NSEG)])
ALL = np.arange(114); NO_BG = np.r_[0:57, 95:114]          # band-major columns: delta, theta, alpha, (beta 57-75, gamma 76-94 dropped), log total
TAILS = ["p_extreme", "p_extreme2", "top10", "nchan_extreme", "iqr_z"]


def seg_z(tr, yg, cols):
    D = c17.year_design(yg); c = tr[Y[tr] == 0]; cm = np.isin(PID, c); Dseg = D[PID]; X = XSEG[:, cols]
    B = np.linalg.lstsq(Dseg[cm], X[cm], rcond=None)[0]; res = X[cm] - Dseg[cm] @ B
    return (X - Dseg @ B) / res.std(0, ddof=np.linalg.matrix_rank(Dseg[cm])), cm


def tails(Z, perm=None):
    nb = Z.shape[1] // 19; out = np.empty((N, 5))
    for i in range(N):
        a = np.abs(Z[STARTS[i]:STARTS[i + 1]])
        if perm is not None:
            a = a[perm[i]]
        q90, q25, q75 = np.percentile(a, [90, 25, 75])
        ch = np.median(a.reshape(len(a), nb, 19).mean(1), 0)                  # per channel: mean |z| over its features, median over segments
        out[i] = [(a > 2.6).mean(), (a > 2.0).mean(), a[a >= q90].mean(), (ch > 2).sum(), q75 - q25]
    return out


def p2_predict(tr, te, yg, cols=ALL, tails_only=False):
    Zs, _ = seg_z(tr, yg, cols); T = tails(Zs)
    zi = r4.normative_z(c17.year_design(yg), tr)[:, cols]
    X = np.hstack([AS, T]) if tails_only else np.hstack([zi, AS, T])
    return ek.nested_lr().fit(X[tr], Y[tr]).predict_proba(X[te])[:, 1]


def checks(tr):
    Zs, cm = seg_z(tr, r4.YEAR_G, ALL)
    c17.pf(np.abs(Zs[cm].mean(0)).max() < 1e-8, f"P2 segment z of training-control segments has mean 0 (max |mean| {np.abs(Zs[cm].mean(0)).max():.1e})")
    print(f"   coverage on training-control segments: |z|>2 {100 * (np.abs(Zs[cm]) > 2).mean():.2f}%, |z|>2.6 {100 * (np.abs(Zs[cm]) > 2.6).mean():.2f}%")
    T = tails(Zs); rng = np.random.default_rng(17200); perm = [rng.permutation(n) for n in NSEG]
    c17.pf(np.allclose(T, tails(Zs, perm)), "negative control (a): tail features identical after within-patient segment shuffle "
           "(all five are order-invariant by construction -> this control is NOT informative)")
    dur = r4.S.dur_min.values
    C = pd.DataFrame({t: [np.corrcoef(T[:, j], NSEG)[0, 1], np.corrcoef(T[:, j], dur)[0, 1]] for j, t in enumerate(TAILS)}, index=["r_nseg", "r_duration"]).T
    print("   correlation of tail features with number of segments / first-session duration (|r| < 0.3 required):"); print(C.round(3).to_string())
    print(f"   all |r| < 0.3: {bool((C.abs() < 0.3).values.all())}  (features are already per-segment proportions/means; no further normalisation, see notes)")
    return C


if __name__ == "__main__":
    a = c17.args(); c17.header("round17_p2"); t0 = time.time()
    print(f"segments {len(XSEG)}, per patient median {np.median(NSEG):.0f} (min {NSEG.min()}, max {NSEG.max()})")
    if a.dry_run:
        r, k, tr, te = c17.JOBS[0]; checks(tr); p = p2_predict(tr, te, r4.YEAR_G); pt = p2_predict(tr, te, r4.YEAR_G, tails_only=True)
        print(f"dry-run fold r{r}k{k}: P2 AUC {roc_auc_score(Y[te], p):.4f} | P2ekor-saja {roc_auc_score(Y[te], pt):.4f} | I2 {roc_auc_score(Y[te], r4.fit_predict('I2', c17.JOBS[0][2], te, 0)):.4f} | {time.time() - t0:.1f} s")
        raise SystemExit
    J = c17.JOBS; C = checks(J[0][2]); C.to_csv(f"{c17.R}/round17_p2_duration_corr.csv")
    P = {"I2": c17.to_matrix(J, Parallel(n_jobs=a.jobs)(delayed(r4.fit_predict)("I2", tr, te, 0) for _, _, tr, te in J))}
    if not c17.pf(abs(c17.pooled_auc(P["I2"]) - 0.71210) < 1e-5, f"I2 == 0.71210 ({c17.pooled_auc(P['I2']):.5f})"):
        raise SystemExit("STOP: reproduction failed")
    run = lambda yg, J_=J, **kw: c17.to_matrix(J_, Parallel(n_jobs=a.jobs)(delayed(p2_predict)(tr, te, yg, **kw) for _, _, tr, te in J_))
    P["P2"] = run(r4.YEAR_G); P["P2ekor"] = run(r4.YEAR_G, tails_only=True); P["P2_noBG"] = run(r4.YEAR_G, cols=NO_BG)
    print(f"P2 AUC {c17.pooled_auc(P['P2']):.5f} | P2ekor-saja {c17.pooled_auc(P['P2ekor']):.5f} | P2 without beta+gamma {c17.pooled_auc(P['P2_noBG']):.5f} (E6a-I2, round 11: 0.6602)", flush=True)
    shuf = []
    for i in range(c17.N_SHUF):
        shuf.append(c17.pooled_auc(run(c17.shuffled_year(i)))); print(f"   year-shuffle {i:2d}: P2 AUC {shuf[-1]:.4f}", flush=True)
    shuf = np.array(shuf); a2 = c17.pooled_auc(P["P2"])
    print(f"year-shuffle null: mean {shuf.mean():.4f} sd {shuf.std(ddof=1):.4f} p95 {np.percentile(shuf, 95):.4f}; P2 > p95: {a2 > np.percentile(shuf, 95)}; "
          f"rank p = {(1 + (shuf >= a2).sum()) / (1 + len(shuf)):.3f}")
    J2 = c17.J169; P169 = {"P2": run(r4.YEAR_G, J2), "I2": c17.to_matrix(J2, Parallel(n_jobs=a.jobs)(delayed(r4.fit_predict)("I2", tr, te, 0) for _, _, tr, te in J2))}
    d, p = ek.corrected_ttest(c17.fold_aucs(J2, P169["P2"]), c17.fold_aucs(J2, P169["I2"]), 135, 34)
    print(f"without <=2009 (169): P2 {c17.pooled_auc(P169['P2']):.4f} | I2 {c17.pooled_auc(P169['I2']):.4f} | dAUC {d:+.4f} p {p:.4f}")
    np.savez(f"{c17.OUT}/p2.npz", P2=P["P2"], P2ekor=P["P2ekor"], P2_noBG=P["P2_noBG"], I2=P["I2"], shuffle_auc=shuf, P2_169=P169["P2"], I2_169=P169["I2"])
    print(f"done in {(time.time() - t0) / 60:.1f} min")
