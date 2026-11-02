"""Round 23 (reports/round23_prediction.md, commit 5bd41b8). CPU only; frozen out-of-fold predictions, no retraining.
A: F2 vs I2 calibration (slope error |1 - slope|, Brier; round4.calib per repeat, averaged) with a paired patient bootstrap (2000x), on the 200
   patients and on the 169 patients without <=2009 (round 22 A predictions). Holm over 4.
B: within-stratum effect of the normative layer per view (I2-C1, NM-MambaRaw, P3-Riemann; round10 estimator) and their mean, with a joint
   within-stratum bootstrap (2000x). Strata: year (primary), device, year x montage. Holm over the 3 strata for the combined effect."""
import round17_common as c17
import json, numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score
import evalkit as ek, round4 as r4, round9 as r9, round18_fusion as f18, round22_eval as e22

J, J169, Y, N = c17.JOBS, c17.J169, c17.Y, c17.N
NB = 2000


def cal(P, idx):
    """Mean over repeats of calibration slope, intercept (round4.calib) and Brier on patients idx (idx may contain repeats)."""
    out = []
    for p in P:
        q, y = p[idx], Y[idx]; a, b = r4.calib(y, q); out.append((b, a, np.mean((q - y) ** 2)))
    return np.array(out).mean(0)                                                    # slope, intercept, brier


def cal_boot_chunk(PA, PB, pats, seeds):
    res = []
    for s in seeds:
        b = np.random.default_rng(s).choice(pats, len(pats))
        if len(np.unique(Y[b])) < 2:
            continue
        sa, sb = cal(PA, b), cal(PB, b); res.append((abs(1 - sb[0]) - abs(1 - sa[0]), sb[2] - sa[2]))  # positive = A (F2) better
    return res


def boot_p(d):
    d = np.asarray(d); return max(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())), 1 / len(d))


def part_a(PA, PB, label, seed):
    pats = np.where(~np.isnan(PA[0]))[0]; sa, sb = cal(PA, pats), cal(PB, pats)
    chunks = np.array_split(np.arange(seed, seed + NB), 12)
    res = np.array(sum(Parallel(n_jobs=12)(delayed(cal_boot_chunk)(PA, PB, pats, c) for c in chunks), []))
    rows = []
    for j, name, point in [(0, "slope error |1-slope| (I2 - F2)", abs(1 - sb[0]) - abs(1 - sa[0])), (1, "Brier (I2 - F2)", sb[2] - sa[2])]:
        rows.append(dict(sample=label, measure=name, F2=sa[0] if j == 0 else sa[2], I2=sb[0] if j == 0 else sb[2], diff=point,
                         lo=np.percentile(res[:, j], 2.5), hi=np.percentile(res[:, j], 97.5), p=boot_p(res[:, j]), n_boot=len(res)))
    return rows, sa, sb


def strat_boot(M, g, seed):
    """Joint within-stratum bootstrap of the three view differences and their mean."""
    rng = np.random.default_rng(seed); groups = [np.where(g == s)[0] for s in np.unique(g)]; out = []
    pairs = [("I2", "C1"), ("NM", "NMraw"), ("P3", "RR")]
    for _ in range(NB):
        b = np.concatenate([rng.choice(v, len(v)) for v in groups]); d = [r9.strat_auc(Y[b], M[x][b], g[b]) - r9.strat_auc(Y[b], M[y_][b], g[b]) for x, y_ in pairs]
        out.append(d + [np.mean(d)])
    return np.array(out)


if __name__ == "__main__":
    c17.header("round23")
    M0 = f18.load_members(); M0["C1"] = np.load(f"{c17.OUT}/p1.npz")["C1"]; M0["NMraw"] = e22.nm_matrix(f"{ek.D}/round16", "raw", J)
    P = {"I2": M0["I2"], "F2": f18.fuse(M0, ["I2", "P3", "NM"]), "F1": f18.fuse(M0, ["I2", "P3"]), "NM": M0["NM"]}
    P169 = {"I2": np.load(f"{c17.OUT}/p1.npz")["I2_169"], "F2": (np.load(f"{c17.OUT}/p1.npz")["I2_169"] + np.load(f"{c17.OUT}/p3.npz")["P3_169"]
                                                                 + e22.nm_matrix(f"{ek.D}/round22/j169", "F", J169)) / 3}
    ok = [c17.pf(abs(c17.pooled_auc(M0[m]) - v) < 1e-5, f"{m} AUC reproduces {v}") for m, v in
          [("I2", 0.71210), ("C1", 0.66627), ("P3", 0.68318), ("RR", 0.61159), ("NM", 0.70509), ("NMraw", 0.64596)]]
    ok += [c17.pf(abs(c17.pooled_auc(P["F2"]) - 0.72709) < 1e-5, "F2 AUC reproduces 0.72709"),
           c17.pf(abs(c17.pooled_auc(P169["F2"]) - 0.69651) < 1e-5 and abs(c17.pooled_auc(P169["I2"]) - 0.6966) < 5e-5, "F2-169 0.69651 and I2-169 0.6966 reproduced")]
    s_f2, s_i2 = cal(P["F2"], np.arange(N)), cal(P["I2"], np.arange(N))
    ok.append(c17.pf(abs(s_f2[0] - 0.93) < 0.005 and abs(s_i2[0] - 0.54) < 0.005 and abs(s_f2[2] - 0.212) < 5e-4 and abs(s_i2[2] - 0.223) < 5e-4,
                     f"calibration point estimates reproduce rounds 18/22 (slope F2 {s_f2[0]:.3f}, I2 {s_i2[0]:.3f}; Brier {s_f2[2]:.4f}, {s_i2[2]:.4f})"))
    if not all(ok):
        raise SystemExit("STOP")
    # ---------------- A
    print("\n== A. calibration F2 vs I2 (paired patient bootstrap 2000x; positive diff = F2 better) ==")
    rows, sa, sb = part_a(P["F2"], P["I2"], "200 patients", 23000); r2, sa2, sb2 = part_a(P169["F2"], P169["I2"], "169 patients (no <=2009, retrained)", 25000)
    A = pd.DataFrame(rows + r2); A["p_holm"] = f18.holm(A.p.values)
    pats = np.arange(N); c17.pf(np.allclose(cal(P["F2"], pats), s_f2), "bootstrap with identity indices equals the point estimate")
    print(A.round(4).to_string(index=False))
    print(f"   intercepts (descriptive): 200: F2 {sa[1]:+.3f}, I2 {sb[1]:+.3f}; 169: F2 {sa2[1]:+.3f}, I2 {sb2[1]:+.3f}")
    for m in ("F1", "NM"):
        s_ = cal(P[m], pats); print(f"   descriptive {m}: slope {s_[0]:.3f}, intercept {s_[1]:+.3f}, Brier {s_[2]:.4f}")
    curve = []
    for m in ("I2", "F2"):
        q = np.nanmean(P[m], 0); bins = np.quantile(q, np.linspace(0, 1, 11)); k = np.clip(np.digitize(q, bins[1:-1]), 0, 9)
        curve += [dict(model=m, decile=d + 1, mean_pred=q[k == d].mean(), obs_rate=Y[k == d].mean(), n=int((k == d).sum())) for d in range(10)]
    pd.DataFrame(curve).to_csv(f"{c17.R}/round23_calibration_curves.csv", index=False); A.to_csv(f"{c17.R}/round23_calibration.csv", index=False)
    # ---------------- B
    print("\n== B. within-stratum effect of the normative layer per view and combined (joint within-stratum bootstrap 2000x) ==")
    Mm = {k_: np.nanmean(v, 0) for k_, v in M0.items()}; rows = []
    for i, (sname, g) in enumerate(c17.STRATA.items()):
        pt = [r9.strat_auc(Y, Mm[x], g) - r9.strat_auc(Y, Mm[y_], g) for x, y_ in [("I2", "C1"), ("NM", "NMraw"), ("P3", "RR")]]; pt.append(np.mean(pt))
        bs = strat_boot(Mm, g, 23500 + i)
        for j, name in enumerate(["spectral I2 - C1", "temporal NM - Mamba-raw", "spatial P3 - Riemann", "combined (mean of 3)"]):
            rows.append(dict(strata=sname, effect=name, diff=pt[j], lo=np.percentile(bs[:, j], 2.5), hi=np.percentile(bs[:, j], 97.5), p=boot_p(bs[:, j])))
    B = pd.DataFrame(rows); cmb = B.effect == "combined (mean of 3)"; B["p_holm_combined"] = np.nan; B.loc[cmb, "p_holm_combined"] = f18.holm(B.p[cmb].values)
    i2c1 = B[(B.strata == "year") & (B.effect == "spectral I2 - C1")]["diff"].iloc[0]
    c17.pf(abs(i2c1 - (0.7072 - 0.6898)) < 2e-3, f"within-year I2 - C1 reproduces round 10 (about +0.017; here {i2c1:+.4f})")
    print(B.round(4).to_string(index=False)); B.to_csv(f"{c17.R}/round23_within_combined.csv", index=False)
