"""Round 10 (reports/round10_prediction.md, commit 5d91da9): (1) nested selection over all formal candidates, (2) sensitivity without the
<=2009 year group, (3) consistent estimator for within-stratum AUC. CPU only."""
import os, numpy as np, pandas as pd, warnings
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ["CUDA_VISIBLE_DEVICES"] = ""
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import evalkit as ek, round4 as r4, round5 as r5, round6_eval as r6, round9 as r9
warnings.filterwarnings("ignore"); R = "/home/miftah_f/RTX-5090/tuep_fresh/reports"
SUBJ, Y = r4.SUBJ, r4.Y; POS = {s: i for i, s in enumerate(SUBJ)}; F = ek.folds(); ACQ = r9.ACQ
CANDS = ["C1", "I1", "I2", "I2b", "H5a", "H5b", "DNM-F", "DNM-M"]
REPORTED = {"C1": 0.66627, "I1": None, "I2": 0.71210, "I2b": None, "H5a": 0.6794, "H5b": 0.7119, "DNM-F": 0.6109, "DNM-M": 0.5827}


def fp(m, tr, te):
    if m in ("C1", "I1", "I2", "I2b"):
        return r4.fit_predict(m, tr, te, 0)
    if m == "H5a":
        return r5.h5a_predict(list(SUBJ[tr]), list(SUBJ[te]))
    if m == "H5b":
        return r5.h5b_predict(tr, te)
    return r6.predict(m, tr, te)


def jobs(keep):
    out = []
    for r in range(10):
        for k in range(5):
            te = np.array([POS[s] for s in F[(F.repeat == r) & (F.fold == k)].subject if keep[POS[s]]])
            out.append((r, k, np.setdiff1d(np.where(keep)[0], te), te))
    return out


J200 = jobs(np.ones(len(Y), bool))


def nested_job(r, k, tr, te):
    inner = {}
    for m in CANDS:
        oof = np.zeros(len(tr))
        for a, b in StratifiedKFold(5, shuffle=True, random_state=100 * r + k).split(tr, Y[tr]):
            oof[b] = fp(m, tr[a], tr[b])
        inner[m] = roc_auc_score(Y[tr], oof)
    best = CANDS[int(np.argmax([inner[m] for m in CANDS]))]
    return dict(r=r, k=k, best=best, inner=inner, p=fp(best, tr, te), plain={m: fp(m, tr, te) for m in CANDS})


def per_repeat(J, preds):
    P = np.full((10, len(Y)), np.nan)
    for (r, _, _, te), p in zip(J, preds):
        P[r, te] = p
    return P


def pooled(P, idx):
    rows = [dict(auc=roc_auc_score(Y[idx], p[idx]), **r4.binm(Y[idx], p[idx] > 0.5), brier=np.mean((p[idx] - Y[idx]) ** 2)) for p in P]
    return pd.DataFrame(rows).mean()


def within(P, idx, strata, tests, seed):
    """Fix 3: point estimate and bootstrap CI from the same estimator (scores averaged over repeats, then within-stratum AUC)."""
    rng = np.random.default_rng(seed); M = {m: np.nanmean(P[m], 0) for m in P}; rows = []
    for sname, g in strata.items():
        gi = g[idx]; yi = Y[idx]; groups = {s: np.where(gi == s)[0] for s in np.unique(gi)}
        point = {m: r9.strat_auc(yi, M[m][idx], gi) for m in P}
        rep = {m: np.mean([r9.strat_auc(yi, P[m][r][idx], gi) for r in range(10)]) for m in P}
        boots = {t: [] for t in tests}
        for _ in range(2000):
            b = np.concatenate([rng.choice(v, len(v)) for v in groups.values()])
            for a, c in tests:
                boots[(a, c)].append(r9.strat_auc(yi[b], M[a][idx][b], gi[b]) - r9.strat_auc(yi[b], M[c][idx][b], gi[b]))
        for a, c in tests:
            d = np.array(boots[(a, c)]); p = max(min(1, 2 * min((d <= 0).mean(), (d >= 0).mean())), 1 / 2000)
            rows.append(dict(strata=sname, test=f"{a} - {c}", **{f"auc_{a}": point[a], f"auc_{c}": point[c]}, diff=point[a] - point[c],
                             lo=np.percentile(d, 2.5), hi=np.percentile(d, 97.5), p=p, diff_repeat_mean=rep[a] - rep[c]))
    T = pd.DataFrame(rows); o = np.argsort(T.p.values); h = np.empty(len(T)); run = 0
    for rank, i in enumerate(o):
        run = max(run, min(1, T.p.values[i] * (len(T) - rank))); h[i] = run
    T["p_holm"] = h; return T


if __name__ == "__main__":
    allidx = np.arange(len(Y))
    # ---------------- 1. nested selection
    out = Parallel(n_jobs=12)(delayed(nested_job)(r, k, tr, te) for r, k, tr, te in J200)
    for m in CANDS:
        a = pooled(per_repeat(J200, [o["plain"][m] for o in out]), allidx).auc
        msg = f"(reported {REPORTED[m]})" if REPORTED[m] else "(not separately reported in a P3 table)"
        ok = REPORTED[m] is None or abs(a - REPORTED[m]) < 5e-4
        print(("PASS" if ok else "FAIL") + f" non-nested {m:6s} AUC {a:.4f} {msg}")
    Pn = per_repeat(J200, [o["p"] for o in out]); S = pooled(Pn, allidx)
    fa = {m: [roc_auc_score(Y[te], o["plain"][m]) for (_, _, _, te), o in zip(J200, out)] for m in ["I2", "C1"]}
    fn = [roc_auc_score(Y[te], o["p"]) for (_, _, _, te), o in zip(J200, out)]
    print("\n== 1. nested selection over 8 candidates (strict P3, 200 patients, threshold 0.5) ==")
    print((100 * S[["auc", "acc", "sens", "spec", "prec", "f1", "bac"]]).round(2).to_string(), f"\n brier {S.brier:.3f}")
    print("   selected:", pd.Series([o["best"] for o in out]).value_counts().to_dict())
    print("   inner AUC (mean over 50 outer folds):", {m: round(np.mean([o["inner"][m] for o in out]), 4) for m in CANDS})
    for name, ref in [("I2", fa["I2"]), ("C1", fa["C1"])]:
        d, p = ek.corrected_ttest(fn, ref); print(f"   nested - {name}: dAUC {d:+.4f}, corrected p {p:.4f}")
    print(f"   winner's optimism (I2 pooled - nested pooled): {pooled(per_repeat(J200, [o['plain']['I2'] for o in out]), allidx).auc - S.auc:+.4f}")
    pd.DataFrame([dict(r=o["r"], k=o["k"], best=o["best"], **{f"inner_{m}": o["inner"][m] for m in CANDS}) for o in out]).to_csv(f"{R}/round10_nested_folds.csv", index=False)
    # ---------------- 3. consistent within-stratum estimator, 200 patients
    P200 = {"I2": per_repeat(J200, [o["plain"]["I2"] for o in out]), "C1": per_repeat(J200, [o["plain"]["C1"] for o in out]),
            "Floor": per_repeat(J200, Parallel(n_jobs=12)(delayed(r9.lr_acq)(tr, te) for _, _, tr, te in J200))}
    print(("PASS" if abs(pooled(P200["Floor"], allidx).auc - 0.6479) < 5e-4 else "FAIL") + f" floor AUC {pooled(P200['Floor'], allidx).auc:.4f} reproduced")
    tests = [("I2", "Floor"), ("I2", "C1")]
    T200 = within(P200, allidx, r9.STRATA, tests, 10000)
    r9a = pd.read_csv(f"{R}/round9_A_tests.csv")
    print(("PASS" if np.allclose(T200.diff_repeat_mean.values, r9a["diff"].values, atol=1e-12) else "FAIL") + " per-repeat means equal round 9 point estimates")
    print("\n== 3. within-stratum AUC, consistent estimator (200 patients) ==")
    print(T200.round(4).to_string(index=False)); T200.to_csv(f"{R}/round10_within_200.csv", index=False)
    # ---------------- 2. without the <=2009 group (169 patients)
    keep = r4.YEAR_G != "<=2009"; J169 = jobs(keep); idx = np.where(keep)[0]
    tr0 = J169[0][2]; full = r4.normative_z(r4.DESIGN["I2"], tr0)
    D2 = r4.DESIGN["I2"][:, np.abs(r4.DESIGN["I2"][keep]).sum(0) > 0]; c = tr0[Y[tr0] == 0]; B = np.linalg.lstsq(D2[c], r4.BP[c], rcond=None)[0]
    z2 = (r4.BP - D2 @ B) / (r4.BP[c] - D2[c] @ B).std(0, ddof=np.linalg.matrix_rank(D2[c]))
    print(("PASS" if np.allclose(full[keep], z2[keep], atol=1e-9) and D2.shape[1] == r4.DESIGN["I2"].shape[1] - 1 else "FAIL") + " I2 with an empty <=2009 dummy == I2 with the design rebuilt without it")
    print(f"\n== 2. without <=2009: {keep.sum()} patients ({int(Y[keep].sum())} epilepsy / {int((1 - Y[keep]).sum())} non) ==")
    P169 = {m: per_repeat(J169, Parallel(n_jobs=12)(delayed(r4.fit_predict)(m, tr, te, 0) for _, _, tr, te in J169)) for m in ["I2", "C1"]}
    P169["Floor"] = per_repeat(J169, Parallel(n_jobs=12)(delayed(r9.lr_acq)(tr, te) for _, _, tr, te in J169))
    rows = {m: pooled(P169[m], idx) for m in P169}; D = pd.DataFrame(rows).T
    print((100 * D[["auc", "acc", "sens", "spec", "prec", "f1", "bac"]]).round(2).join(D[["brier"]].round(3)).to_string())
    f169 = {m: [roc_auc_score(Y[te], P169[m][r][te]) for r, _, _, te in J169] for m in P169}
    for a, b in tests:
        ntr, nte = int(keep.sum() * 0.8), int(round(keep.sum() * 0.2))           # outer split sizes of the 169-patient folds (135/34)
        d, p = ek.corrected_ttest(f169[a], f169[b], ntr, nte); print(f"   {a} - {b}: dAUC {d:+.4f}, corrected p ({ntr}/{nte}) {p:.4f}")
    T169 = within(P169, idx, {k: v for k, v in r9.STRATA.items()}, tests, 10169)
    print("\n   within-stratum AUC, 169 patients:"); print(T169.round(4).to_string(index=False))
    D.to_csv(f"{R}/round10_169_pooled.csv"); T169.to_csv(f"{R}/round10_within_169.csv", index=False)
