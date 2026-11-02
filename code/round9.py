"""Round 9 (reports/round9_prediction.md, commit 0d127e0): robustness of I2 under the TUEP-Strict protocol.
A within-stratum AUC (year, device, year x device) vs the metadata floor and C1; B leave-one-device-out; C specification curve;
D cross-session consistency; E subgroups. CPU only."""
import os, sys, numpy as np, pandas as pd, warnings
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ["CUDA_VISIBLE_DEVICES"] = ""      # CPU only: XGBoost 3 otherwise opens a CUDA context per worker
from joblib import Parallel, delayed
from scipy import stats
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import evalkit as ek, round4 as r4
warnings.filterwarnings("ignore"); R = "/home/miftah_f/RTX-5090/tuep_fresh/reports"
SUBJ, Y, BP, AGESEX = r4.SUBJ, r4.Y, r4.BP, r4.AGESEX; POS = {s: i for i, s in enumerate(SUBJ)}; F = ek.folds()
with np.load(f"{ek.D}/p3_patient_table.npz", allow_pickle=True) as _T:
    ACQ = _T["acq"]
JOBS = [(r, np.setdiff1d(np.arange(len(Y)), te), te) for r in range(10) for k in range(5)
        for te in [np.array([POS[s] for s in F[(F.repeat == r) & (F.fold == k)].subject])]]
STRATA = {"year": r4.YEAR_G, "device": r4.DEV_G, "year x device": np.char.add(np.char.add(r4.YEAR_G, "|"), r4.DEV_G)}


def strat_auc(y, p, g):
    """Concordance over (epilepsy, non-epilepsy) pairs from the same stratum; ties count 0.5."""
    num = den = 0.0
    for s in np.unique(g):
        a, b = p[(g == s) & (y == 1)], p[(g == s) & (y == 0)]
        if len(a) and len(b):
            d = a[:, None] - b[None, :]; num += (d > 0).sum() + 0.5 * (d == 0).sum(); den += d.size
    return num / den


def oof(fp):
    """fp(tr, te) -> probabilities; returns (10, 200) out-of-fold predictions."""
    out = Parallel(n_jobs=12)(delayed(fp)(tr, te) for _, tr, te in JOBS); P = np.zeros((10, len(Y)))
    for (r, _, te), o in zip(JOBS, out):
        P[r, te] = o
    return P


def lr_acq(tr, te):
    return ek.nested_lr().fit(ACQ[tr], Y[tr]).predict_proba(ACQ[te])[:, 1]


# ---------------------------------------------------------------- C specification curve
HARM = {"none": None, "age+sex": "I1", "age+sex+year": "I2", "age+sex+device": "I2b"}


def spec_features(harm, feats, cov, tr):
    X = BP if HARM[harm] is None else r4.normative_z(r4.DESIGN[HARM[harm]], tr)
    X = X if feats == "all" else X[:, :95]
    return np.hstack([X, AGESEX]) if cov else X


def spec_model(clf, seed):
    if clf == "LR-L2":
        return ek.nested_lr()
    if clf == "LR-L1":
        return make_pipeline(StandardScaler(), LogisticRegressionCV(Cs=ek.CS, cv=5, scoring="roc_auc", class_weight="balanced", penalty="l1", solver="liblinear", max_iter=5000))
    return RandomForestClassifier(n_estimators=500, class_weight="balanced", random_state=seed, n_jobs=1)


def spec_fp(harm, feats, cov, clf, tr, te, seed):
    X = spec_features(harm, feats, cov, tr); return spec_model(clf, seed).fit(X[tr], Y[tr]).predict_proba(X[te])[:, 1]


def pooled_auc(P):
    return np.mean([roc_auc_score(Y, p) for p in P])


# ---------------------------------------------------------------- D cross-session
def norm_fit(design, tr):
    c = tr[Y[tr] == 0]; B = np.linalg.lstsq(design[c], BP[c], rcond=None)[0]
    sd = (BP[c] - design[c] @ B).std(0, ddof=np.linalg.matrix_rank(design[c])); return B, sd


def year_row(groups):
    levels = sorted(set(r4.YEAR_G)); ref = pd.Series(r4.YEAR_G).value_counts().idxmax()
    return np.column_stack([(groups == l).astype(float) for l in levels if l != ref])


if __name__ == "__main__":
    out = {}
    P = {"I2": oof(lambda tr, te: r4.fit_predict("I2", tr, te, 0)), "C1": oof(lambda tr, te: r4.fit_predict("C1", tr, te, 0)), "Floor": oof(lr_acq)}
    print(("PASS" if abs(pooled_auc(P["C1"]) - 0.66627) < 1e-5 and abs(pooled_auc(P["I2"]) - 0.71210) < 5e-5 and abs(pooled_auc(P["Floor"]) - 0.6479) < 5e-4 else "FAIL")
          + f" C1 {pooled_auc(P['C1']):.5f}, I2 {pooled_auc(P['I2']):.5f}, floor {pooled_auc(P['Floor']):.5f} reproduced")
    yy = np.array([1, 1, 0, 0, 1, 0]); pp = np.array([.9, .2, .1, .5, .3, .4]); gg = np.array(list("aaabbb"))
    print(("PASS" if abs(strat_auc(yy, pp, gg) - 0.5) < 1e-12 and abs(strat_auc(Y, P["C1"][0], np.zeros(len(Y))) - roc_auc_score(Y, P["C1"][0])) < 1e-12 else "FAIL")
          + " stratified AUC: hand example and single-stratum == ordinary AUC")
    # ---------------- A
    print("\n== A. within-stratum AUC (pairs from the same stratum only), mean over 10 repeats ==")
    rows = []; rng = np.random.default_rng(9000); M = {m: P[m].mean(0) for m in P}
    for sname, g in STRATA.items():
        v = {m: np.mean([strat_auc(Y, P[m][r], g) for r in range(10)]) for m in P}
        npairs = sum(((g == s) & (Y == 1)).sum() * ((g == s) & (Y == 0)).sum() for s in np.unique(g))
        idx = {s: np.where(g == s)[0] for s in np.unique(g)}; bs = {("I2", "Floor"): [], ("I2", "C1"): []}
        for _ in range(2000):
            b = np.concatenate([rng.choice(i, len(i)) for i in idx.values()])
            for a, c in bs:
                bs[(a, c)].append(strat_auc(Y[b], M[a][b], g[b]) - strat_auc(Y[b], M[c][b], g[b]))
        print(f"   {sname:14s} ({npairs} within-stratum pairs of {100 * 100}): " + "  ".join(f"{m} {v[m]:.3f}" for m in P))
        for (a, c), d in bs.items():
            d = np.array(d); p = min(1, 2 * min((d <= 0).mean(), (d >= 0).mean()))
            rows.append(dict(strata=sname, test=f"{a} - {c}", diff=v[a] - v[c], lo=np.percentile(d, 2.5), hi=np.percentile(d, 97.5), p=max(p, 1 / 2000)))
        out[f"A_{sname}"] = v
    T = pd.DataFrame(rows); o = np.argsort(T.p.values); h = np.empty(len(T)); run = 0
    for rank, i in enumerate(o):
        run = max(run, min(1, T.p.values[i] * (len(T) - rank))); h[i] = run
    T["p_holm"] = h; print(T.round(4).to_string(index=False)); T.to_csv(f"{R}/round9_A_tests.csv", index=False)
    # ---------------- B
    print("\n== B. leave-one-device-out (train on the other device + 'other', test on the held-out device) ==")
    rb = []
    for held in ["X", "Intution/Ultrasom"]:
        te = np.where(r4.DEV_G == held)[0]; tr = np.where(r4.DEV_G != held)[0]; boots = [rng.integers(0, len(te), len(te)) for _ in range(2000)]
        for m in ["C1", "I1", "I2"]:
            p = r4.fit_predict(m, tr, te, 0); a = roc_auc_score(Y[te], p); b = [roc_auc_score(Y[te][i], p[i]) for i in boots if len(np.unique(Y[te][i])) == 2]
            rb.append(dict(held_out=held, n_test=len(te), n_train=len(tr), method=m, auc=a, lo=np.percentile(b, 2.5), hi=np.percentile(b, 97.5), bac=r4.binm(Y[te], p > 0.5)["bac"]))
    B = pd.DataFrame(rb); print(B.round(3).to_string(index=False)); B.to_csv(f"{R}/round9_B_lodo.csv", index=False)
    # ---------------- C
    print("\n== C. specification curve (48 specifications, P3 AUC) ==")
    specs = [(h_, f_, c_, k_) for h_ in HARM for f_ in ["all", "rel"] for c_ in [True, False] for k_ in ["LR-L2", "LR-L1", "RF"]]
    res = []
    for sp in specs:
        Pp = oof(lambda tr, te, sp=sp: spec_fp(*sp, tr, te, 0)); res.append(dict(harm=sp[0], feats=sp[1], cov=sp[2], clf=sp[3], auc=pooled_auc(Pp)))
    C = pd.DataFrame(res).sort_values("auc").reset_index(drop=True)
    c1 = C[(C.harm == "none") & (C.feats == "all") & C["cov"] & (C.clf == "LR-L2")].auc.iloc[0]; i2 = C[(C.harm == "age+sex+year") & (C.feats == "all") & C["cov"] & (C.clf == "LR-L2")].auc.iloc[0]
    print(("PASS" if abs(c1 - 0.66627) < 1e-5 and abs(i2 - 0.71210) < 5e-5 else "FAIL") + f" specification C1 {c1:.5f} and I2 {i2:.5f} reproduce the primary models")
    print(C.round(4).to_string()); C.to_csv(f"{R}/round9_C_speccurve.csv", index=False)
    piv = C.pivot_table(index=["feats", "cov", "clf"], columns="harm", values="auc"); dy = piv["age+sex+year"] - piv["none"]
    print(f"\n   median AUC by harmonisation: {C.groupby('harm').auc.median().round(4).to_dict()}")
    print(f"   age+sex+year minus none, 12 matched pairs: {np.round(dy.values, 4).tolist()}; positive {int((dy > 0).sum())}/12, "
          f"sign test p = {stats.binomtest(int((dy > 0).sum()), 12, 0.5).pvalue:.4f}")
    for other in ["age+sex", "age+sex+device"]:
        d2 = piv["age+sex+year"] - piv[other]; print(f"   age+sex+year minus {other}: positive {int((d2 > 0).sum())}/12, median {d2.median():+.4f}, sign test p = {stats.binomtest(int((d2 > 0).sum()), 12, 0.5).pvalue:.4f}")
    # ---------------- D
    print("\n== D. cross-session consistency (models trained on first sessions of training patients) ==")
    import round5 as r5
    XS, MS = r5.XS, r5.MS; yg = np.where(MS.ygroup.values == ">=2014", "2013", MS.ygroup.values)
    desS = np.column_stack([np.ones(len(MS)), MS.age.values, MS.male.values, year_row(yg)])
    B0, sd0 = norm_fit(r4.DESIGN["I2"], JOBS[0][1])
    print(("PASS" if np.allclose(((XS[[r5.FIRST[s] for s in SUBJ]] - desS[[r5.FIRST[s] for s in SUBJ]] @ B0) / sd0), r4.normative_z(r4.DESIGN["I2"], JOBS[0][1])) else "FAIL")
          + " session-level I2 transform reproduces the first-session normative z")
    rs = {m: [] for m in ["C1", "I2"]}; ra = {m: [] for m in ["C1", "I2"]}

    def later_job(r, tr, te):
        res = {}
        for m in ["C1", "I2"]:
            Xtr = r4.features(m, tr); mdl = ek.nested_lr().fit(Xtr[tr], Y[tr])
            if m == "C1":
                XL = np.column_stack([XS, MS.age.values, MS.male.values])
            else:
                Bm, sdm = norm_fit(r4.DESIGN["I2"], tr); XL = np.column_stack([(XS - desS @ Bm) / sdm, MS.age.values, MS.male.values])
            first = mdl.predict_proba(Xtr[te])[:, 1]; later = {}
            for j, s in enumerate(SUBJ[te]):
                ids = np.where((MS.subject.values == s) & ~MS["first"].values)[0]
                if len(ids):
                    later[s] = (first[j], mdl.predict_proba(XL[ids])[:, 1].mean(), Y[POS[s]])
            res[m] = later
        return r, res
    outD = Parallel(n_jobs=12)(delayed(later_job)(r, tr, te) for r, tr, te in JOBS)
    for m in ["C1", "I2"]:
        for r in range(10):
            L = {}
            for rr, res in outD:
                if rr == r:
                    L.update(res[m])
            a = np.array(list(L.values())); rs[m].append(stats.spearmanr(a[:, 0], a[:, 1])[0]); ra[m].append(roc_auc_score(a[:, 2], a[:, 1]))
        print(f"   {m}: patients with later sessions {len(a)} ({int(a[:, 2].sum())} epilepsy / {int(len(a) - a[:, 2].sum())} non); "
              f"Spearman(first, later) {np.mean(rs[m]):.3f}; AUC on later sessions {np.mean(ra[m]):.3f}")
    # ---------------- E
    print("\n== E. subgroups (scores averaged over 10 repeats; bootstrap 95% CI) ==")
    age_g = np.where(r4.AGE <= 30, "<=30", np.where(r4.AGE <= 50, "31-50", ">50")); sex_g = np.where(r4.MALE == 1, "male", "female"); re_ = []
    for gname, g in [("sex", sex_g), ("age", age_g)]:
        for s in np.unique(g):
            i = np.where(g == s)[0]; bt = [rng.choice(i, len(i)) for _ in range(2000)]
            for m in ["C1", "I2"]:
                b = [roc_auc_score(Y[j], M[m][j]) for j in bt if len(np.unique(Y[j])) == 2]
                re_.append(dict(group=gname, level=s, n=len(i), n_epi=int(Y[i].sum()), method=m, auc=roc_auc_score(Y[i], M[m][i]), lo=np.percentile(b, 2.5), hi=np.percentile(b, 97.5)))
    E = pd.DataFrame(re_); print(E.round(3).to_string(index=False)); E.to_csv(f"{R}/round9_E_subgroups.csv", index=False)
