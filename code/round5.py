"""Round 5 (reports/round5_prediction.md, commit 85897b4): harmonised all-session training (H5a), location-scale harmonisation (H5b),
compared with C1 and I2 on the strict P3 protocol; negative control for H5a with permuted session year groups. CPU only."""
import os, sys, numpy as np, pandas as pd, warnings
os.environ.setdefault("OMP_NUM_THREADS", "1")
from joblib import Parallel, delayed
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
import evalkit as ek, round4 as r4
warnings.filterwarnings("ignore"); R = "/home/miftah_f/RTX-5090/tuep_fresh/reports"; N_PERM = 50
SUBJ, Y, BP, AGESEX = r4.SUBJ, r4.Y, r4.BP, r4.AGESEX; POS = {s: i for i, s in enumerate(SUBJ)}


def session_table():
    """Session features = median of clean windows of all recordings of the session (same construction as the patient table)."""
    z = np.load(f"{ek.D}/features_bandpower_windows.npz", allow_pickle=True); Xw, win_h5 = z["X"], z["h5_path"]
    rec = pd.read_csv(f"{ek.D}/features_bandpower_recordings.csv"); rec = rec[rec.n_clean > 0].reset_index(drop=True)
    hdr = pd.read_csv(f"{ek.D}/edf_headers.csv"); hdr["age"] = hdr.patient.str.extract(r"Age:(-?\d+)")[0].astype(float)
    rec = rec.merge(hdr[["file", "age", "year"]].rename(columns={"year": "year_h"}), on="file", how="left")
    p = {h: i for i, h in enumerate(rec.h5_path)}; wr = np.array([p.get(h, -1) for h in win_h5]); Xw, wr = Xw[wr >= 0], wr[wr >= 0]
    keys = rec[["subject", "session_no"]].drop_duplicates().sort_values(["subject", "session_no"]).reset_index(drop=True)
    feats, meta = [], []
    for s, n in keys.itertuples(index=False):
        ids = np.where((rec.subject.values == s) & (rec.session_no.values == n))[0]
        feats.append(np.median(Xw[np.isin(wr, ids)], axis=0)); r0 = rec.iloc[ids[0]]
        meta.append(dict(subject=s, session_no=n, label=r0.label, age=r0.age, year=r0.year_h))
    M = pd.DataFrame(meta); M["first"] = M.session_no == M.groupby("subject").session_no.transform("min")
    M["male"] = AGESEX[[POS[s] for s in M.subject], 1]
    M["ygroup"] = np.where(M.year <= 2009, "<=2009", np.where(M.year >= 2014, ">=2014", M.year.astype(int).astype(str)))
    return np.vstack(feats), M


XS, MS = session_table()
FIRST = {s: i for i, s in zip(MS.index[MS["first"]], MS.subject[MS["first"]])}


def design(groups):
    levels = sorted(set(groups)); ref = pd.Series(groups).value_counts().idxmax()
    return np.column_stack([np.ones(len(MS)), MS.age.values, MS.male.values] + [(groups == l).astype(float) for l in levels if l != ref])


DES = design(MS.ygroup.values)


def h5a_predict(tr_p, te_p, des=DES):
    tr = np.where(MS.subject.isin(tr_p))[0]; te = np.array([FIRST[s] for s in te_p])
    assert not set(MS.subject.values[tr]) & set(te_p)
    c = tr[MS.label.values[tr] == 0]; B = np.linalg.lstsq(des[c], XS[c], rcond=None)[0]
    sd = (XS[c] - des[c] @ B).std(0, ddof=np.linalg.matrix_rank(des[c])); Z = (XS - des @ B) / sd
    X = np.column_stack([Z, MS.age.values, MS.male.values]); yy = MS.label.values
    w = 1.0 / MS.subject.iloc[tr].map(MS.subject.iloc[tr].value_counts()).values
    m = ek.nested_lr(); m.set_params(logisticregressioncv__cv=list(GroupKFold(5).split(X[tr], yy[tr], MS.subject.values[tr])))
    m.fit(X[tr], yy[tr], logisticregressioncv__sample_weight=w); return m.predict_proba(X[te])[:, 1]


def h5b_predict(tr, te):
    """I2 with location + scale: residual sd per year group from training controls (pooled sd if < 3 controls in the group)."""
    des = r4.DESIGN["I2"]; c = tr[Y[tr] == 0]; B = np.linalg.lstsq(des[c], BP[c], rcond=None)[0]; res = BP - des @ B
    pooled = res[c].std(0, ddof=np.linalg.matrix_rank(des[c])); sd = np.tile(pooled, (len(Y), 1))
    for g in set(r4.YEAR_G):
        cg = c[r4.YEAR_G[c] == g]
        if len(cg) >= 3:
            sd[r4.YEAR_G == g] = res[cg].std(0, ddof=1)
    X = np.hstack([res / sd, AGESEX]); return ek.nested_lr().fit(X[tr], Y[tr]).predict_proba(X[te])[:, 1]


F = ek.folds(); JOBS = [(r, np.setdiff1d(np.arange(len(Y)), te), te) for r in range(10) for k in range(5)
                        for te in [np.array([POS[s] for s in F[(F.repeat == r) & (F.fold == k)].subject])]]


def run(method, des=DES):
    def one(tr, te):
        if method == "H5a":
            return h5a_predict(list(SUBJ[tr]), list(SUBJ[te]), des)
        if method == "H5b":
            return h5b_predict(tr, te)
        return r4.fit_predict(method, tr, te, 0)
    return Parallel(n_jobs=12)(delayed(one)(tr, te) for _, tr, te in JOBS)


def summarise(out):
    per = []
    for r in range(10):
        p = np.zeros(len(Y))
        for (rr, _, te), o in zip(JOBS, out):
            if rr == r:
                p[te] = o
        ci, cs = r4.calib(Y, p); per.append(dict(auc=roc_auc_score(Y, p), **r4.binm(Y, p > 0.5), brier=np.mean((p - Y) ** 2), calib_slope=cs))
    return pd.DataFrame(per).mean(), [roc_auc_score(Y[te], o) for (_, _, te), o in zip(JOBS, out)]


if __name__ == "__main__":
    first_feats = np.vstack([XS[FIRST[s]] for s in SUBJ])
    print(("PASS" if np.allclose(first_feats, BP) else "FAIL") + " first-session features equal the P3 patient table")
    print(f"sessions {len(MS)} ({int(MS.label.sum())} epilepsy / {int((1 - MS.label).sum())} non); year groups {MS.ygroup.value_counts().to_dict()}")
    res = {m: summarise(run(m)) for m in ["C1", "I2", "H5a", "H5b"]}
    print(("PASS" if abs(res["C1"][0].auc - 0.66627) < 1e-5 else "FAIL") + f" C1 AUC {res['C1'][0].auc:.5f}; " +
          ("PASS" if abs(res["I2"][0].auc - 0.71206) < 5e-5 else "FAIL") + f" I2 AUC {res['I2'][0].auc:.5f}")
    D = pd.DataFrame({m: v[0] for m, v in res.items()}).T
    print("\n== strict P3, 200 patients, threshold 0.5 =="); print((100 * D[["auc", "acc", "sens", "spec", "prec", "f1", "bac"]]).round(2).join(D[["brier", "calib_slope"]].round(3)).to_string())
    tests = [("H5a vs C1", "H5a", "C1"), ("H5a vs I2", "H5a", "I2"), ("H5b vs I2", "H5b", "I2")]
    pv = [ek.corrected_ttest(res[a][1], res[b][1]) for _, a, b in tests]; praw = np.array([p for _, p in pv]); o = np.argsort(praw); h = np.empty(3); run_ = 0
    for rank, i in enumerate(o):
        run_ = max(run_, min(1, praw[i] * (3 - rank))); h[i] = run_
    print("\n-- corrected t-test (160/40), Holm over 3:")
    for (n, _, _), (d, p), ph in zip(tests, pv, h):
        print(f"   {n:10s} dAUC {d:+.4f}  p {p:.4f}  p_Holm {ph:.4f}")
    D.to_csv(f"{R}/round5_results.csv")
    if "nonull" not in sys.argv:
        rng = np.random.default_rng(5050); null = []
        for i in range(N_PERM):
            null.append(summarise(run("H5a", design(rng.permutation(MS.ygroup.values))))[0].auc)
        null = np.array(null); np.save(f"{R}/round5_null.npy", null)
        print(f"\nnegative control H5a (session year groups permuted {N_PERM}x): mean {null.mean():.4f}, sd {null.std():.4f}, 95th pct {np.percentile(null, 95):.4f}, "
              f"max {null.max():.4f}; real {res['H5a'][0].auc:.4f}, p = {(1 + (null >= res['H5a'][0].auc).sum()) / (1 + N_PERM):.3f}")
