"""Round 4 (reports/round4_prediction.md, commit bff56a8): improvements of C1 with a stated basis.
I1 normative deviations (age, sex), I2 harmonised normative deviations (+ recording-year group), I2b (+ device group),
I3 C1 + Platt recalibration. Normative/harmonisation models are fitted on the NON-epilepsy training patients only. CPU only."""
import os, numpy as np, pandas as pd, warnings
os.environ.setdefault("OMP_NUM_THREADS", "1")
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import evalkit as ek, matched_design as md
warnings.filterwarnings("ignore"); R = "/home/miftah_f/RTX-5090/tuep_fresh/reports"
with np.load(f"{ek.D}/p3_patient_table.npz", allow_pickle=True) as _T:      # plain arrays (an open NpzFile cannot be sent to workers)
    SUBJ, Y, BP, AGESEX = _T["subjects"], _T["y"], _T["bandpower"], _T["agesex"]
AGE, MALE = AGESEX[:, 0], AGESEX[:, 1]
S = md.subject_table().loc[SUBJ]; assert np.allclose(S.age.values, AGE)
YEAR_G = np.where(S.year <= 2009, "<=2009", S.year.astype(int).astype(str)).astype(str)
DEV_G = np.where(S.device.isin(["X", "Intution/Ultrasom"]), S.device, "other").astype(str)


def dummies(g):
    levels = sorted(set(g)); ref = pd.Series(g).value_counts().idxmax()
    return np.column_stack([(g == l).astype(float) for l in levels if l != ref])


DESIGN = {"I1": np.column_stack([np.ones(len(Y)), AGE, MALE]),
          "I2": np.column_stack([np.ones(len(Y)), AGE, MALE, dummies(YEAR_G)]),
          "I2b": np.column_stack([np.ones(len(Y)), AGE, MALE, dummies(DEV_G)])}


def normative_z(design, tr):
    """OLS of every band-power feature on the design, fitted on the non-epilepsy training patients only; z for all patients."""
    c = tr[Y[tr] == 0]; B = np.linalg.lstsq(design[c], BP[c], rcond=None)[0]
    res = BP[c] - design[c] @ B; sd = res.std(0, ddof=np.linalg.matrix_rank(design[c]))
    return (BP - design @ B) / sd


def features(method, tr):
    if method in ("C1", "I3"):
        return np.hstack([BP, AGESEX])
    return np.hstack([normative_z(DESIGN[method], tr), AGESEX])


def fit_predict(method, tr, te, seed):
    X = features(method, tr); m = ek.nested_lr().fit(X[tr], Y[tr]); p = m.predict_proba(X[te])[:, 1]
    if method != "I3":
        return p
    oof = np.zeros(len(tr))                                              # Platt recalibration on inner out-of-fold predictions
    for a, b in StratifiedKFold(5, shuffle=True, random_state=seed).split(tr, Y[tr]):
        oof[b] = ek.nested_lr().fit(X[tr[a]], Y[tr[a]]).predict_proba(X[tr[b]])[:, 1]
    lg = lambda q: np.log(np.clip(q, 1e-6, 1 - 1e-6) / (1 - np.clip(q, 1e-6, 1 - 1e-6)))
    cal = LogisticRegression(penalty=None).fit(lg(oof)[:, None], Y[tr]); return cal.predict_proba(lg(p)[:, None])[:, 1]


def binm(y, q):
    se, sp = q[y == 1].mean(), 1 - q[y == 0].mean(); tp, fp = (q & (y == 1)).sum(), (q & (y == 0)).sum(); pr = tp / max(tp + fp, 1)
    return dict(acc=(q == y).mean(), sens=se, spec=sp, prec=pr, f1=2 * pr * se / max(pr + se, 1e-9), bac=(se + sp) / 2)


def calib(y, p):
    lg = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6))); m = LogisticRegression(penalty=None).fit(lg[:, None], y)
    return m.intercept_[0], m.coef_[0, 0]


if __name__ == "__main__":
    F = ek.folds(); pos = {s: i for i, s in enumerate(SUBJ)}
    jobs = [(r, k, np.setdiff1d(np.arange(len(Y)), te), te) for r in range(10) for k in range(5)
            for te in [np.array([pos[s] for s in F[(F.repeat == r) & (F.fold == k)].subject])]]
    # double checks
    z = normative_z(DESIGN["I2"], jobs[0][2]); c = jobs[0][2][Y[jobs[0][2]] == 0]
    print(("PASS" if np.abs(z[c].mean(0)).max() < 1e-8 else "FAIL") + f" normative z of training controls has mean 0 (max |mean| {np.abs(z[c].mean(0)).max():.1e})")
    print(f"year groups {pd.Series(YEAR_G).value_counts().to_dict()}; device groups {pd.Series(DEV_G).value_counts().to_dict()}")
    methods = ["C1", "I1", "I2", "I2b", "I3"]; preds = {}; fa = {}
    for m in methods:
        out = Parallel(n_jobs=12)(delayed(fit_predict)(m, tr, te, 100 * r + k) for r, k, tr, te in jobs); preds[m] = out
        fa[m] = [roc_auc_score(Y[te], o) for (_, _, _, te), o in zip(jobs, out)]
    rows = []
    for m in methods:
        per = []
        for r in range(10):
            p = np.zeros(len(Y))
            for (rr, _, _, te), o in zip(jobs, preds[m]):
                if rr == r:
                    p[te] = o
            ci, cs = calib(Y, p); per.append(dict(auc=roc_auc_score(Y, p), **binm(Y, p > 0.5), brier=np.mean((p - Y) ** 2), calib_int=ci, calib_slope=cs))
        rows.append(dict(method=m, **pd.DataFrame(per).mean().to_dict()))
    D = pd.DataFrame(rows).set_index("method")
    print(("PASS" if abs(D.loc["C1", "auc"] - 0.66627) < 1e-5 else "FAIL") + f" C1 reproduces the primary analysis (AUC {D.loc['C1', 'auc']:.5f})")
    print(("PASS" if max(abs(a - b) for a, b in zip(fa["I3"], fa["C1"])) < 1e-12 else "FAIL") + " recalibration leaves every fold AUC unchanged (monotone)")
    print("\n== strict P3, 200 patients, canonical folds, threshold 0.5 ==")
    print((100 * D[["auc", "acc", "sens", "spec", "prec", "f1", "bac"]]).round(2).join(D[["brier", "calib_int", "calib_slope"]].round(3)).to_string())
    tests = {m: ek.corrected_ttest(fa[m], fa["C1"]) for m in ["I1", "I2", "I2b"]}; names = list(tests); praw = np.array([tests[n][1] for n in names])
    o = np.argsort(praw); h = np.empty(3); run = 0
    for rank, i in enumerate(o):
        run = max(run, min(1, praw[i] * (3 - rank))); h[i] = run
    print("\n-- AUC vs C1 (corrected t-test 160/40, Holm over I1, I2, I2b):")
    for n, a in zip(names, h):
        print(f"   {n:4s} dAUC {tests[n][0]:+.4f}  p {tests[n][1]:.3f}  p_Holm {a:.3f}")
    D.assign(dAUC=[np.nan] + [tests.get(m, (np.nan,))[0] for m in methods[1:]]).to_csv(f"{R}/round4_results.csv")
    # temporal robustness: train <= 2011 (all patients), test 2012-2013, threshold 0.5
    print("\n== temporal (train <= 2011, test 2012-2013) ==")
    tr = np.where(S.year.values <= 2011)[0]; te = np.where(S.year.values > 2011)[0]; rng = np.random.default_rng(0)
    boots = [rng.integers(0, len(te), len(te)) for _ in range(2000)]; P = {}
    for m in ["C1", "I1", "I2b"]:
        P[m] = fit_predict(m, tr, te, 0); a = roc_auc_score(Y[te], P[m]); b = [roc_auc_score(Y[te][i], P[m][i]) for i in boots]
        print(f"   {m:4s} AUC {a:.3f} [{np.percentile(b, 2.5):.3f}-{np.percentile(b, 97.5):.3f}]  BAC {binm(Y[te], P[m] > 0.5)['bac']:.3f}")
    for m in ["I1", "I2b"]:
        d = np.array([roc_auc_score(Y[te][i], P[m][i]) - roc_auc_score(Y[te][i], P["C1"][i]) for i in boots])
        print(f"   {m} - C1: dAUC {roc_auc_score(Y[te], P[m]) - roc_auc_score(Y[te], P['C1']):+.3f}, 95% CI {np.percentile(d, 2.5):+.3f} to {np.percentile(d, 97.5):+.3f}")
