"""Round 6 ATM evaluation (reports/round6_atm_prediction.md, commit 4a7550e + addendum b909763): DNM-F (dynamic network metrics + LR)
and DNM-M (control-based year-harmonised normative deviations of the same metrics), vs C1 and I2 on the strict P3 protocol. CPU only."""
import os, numpy as np, pandas as pd, warnings
os.environ.setdefault("OMP_NUM_THREADS", "1")
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score
import evalkit as ek, round4 as r4
warnings.filterwarnings("ignore"); R = "/home/miftah_f/RTX-5090/tuep_fresh/reports"; N_PERM = 50
Y, AGESEX = r4.Y, r4.AGESEX
with np.load(f"{ek.D}/dnm_features.npz", allow_pickle=True) as z:
    assert (z["subjects"] == r4.SUBJ).all(); XD = z["X"]
F = ek.folds(); POS = {s: i for i, s in enumerate(r4.SUBJ)}
JOBS = [(r, np.setdiff1d(np.arange(len(Y)), te), te) for r in range(10) for k in range(5)
        for te in [np.array([POS[s] for s in F[(F.repeat == r) & (F.fold == k)].subject])]]


def norm_z(X, des, tr):
    c = tr[Y[tr] == 0]; B = np.linalg.lstsq(des[c], X[c], rcond=None)[0]
    sd = (X[c] - des[c] @ B).std(0, ddof=np.linalg.matrix_rank(des[c])); return (X - des @ B) / sd


def predict(method, tr, te, des=None):
    if method in ("C1", "I2"):
        return r4.fit_predict(method, tr, te, 0)
    X = XD if method == "DNM-F" else norm_z(XD, r4.DESIGN["I2"] if des is None else des, tr)
    X = np.hstack([X, AGESEX]); return ek.nested_lr().fit(X[tr], Y[tr]).predict_proba(X[te])[:, 1]


def run(method, des=None):
    return Parallel(n_jobs=12)(delayed(predict)(method, tr, te, des) for _, tr, te in JOBS)


def summarise(out):
    per = []
    for r in range(10):
        p = np.zeros(len(Y))
        for (rr, _, te), o in zip(JOBS, out):
            if rr == r:
                p[te] = o
        per.append(dict(auc=roc_auc_score(Y, p), **r4.binm(Y, p > 0.5), brier=np.mean((p - Y) ** 2)))
    return pd.DataFrame(per).mean(), [roc_auc_score(Y[te], o) for (_, _, te), o in zip(JOBS, out)]


if __name__ == "__main__":
    print(f"DNM features {XD.shape}, finite {np.isfinite(XD).all()}")
    res = {m: summarise(run(m)) for m in ["C1", "I2", "DNM-F", "DNM-M"]}
    print(("PASS" if abs(res["C1"][0].auc - 0.66627) < 1e-5 and abs(res["I2"][0].auc - 0.71210) < 5e-5 else "FAIL") +
          f" C1 {res['C1'][0].auc:.5f} and I2 {res['I2'][0].auc:.5f} reproduced")
    D = pd.DataFrame({m: v[0] for m, v in res.items()}).T
    print("\n== strict P3, 200 patients, threshold 0.5 =="); print((100 * D[["auc", "acc", "sens", "spec", "prec", "f1", "bac"]]).round(2).join(D[["brier"]].round(3)).to_string())
    tests = [("DNM-M vs DNM-F", "DNM-M", "DNM-F"), ("DNM-M vs C1", "DNM-M", "C1"), ("DNM-M vs I2", "DNM-M", "I2"), ("DNM-F vs C1", "DNM-F", "C1")]
    pv = [ek.corrected_ttest(res[a][1], res[b][1]) for _, a, b in tests]; praw = np.array([p for _, p in pv]); o = np.argsort(praw); h = np.empty(4); run_ = 0
    for rank, i in enumerate(o):
        run_ = max(run_, min(1, praw[i] * (4 - rank))); h[i] = run_
    print("\n-- corrected t-test (160/40), Holm over 4:")
    for (n, _, _), (d, p), ph in zip(tests, pv, h):
        print(f"   {n:15s} dAUC {d:+.4f}  p {p:.4f}  p_Holm {ph:.4f}")
    D.to_csv(f"{R}/round6_results.csv")
    rng = np.random.default_rng(6060); null = []
    for i in range(N_PERM):
        g = rng.permutation(r4.YEAR_G); des = np.column_stack([np.ones(len(Y)), r4.AGE, r4.MALE, r4.dummies(g)])
        null.append(summarise(run("DNM-M", des))[0].auc)
    null = np.array(null); np.save(f"{R}/round6_null.npy", null)
    print(f"\nnegative control DNM-M (year groups permuted {N_PERM}x): mean {null.mean():.4f}, sd {null.std():.4f}, 95th pct {np.percentile(null, 95):.4f}; "
          f"real {res['DNM-M'][0].auc:.4f}, p = {(1 + (null >= res['DNM-M'][0].auc).sum()) / (1 + N_PERM):.3f}")
