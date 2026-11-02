"""Round 15 (reports/round15_prediction.md, commit c34da6d): preprocessing multiverse (point 2), nonlinear age in the normative model (point 4),
Riemannian/CNN baselines in direct tests (point 7), power analysis (point 8). CPU only."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ["CUDA_VISIBLE_DEVICES"] = ""
import json, itertools, warnings, numpy as np, pandas as pd, h5py
from multiprocessing import Pool
from joblib import Parallel, delayed
from scipy import stats
from scipy.signal import welch
from sklearn.metrics import roc_auc_score
import evalkit as ek, round4 as r4, round9 as r9, round10 as r10, round13 as r13, features as fe
warnings.filterwarnings("ignore"); R = "/home/miftah_f/RTX-5090/tuep_fresh/reports"
SUBJ, Y, AS, N, DES, YG = r4.SUBJ, r4.Y, r4.AGESEX, len(r4.Y), r4.DESIGN["I2"], r4.YEAR_G
JOBS = r13.JOBS; CI_ = {c: i for i, c in enumerate(fe.CH)}
DB = ["FP1-F7", "F7-T3", "T3-T5", "T5-O1", "FP2-F8", "F8-T4", "T4-T6", "T6-O2", "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
      "FP2-F4", "F4-C4", "C4-P4", "P4-O2", "FZ-CZ", "CZ-PZ"]
BIP = np.array([[CI_[p.split("-")[0]], CI_[p.split("-")[1]]] for p in DB])


def feats(w):
    """features.extract formula for windows w (n, C, T): log relative power 5 bands x C + log total power x C."""
    f_, P = welch(w, fs=fe.FS, nperseg=400, axis=2); tot = P[:, :, (f_ >= 1) & (f_ < 45)].sum(2) + 1e-12
    rel = [np.log(P[:, :, (f_ >= lo) & (f_ < hi)].sum(2) / tot + 1e-12) for lo, hi in fe.BANDS.values()]
    return np.concatenate(rel + [np.log(tot)], axis=1).astype(np.float32)


def patient_windows(s, paths):
    out = {}
    for L in (10, 4):
        n_ = L * fe.FS
        for ref in ("CAR", "bipolar"):
            F_, ptp = [], []
            for h in paths:
                with h5py.File(f"{fe.D}/{h}") as f:
                    x = f["eeg"][()].astype(np.float32)
                x = x[:, fe.EDGE:x.shape[1] - fe.EDGE]
                if ref == "bipolar":
                    x = x[BIP[:, 0]] - x[BIP[:, 1]]
                k = x.shape[1] // n_
                if k == 0:
                    continue
                w = x[:, :k * n_].reshape(len(x), k, n_).transpose(1, 0, 2); F_.append(feats(w)); ptp.append((w.max(2) - w.min(2)).max(1))
            out[(L, ref)] = (np.vstack(F_), np.concatenate(ptp))
    return s, out


def aggregate(W, L, ref, thr, how):
    rows, fallback = [], 0
    for s in SUBJ:
        F_, p = W[s][(L, ref)]; keep = p <= thr if np.isfinite(thr) else np.ones(len(p), bool)
        if not keep.any():
            keep = np.ones(len(p), bool); fallback += 1
        rows.append(np.median(F_[keep], 0) if how == "median" else F_[keep].mean(0))
    return np.vstack(rows), fallback


def rcs3(age, knots):
    k1, k2, k3 = knots; pos = lambda v: np.maximum(v, 0) ** 3
    return (pos(age - k1) - pos(age - k2) * (k3 - k1) / (k3 - k2) + pos(age - k3) * (k2 - k1) / (k3 - k2)) / (k3 - k1) ** 2


def fit(method, tr, te, X):
    c = tr[Y[tr] == 0]
    if method == "C1":
        F_ = np.hstack([X, AS])
    elif method == "I2":
        F_ = np.hstack([r13.ols_z(X, DES, c), AS])
    elif method == "I2s":
        kn = np.percentile(r4.AGE[c], [10, 50, 90]); D = np.column_stack([DES[:, :2], rcs3(r4.AGE, kn), DES[:, 2:]])
        F_ = np.hstack([r13.ols_z(X, D, c), AS])
    else:                                                                                  # "LR": features as given (+ age/sex)
        F_ = np.hstack([X, AS])
    return ek.nested_lr().fit(F_[tr], Y[tr]).predict_proba(F_[te])[:, 1]


def run(method, X):
    out = Parallel(n_jobs=12)(delayed(fit)(method, tr, te, X) for _, _, tr, te in JOBS); P = np.zeros((10, N))
    for (r, _, _, te), o in zip(JOBS, out):
        P[r, te] = o
    return P, np.array([roc_auc_score(Y[te], o) for (_, _, _, te), o in zip(JOBS, out)])


pooled = lambda P: np.mean([roc_auc_score(Y, p) for p in P])


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); h = np.empty(len(p)); run_ = 0
    for k, i in enumerate(o):
        run_ = max(run_, min(1, p[i] * (len(p) - k))); h[i] = run_
    return h


if __name__ == "__main__":
    _, Wd, _ = r13.window_table()                                                         # recording order per patient (h5 paths below)
    rec = pd.read_csv(f"{ek.D}/features_bandpower_recordings.csv"); rec = rec[rec.n_clean > 0]
    first = rec[rec.session_no == rec.groupby("subject").session_no.transform("min")]
    paths = {s: sorted(first[first.subject == s].h5_path) for s in SUBJ}
    with Pool(12) as pool:
        W = dict(pool.starmap(patient_windows, [(s, paths[s]) for s in SUBJ]))
    base, fb = aggregate(W, 10, "CAR", 800.0, "median")
    print(("PASS" if np.array_equal(base, r4.BP) else ("PASS (allclose 1e-5)" if np.allclose(base, r4.BP, atol=1e-5) else "FAIL"))
          + f" base specification (CAR, 10 s, 800 uV, median) reproduces the P3 band-power table; fallback patients {fb}")
    # ---------------- point 2: multiverse
    specs = list(itertools.product(["CAR", "bipolar"], [10, 4], [800.0, 400.0, np.inf], ["median", "mean"])); rows = []; PI2 = {}
    for ref, L, thr, how in specs:
        X, fb = aggregate(W, L, ref, thr, how); Pi, fai = run("I2", X); Pc, fac = run("C1", X)
        wy = r9.strat_auc(Y, Pi.mean(0), YG); rows.append(dict(reference=ref, window_s=L, ptp_uV=thr, aggregation=how, fallback=fb, auc_I2=pooled(Pi), auc_C1=pooled(Pc),
                                                                 dAUC=pooled(Pi) - pooled(Pc), within_year_I2=wy))
        if (ref, L, thr, how) == ("CAR", 10, 800.0, "median"):
            PI2, PC1, FA_I2, FA_C1 = Pi, Pc, fai, fac
        print(f"   {ref:7s} {L:2d}s ptp {thr:>5} {how:6s} (fallback {fb}): I2 {rows[-1]['auc_I2']:.4f}  C1 {rows[-1]['auc_C1']:.4f}  d {rows[-1]['dAUC']:+.4f}  I2 within-year {wy:.3f}", flush=True)
    MV = pd.DataFrame(rows); MV.to_csv(f"{R}/round15_multiverse.csv", index=False)
    print(("PASS" if abs(pooled(PI2) - 0.71210) < 1e-5 else "FAIL") + f" base specification I2 AUC {pooled(PI2):.5f}")
    npos = int((MV.dAUC > 0).sum()); print(f"\n== multiverse: I2 AUC {MV.auc_I2.min():.3f}-{MV.auc_I2.max():.3f} (median {MV.auc_I2.median():.3f}); "
                                           f"C1 {MV.auc_C1.min():.3f}-{MV.auc_C1.max():.3f}; I2 > C1 in {npos}/24 (sign test p {stats.binomtest(npos, 24).pvalue:.4f}); "
                                           f"I2 within-year {MV.within_year_I2.min():.3f}-{MV.within_year_I2.max():.3f}")
    print("   by factor (median I2 AUC): " + "; ".join(f"{c}: " + ", ".join(f"{k}={v:.3f}" for k, v in MV.groupby(c).auc_I2.median().items()) for c in ["reference", "window_s", "ptp_uV", "aggregation"]))
    # ---------------- point 4: nonlinear age
    Ps, fas = run("I2s", r4.BP); d4, p4 = ek.corrected_ttest(fas, FA_I2)
    print(f"\n== point 4: I2 with restricted-cubic-spline age (3 knots): AUC {pooled(Ps):.4f} vs I2 {pooled(PI2):.4f}; d {d4:+.4f}, corrected p {p4:.3f}; "
          f"within-year {r9.strat_auc(Y, Ps.mean(0), YG):.3f}")
    # ---------------- point 7: Riemannian and CNN baselines
    z = np.load(f"{ek.D}/riemann_p3.npz", allow_pickle=True); assert list(z["subjects"]) == list(SUBJ)
    Pr, far = run("LR", z["X"].astype(np.float64)); base7 = {"C2 Riemannian (+age/sex)": (Pr, far)}
    for arch in ["shallow", "deep4"]:
        d = json.load(open(f"{ek.D}/cnn_p3_{arch}.json")); P = np.zeros((10, N)); fa = []
        for r, k, tr, te in JOBS:
            pr = np.array([d[f"{r}-{k}"][s] for s in SUBJ[te]]); P[r, te] = pr; fa.append(roc_auc_score(Y[te], pr))
        base7[f"{'ShallowNet' if arch == 'shallow' else 'Deep4'} (braindecode, pre-git)"] = (P, np.array(fa))
    print("\n== point 7: I2 vs classic from-scratch baselines (Holm over 3) ==")
    tests = [(n, *ek.corrected_ttest(FA_I2, v[1])) for n, v in base7.items()]; h = holm([t[2] for t in tests])
    rows7 = []
    for (n, dd, pp), hh in zip(tests, h):
        Pm = base7[n][0]; sm = pd.DataFrame([dict(auc=roc_auc_score(Y, p), **r4.binm(Y, p > 0.5)) for p in Pm]).mean()
        rows7.append(dict(method=n, **sm.to_dict(), dAUC_I2_minus=dd, p=pp, p_holm=hh))
        print(f"   {n:38s} AUC {sm.auc:.4f} acc {sm.acc:.3f} sens {sm.sens:.3f} spec {sm.spec:.3f} bac {sm.bac:.3f} | I2 - it: {dd:+.4f}, p {pp:.4f}, Holm {hh:.4f}")
    pd.DataFrame(rows7).to_csv(f"{R}/round15_baselines.csv", index=False)
    print("   cnn_results.txt (for the reproduction check):"); print("   " + open(f"{R}/cnn_results.txt").read().replace("\n", "\n   "))
    # ---------------- point 8: power
    rng = np.random.default_rng(15000); mi, mc = PI2.mean(0), PC1.mean(0); dp, dw = [], []
    for _ in range(2000):
        b = rng.integers(0, N, N)
        if len(np.unique(Y[b])) == 2:
            dp.append(roc_auc_score(Y[b], mi[b]) - roc_auc_score(Y[b], mc[b])); dw.append(r9.strat_auc(Y[b], mi[b], YG[b]) - r9.strat_auc(Y[b], mc[b], YG[b]))
    se_p, se_w = np.std(dp, ddof=1), np.std(dw, ddof=1)
    obs_p = roc_auc_score(Y, mi) - roc_auc_score(Y, mc); obs_w = r9.strat_auc(Y, mi, YG) - r9.strat_auc(Y, mc, YG)
    nreq = lambda delta, se: 200 * ((1.96 + 0.8416) * se / delta) ** 2
    print("\n== point 8: power (approximate; SE scales as 1/sqrt(N)) ==")
    for name, delta, se in [("pooled, observed", obs_p, se_p), ("pooled, selection-corrected (nested - C1 = 0.028)", 0.6946 - 0.66627, se_p), ("within-year, observed", obs_w, se_w)]:
        print(f"   {name:48s}: delta {delta:+.4f}, SE at N=200 {se:.4f}, power at N=200 {stats.norm.cdf(abs(delta) / se - 1.96):.2f}, N for 80% power ~ {nreq(abs(delta), se):,.0f}")
