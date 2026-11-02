"""Round 13 (reports/round13_prediction.md, commit a78fe2d): montage strata and leave-one-montage-out (point 1),
fixed number of windows per patient and duration confound (point 3). CPU only."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ["CUDA_VISIBLE_DEVICES"] = ""
import warnings, numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score
import evalkit as ek, round4 as r4, round9 as r9, round10 as r10
warnings.filterwarnings("ignore"); R = "/home/miftah_f/RTX-5090/tuep_fresh/reports"
SUBJ, Y, AS, N = r4.SUBJ, r4.Y, r4.AGESEX, len(r4.Y); POS = {s: i for i, s in enumerate(SUBJ)}; F = ek.folds()
JOBS = [(r, k, np.setdiff1d(np.arange(N), te), te) for r in range(10) for k in range(5)
        for te in [np.array([POS[s] for s in F[(F.repeat == r) & (F.fold == k)].subject])]]
MONT = np.where(r4.S.montage.values == "02_tcp_le", "LE", "AR")                    # AR_A (1 patient) merged into AR, as pre-registered
YG = r4.YEAR_G
with np.load(f"{ek.D}/p3_patient_table.npz", allow_pickle=True) as _T:
    ACQ = _T["acq"]


def i2_design(tr):
    """I2 design with reference = most frequent year group among training patients; absent groups -> effect 0 (reference)."""
    present = sorted(set(YG[tr])); ref = pd.Series(YG[tr]).value_counts().idxmax()
    return np.column_stack([np.ones(N), r4.AGE, r4.MALE] + [(YG == g).astype(float) for g in present if g != ref])


def ols_z(X, D, fit):
    B = np.linalg.lstsq(D[fit], X[fit], rcond=None)[0]; sd = (X[fit] - D[fit] @ B).std(0, ddof=np.linalg.matrix_rank(D[fit])); return (X - D @ B) / sd


def fit_predict(method, tr, te, X):
    c = tr[Y[tr] == 0]
    if method == "C1":
        F_ = np.hstack([X, AS])
    elif method == "I1":
        F_ = np.hstack([ols_z(X, r4.DESIGN["I1"], c), AS])
    elif method == "I2":
        F_ = np.hstack([ols_z(X, i2_design(tr), c), AS])
    else:                                                                            # metadata floor
        F_ = ACQ
    return ek.nested_lr().fit(F_[tr], Y[tr]).predict_proba(F_[te])[:, 1]


def run(method, X):
    out = Parallel(n_jobs=12)(delayed(fit_predict)(method, tr, te, X) for _, _, tr, te in JOBS); P = np.zeros((10, N))
    for (r, _, _, te), o in zip(JOBS, out):
        P[r, te] = o
    return P, np.array([roc_auc_score(Y[te], o) for (_, _, _, te), o in zip(JOBS, out)])


def summ(P):
    return pd.DataFrame([dict(auc=roc_auc_score(Y, p), **r4.binm(Y, p > 0.5), brier=np.mean((p - Y) ** 2)) for p in P]).mean()


def boot_auc(y, p, seed, n=2000):
    rng = np.random.default_rng(seed); b = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) == 2:
            b.append(roc_auc_score(y[i], p[i]))
    return np.percentile(b, [2.5, 97.5])


def window_table():
    """First-session clean windows per patient, recordings ordered by h5_path, windows in temporal order (row order of the npz)."""
    z = np.load(f"{ek.D}/features_bandpower_windows.npz", allow_pickle=True); Xw, wh = z["X"], z["h5_path"]
    rec = pd.read_csv(f"{ek.D}/features_bandpower_recordings.csv"); rec = rec[rec.n_clean > 0]
    first = rec[rec.session_no == rec.groupby("subject").session_no.transform("min")]
    hdr = pd.read_csv(f"{ek.D}/edf_headers.csv")[["file", "duration_s"]]; first = first.merge(hdr, on="file", how="left")
    rows = {h: np.where(wh == h)[0] for h in first.h5_path}
    W = {s: np.concatenate([rows[h] for h in sorted(first[first.subject == s].h5_path)]) for s in SUBJ}
    dur = first.groupby("subject").duration_s.sum().loc[SUBJ].values
    return Xw, W, dur


if __name__ == "__main__":
    Xw, W, dur = window_table(); nwin = np.array([len(W[s]) for s in SUBJ])
    full = np.vstack([np.median(Xw[W[s]], 0) for s in SUBJ])
    print(("PASS" if np.array_equal(full, r4.BP) else ("PASS (allclose)" if np.allclose(full, r4.BP, atol=1e-6) else "FAIL")) + " rebuilding with all windows reproduces the P3 band-power table")
    print(f"montage strata: {pd.crosstab(MONT, Y).to_dict()}; windows per patient median {np.median(nwin):.0f} (min {nwin.min()}, max {nwin.max()})")
    # ---------------- reference models (full features)
    P, FA = {}, {}
    for m in ["I2", "C1", "Floor"]:
        P[m], FA[m] = run(m, r4.BP)
    s = {m: summ(P[m]).auc for m in P}
    print(("PASS" if abs(s["C1"] - 0.66627) < 1e-5 and abs(s["I2"] - 0.71210) < 1e-5 and abs(s["Floor"] - 0.6479) < 5e-4 else "FAIL") +
          f" reference AUCs C1 {s['C1']:.5f}, I2 {s['I2']:.5f} (I2 design with training-reference rule), floor {s['Floor']:.4f}")
    # ---------------- 1a within montage
    strata = {"montage": MONT, "montage x year": np.char.add(np.char.add(MONT.astype(str), "|"), YG)}
    T1 = r10.within({m: P[m] for m in ["I2", "C1", "Floor"]}, np.arange(N), strata, [("I2", "Floor"), ("I2", "C1")], 13001)
    print("\n== 1a. within-montage AUC (round10 estimator; Holm over 4) =="); print(T1.round(4).to_string(index=False))
    for g in ["AR", "LE"]:
        i = MONT == g; print(f"   AUC inside {g} only (n={i.sum()}, {Y[i].sum()} epi): " + "  ".join(f"{m} {roc_auc_score(Y[i], P[m].mean(0)[i]):.3f}" for m in P))
    # ---------------- 1b leave-one-montage-out
    print("\n== 1b. leave-one-montage-out (descriptive) ==")
    rows = []
    for held in ["LE", "AR"]:
        te = np.where(MONT == held)[0]; tr = np.where(MONT != held)[0]
        for m in ["C1", "I1", "I2"]:
            p = fit_predict(m, tr, te, r4.BP); a = roc_auc_score(Y[te], p); ci = boot_auc(Y[te], p, 13100 + len(rows)); bm = r4.binm(Y[te], p > 0.5)
            rows.append(dict(held_out=held, n_test=len(te), n_train=len(tr), method=m, auc=a, lo=ci[0], hi=ci[1], bac=bm["bac"]))
            print(f"   train {'AR' if held == 'LE' else 'LE'} -> test {held} (n={len(te)}): {m} AUC {a:.3f} [{ci[0]:.3f}-{ci[1]:.3f}], BAC {bm['bac']:.3f}")
    # ---------------- 3a fixed number of windows
    print("\n== 3a. fixed number of first clean windows ==")
    res = []
    for n in (30, 10):
        Xn = np.vstack([np.median(Xw[W[s][:n]], 0) for s in SUBJ]); short = int((nwin < n).sum())
        for m in ["C1", "I2"]:
            Pn, fan = run(m, Xn); sm = summ(Pn); d, p = ek.corrected_ttest(fan, FA[m])
            res.append(dict(n_windows=n, method=m, patients_below_n=short, **sm.to_dict(), dAUC_vs_full=d, p_vs_full=p))
            print(f"   N={n} ({short} patients with fewer): {m} AUC {sm.auc:.4f} (full {s[m]:.4f}; d {d:+.4f}, p {p:.3f}); acc {sm.acc:.3f} sens {sm.sens:.3f} spec {sm.spec:.3f} bac {sm.bac:.3f}")
            P[f"{m}_N{n}"] = Pn
        Tn = r10.within({f"I2_N{n}": P[f"I2_N{n}"], "Floor": P["Floor"]}, np.arange(N), {"year": YG}, [(f"I2_N{n}", "Floor")], 13200 + n)
        print(f"   within-year: I2_N{n} {Tn[f'auc_I2_N{n}'].iloc[0]:.3f} vs floor {Tn.auc_Floor.iloc[0]:.3f}, diff {Tn['diff'].iloc[0]:+.3f} [{Tn.lo.iloc[0]:+.3f}, {Tn.hi.iloc[0]:+.3f}], p {Tn.p.iloc[0]:.4f}")
    # ---------------- 3b duration confound
    print("\n== 3b. duration confound (no training; larger value -> epilepsy) ==")
    for name, v in [("clean windows", nwin.astype(float)), ("first-session duration (s)", dur)]:
        a = roc_auc_score(Y, v); wy = r9.strat_auc(Y, v, YG); print(f"   {name}: AUC {a:.3f} [{boot_auc(Y, v, 13300)[0]:.3f}-{boot_auc(Y, v, 13300)[1]:.3f}]; within year {wy:.3f}")
    T1.to_csv(f"{R}/round13_within_montage.csv", index=False); pd.DataFrame(rows).to_csv(f"{R}/round13_lomo.csv", index=False); pd.DataFrame(res).to_csv(f"{R}/round13_fixed_windows.csv", index=False)
