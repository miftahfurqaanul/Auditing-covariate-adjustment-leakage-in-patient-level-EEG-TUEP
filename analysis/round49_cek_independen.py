#!/usr/bin/env python3
"""Round 49: independent check (second implementation) of the output in hasil49/. Written after the run; does not change the raw output.
Groups: (1) 49A strict nesting of I2 again with a second implementation; (2) 49A/49C per-fold AUC with a self-written Mann-Whitney; (3) 49C NB test and Holm explicit;
(4) 49B decision frequencies with the second code; (5) 49B re-replication = round 29; (6) 49E strata-weighted AUC with brute-force pairs; (7) 49F kurtosis with manual moments;
(8) strict NM: fold completeness and test patients; (9) 49D primary test with an explicit formula.
Usage: CUDA_VISIBLE_DEVICES="" python round49_cek_independen.py"""
import os, sys, json, glob
os.environ["CUDA_VISIBLE_DEVICES"] = ""; os.environ.setdefault("OMP_NUM_THREADS", "1")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, "/home/miftah_f/RTX-5090/tuep_fresh/code")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy import stats
OUT = os.path.join(HERE, "hasil49"); RES = []


def grp(name, ok, detail=""):
    RES.append((name, bool(ok), detail)); print(("OK  " if ok else "FAIL ") + name + (f": {detail}" if detail else ""), flush=True)


def mw_auc(y, s):
    """Mann-Whitney AUC with average ranks for ties (own implementation)."""
    y = np.asarray(y); s = np.asarray(s, float); o = np.argsort(s, kind="mergesort"); r = np.empty(len(s)); i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[o[j + 1]] == s[o[i]]: j += 1
        r[o[i:j + 1]] = (i + j) / 2 + 1; i = j + 1
    n1 = y.sum(); n0 = len(y) - n1; return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


import round28_analisis as A
J, Y, N = A.J, A.Y, A.N
K = np.load(f"{OUT}/A_prediksi_ketat.npz")

# (1) strict nesting I2, second implementation: explicit loop, OLS via pseudo-inverse, manual standardization
from sklearn.linear_model import LogisticRegression
import evalkit as ek
D, BP, AS, CS = A.D_I2, A.BP, A.c17.AS, ek.CS
def feats(fit):
    B = np.linalg.pinv(D[fit]) @ BP[fit]; res = BP[fit] - D[fit] @ B; sd = res.std(0, ddof=np.linalg.matrix_rank(D[fit])); Z = np.hstack([(BP - D @ B) / sd, AS])
    mu, sg = Z[fit].mean(0), Z[fit].std(0); sg[sg == 0] = 1; return (Z - mu) / sg
def strict_i2(tr, te):
    trs = np.asarray(tr); yy = Y[trs]; folds = np.zeros(len(trs), int)          # same order as the StratifiedKFold call in the main code
    for c in (0, 1):                                                                        # StratifiedKFold(5) without shuffling: class indices split sequentially
        idx = np.where(yy == c)[0]; sizes = np.full(5, len(idx) // 5); sizes[: len(idx) % 5] += 1; st = np.r_[0, np.cumsum(sizes)]
        for k in range(5): folds[idx[st[k]:st[k + 1]]] = k
    sc = np.zeros(len(CS))
    for k in range(5):
        a, b = trs[folds != k], trs[folds == k]; G = feats(a); est = LogisticRegression(class_weight="balanced", max_iter=5000, warm_start=True)
        for i, c in enumerate(CS):
            sc[i] += mw_auc(Y[b], est.set_params(C=c).fit(G[a], Y[a]).decision_function(G[b]))
    G = feats(trs); return LogisticRegression(C=CS[int(np.argmax(sc))], class_weight="balanced", max_iter=5000).fit(G[trs], Y[trs]).predict_proba(G[te])[:, 1]
mx = 0
for f in range(3):
    r, k, tr, te = J[f]; mx = max(mx, np.abs(strict_i2(tr, te) - K["I2"][r, te]).max())
grp("1 49A: strict nesting of I2 repeated (3 folds, second implementation)", mx < 5e-3, f"max prediction difference {mx:.2e}")

# (2) per-fold AUC with own Mann-Whitney vs summary
Ar = pd.read_csv(f"{OUT}/ringkas_A_ketat_vs_asli.csv").set_index("model"); Cr = pd.read_csv(f"{OUT}/ringkas_C_auc.csv").set_index("model")
fold = lambda P: np.array([mw_auc(Y[te], P[r, te]) for r, _, _, te in J])
d2 = max(abs(fold(K[m]).mean() - Ar.loc[m, "ketat_per_lipatan"]) for m in ("I2", "P3", "C1", "Floor")) + max(abs(fold(K[m]).mean() - Cr.loc[m, "per_lipatan"]) for m in ("N0", "N1", "N2"))
grp("2 49A/49C: per-fold AUC (own Mann-Whitney)", d2 < 1e-9, f"max difference {d2:.1e}")

# (3) explicit NB test and Holm
Ck = pd.read_csv(f"{OUT}/ringkas_C_kontras.csv").set_index("kontras"); ok3 = True
ps = []
for a in ("N1", "N2"):
    d = fold(K[a]) - fold(K["N0"]); se = np.sqrt(np.var(d, ddof=1) * (1 / 50 + 40 / 160)); t = d.mean() / se; p = 2 * (1 - stats.t.cdf(abs(t), 49)); ps.append(p)
    ok3 &= abs(d.mean() - Ck.loc[f"{a} - N0", "selisih"]) < 1e-12 and abs(p - Ck.loc[f"{a} - N0", "p"]) < 1e-9
hl = sorted(ps); h = [min(1, max(2 * hl[0], hl[1]))] * 2 if False else [min(1.0, max(2 * min(ps), max(ps)))] * 2
ok3 &= abs(h[0] - Ck.loc["N1 - N0", "holm"]) < 1e-9
grp("3 49C: NB t-test and Holm (explicit formula)", ok3, f"p {ps[0]:.4f}, {ps[1]:.4f}; Holm {h[0]:.4f}")

# (4) decision frequencies with the second code (explicit loop, different RNG)
F = pd.read_csv(f"{OUT}/B_frekuensi.csv"); rng = np.random.default_rng(777); ok4 = True; det = []
pools = {"r29 C1 H1 (gabungan)": pd.read_csv(os.path.join(HERE, "hasil29/A_kebocoran_replikasi.csv")).query("scenario == 'H1_all'").C1.values,
         "r29 V0 H1 (gabungan)": pd.read_csv(os.path.join(HERE, "hasil29/A_kebocoran_replikasi.csv")).query("scenario == 'H1_all'").V0.values}
for name, x in pools.items():
    dec = []
    for _ in range(20000):
        s_ = x[rng.integers(0, len(x), 20)]; m, se = s_.mean(), s_.std(ddof=1) / np.sqrt(20)
        dec.append(0 if (m <= 0.52 and m + 1.645 * se <= 0.54) else (2 if m > 0.55 else 1))
    fq = np.bincount(dec, minlength=3) / len(dec); ref = F[(F.kumpulan == name) & (F.n == 20)].iloc[0]
    ok4 &= abs(fq[0] - ref.p_tidak_ada_bukti) < 0.01 and abs(fq[2] - ref.p_kebocoran) < 0.01; det.append(f"{name}: {fq.round(3).tolist()}")
grp("4 49B: decision frequencies (second code, 20,000 draws)", ok4, "; ".join(det))

# (5) re-replication of C1/V4 = round 29
z = np.load(f"{OUT}/B_pasien_prediksi.npz"); a29 = pd.read_csv(os.path.join(HERE, "hasil29/A_kebocoran_replikasi.csv")).query("scenario == 'H1_all'").sort_values("rep")
d5 = max(abs(np.mean([mw_auc(Y, p) for p in z[f"{v}_{r}"]]) - a29[v].values[r]) for v in ("C1", "V4") for r in range(100))
grp("5 49B: 100 replicates of C1 and V4 = round 29 (own Mann-Whitney)", d5 < 1e-9, f"max difference {d5:.1e}")

# (6) AUC berbobot strata, brute force
import round41_strata_kaliper as R41
E = pd.read_csv(f"{OUT}/E_strata_bobot.csv"); S = R41.MEAN[:, R41.MODELS.index("F2_adj")]; G = R41.STR["year x device"]; age = R41.AGE; ok6 = True
for k in (5.0, 10.0, 15.0):
    per = []; allp = []; nolarge = []
    for c in np.unique(G):
        ee = np.where((G == c) & (Y == 1))[0]; nn = np.where((G == c) & (Y == 0))[0]; v = []
        for i in ee:
            for j in nn:
                if abs(age[i] - age[j]) <= k: v.append(1.0 if S[i] > S[j] else (0.5 if S[i] == S[j] else 0.0))
        if v: per.append(np.mean(v)); allp += v; nolarge += v if c != "2012|X" else []
    q = E[(E.k == k) & (E.model == "F2_adj")].set_index("pembobotan").auc
    ok6 &= abs(np.mean(per) - q["bobot sama per sel"]) < 1e-12 and abs(np.mean(allp) - q["bobot pasangan (asli)"]) < 1e-12 and abs(np.mean(nolarge) - q["tanpa sel 2012|X"]) < 1e-12
grp("6 49E: AUC berbobot strata (pasangan brute force)", ok6)

# (7) NM kurtosis with manual moments
import normmamba as nm
X = nm.XSEG.astype(np.float64); m = X.mean(0); c = X - m; m2 = (c ** 2).mean(0); m3 = (c ** 3).mean(0); m4 = (c ** 4).mean(0); kur = m4 / m2 ** 2 - 3; sk = m3 / m2 ** 1.5
Fk = pd.read_csv(f"{OUT}/F_kurtosis.csv"); q = Fk[Fk.fitur == "NM"].iloc[0]
grp("7 49F: kurtosis/skew NM (momen manual)", abs(np.median(kur) - q.kurtosis_median) < 1e-6 and abs(kur.max() - q.kurtosis_maks) < 1e-6 and abs(np.median(sk) - q.skew_median) < 1e-6, f"{np.median(kur):.4f}, {kur.max():.4f}, {np.median(sk):.4f}")

# (8) strict NM: completeness and test patients
fs = sorted(glob.glob(f"{OUT}/nm_ketat/r*_k*.json"))
if len(fs) == 50:
    import round4 as r4
    ok8 = True
    for r, k, tr, te in J:
        d = json.load(open(f"{OUT}/nm_ketat/r{r}_k{k}.json")); ok8 &= d["variant"] == "V4-ketat" and set(d["test"]) == set(r4.SUBJ[te]) and abs(mw_auc(Y[te], np.array([d["test"][s] for s in r4.SUBJ[te]])) - d["test_auc"]) < 1e-9
    grp("8 strict NM: 50 complete folds, test patients = canonical fold, test AUC recomputed", ok8)
else:
    grp("8 strict NM: not complete", False, f"{len(fs)} of 50 files")

# (9) 49D primary test with an explicit formula
pD = f"{OUT}/D_replikasi.csv"
if os.path.exists(f"{OUT}/D_SELESAI"):
    Dd = pd.read_csv(pD); L = Dd[(Dd.dist == "lognormal") & (Dd.m == 50)]; a1 = (L[L.icc == 0.95].RFwin_A - L[L.icc == 0.95].RFwin_none).values; a0 = (L[L.icc == 0.0].RFwin_A - L[L.icc == 0.0].RFwin_none).values
    v1, v0 = a1.var(ddof=1) / len(a1), a0.var(ddof=1) / len(a0); t1 = (a1.mean() - a0.mean()) / np.sqrt(v1 + v0); df = (v1 + v0) ** 2 / (v1 ** 2 / (len(a1) - 1) + v0 ** 2 / (len(a0) - 1)); p1 = 1 - stats.t.cdf(t1, df)
    q0 = L[L.icc == 0.0].sort_values("rep"); dd = (q0.RFwin_A - q0.RFwin_none) - (q0.RFpat_C - q0.RFpat_none); t2 = dd.mean() / (dd.std(ddof=1) / np.sqrt(len(dd))); p2 = 1 - stats.t.cdf(t2, len(dd) - 1)
    Pr = pd.read_csv(f"{OUT}/ringkas_D_primer.csv"); ok9 = abs(Pr.p.iloc[0] - p1) < 1e-9 and abs(Pr.p.iloc[1] - p2) < 1e-9 and len(Dd) == 720 and not Dd.duplicated(["icc", "m", "dist", "rep"]).any()
    grp("9 49D: 720 unique replicates; explicit D1 (Welch) and D2 (paired) tests", ok9, f"p {p1:.4g}, {p2:.4g}")
else:
    grp("9 49D: not finished", False)

n_ok = sum(ok for _, ok, _ in RES)
print(f"\n{n_ok} of {len(RES)} groups passed" + (f"; ALL {len(RES)} CHECK GROUPS PASSED" if n_ok == len(RES) else ""))
