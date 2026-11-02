#!/usr/bin/env python3
"""Round 30: recomputation of secondary numbers that depended on the old normative layer (PRAREGISTRASI_putaran30.md, addendum 1). CPU only.
1. published/pretrained comparators; 2. Kural transfer; 3. deviation map (A5), transients (A6), between sessions (A7a), hard patients (A8).
Usage: CUDA_VISIBLE_DEVICES="" python round30_eval2.py [--smoke]"""
import os, sys, json, hashlib
os.environ["CUDA_VISIBLE_DEVICES"] = ""; os.environ.setdefault("OMP_NUM_THREADS", "1")
HERE = os.path.dirname(os.path.abspath(__file__)); CODE = "/home/miftah_f/RTX-5090/tuep_fresh/code"; sys.path.insert(0, HERE); sys.path.insert(0, CODE)
import numpy as np, pandas as pd
from scipy import stats
from joblib import Parallel, delayed
from multiprocessing import Pool
from sklearn.metrics import roc_auc_score
import round28_analisis as A, round29_sensitivitas as S
import round7 as r7, round12_kural as k12, round27_validity as v, round4 as r4, round10 as r10, round17_common as c17, features as fe, tang_fixed as tf

SMOKE = "--smoke" in sys.argv
OUT = os.path.join(HERE, "hasil30" if not SMOKE else "hasil30/smoke"); os.makedirs(OUT, exist_ok=True)
Y, N, J, SUBJ, ERA, BP, AS, D_I2, ek = A.Y, A.N, A.J, c17.SUBJ, A.ERA, A.BP, A.c17.AS, A.D_I2, A.ek
EV = np.load(os.path.join(HERE, "hasil30", "smoke", "eval_prediksi_v4.npz") if SMOKE else os.path.join(HERE, "hasil30", "eval_prediksi_v4.npz")); PV = {k: EV[k] for k in EV.files}
F2V4 = PV["F2_adj"]; I2V4 = PV["I2_adj"]; P3V4 = PV["P3_adj"]; NMV4 = PV["NM_adj"]


def fit_v4(X, D, tr):
    B = np.linalg.lstsq(D[tr], X[tr], rcond=None)[0]; sd = (X[tr] - D[tr] @ B).std(0, ddof=np.linalg.matrix_rank(D[tr])); return B, sd


def nz4(X, D, tr):
    B, sd = fit_v4(X, D, tr); return (X - D @ B) / sd


def lr4(X, tr, te):
    F = np.hstack([nz4(X, D_I2, tr), AS]); return ek.nested_lr().fit(F[tr], Y[tr]).predict_proba(F[te])[:, 1]


rows_all = []
# ===================== 1. pembanding =====================
z7 = np.load(f"{A.ek.D}/../reports/round7_fold_aucs.npz", allow_pickle=True) if os.path.exists(f"{A.ek.D}/../reports/round7_fold_aucs.npz") else np.load("/home/miftah_f/RTX-5090/tuep_fresh/reports/round7_fold_aucs.npz", allow_pickle=True)
J199 = r7.J199
p199 = Parallel(n_jobs=12)(delayed(lr4)(BP, tr, te) for _, _, tr, te in J199); fa199 = np.array([roc_auc_score(Y[te], p) for (_, _, _, te), p in zip(J199, p199)])
FA = {"I2-adj": c17.fold_aucs(J, I2V4), "F2-adj": c17.fold_aucs(J, F2V4), "I2-adj@199": fa199}
comp = [("I2-adj@199", "Tang-F@199", z7["Tang-F@199"]), ("I2-adj", "Rohira-F", z7["Rohira-F"]), ("I2-adj", "DNM-F", z7["DNM-F"]), ("I2-adj", "BENDR beku", z7["BENDR"]), ("I2-adj", "LaBraM beku", z7["LaBraM"]),
        ("F2-adj", "Rohira-F", z7["Rohira-F"]), ("F2-adj", "DNM-F", z7["DNM-F"]), ("F2-adj", "BENDR beku", z7["BENDR"]), ("F2-adj", "LaBraM beku", z7["LaBraM"])]
rows = []
for a, b, fb in comp:
    d, lo, hi, p = S.nb_diff(FA[a], np.asarray(fb)); rows.append(dict(test=f"{a} - {b}", diff=d, lo=lo, hi=hi, p=p, folds_better=int((FA[a] > np.asarray(fb)).sum())))
C1 = pd.DataFrame(rows); C1["q_bh"] = stats.false_discovery_control(C1.p.values)
# sizes of I2-adj@199 vs. C1@199 for context
C1.to_csv(f"{OUT}/eval2_pembanding.csv", index=False); print(C1.round(4).to_string(index=False), flush=True)
print("I2-adj@199 per fold:", round(100 * fa199.mean(), 2), "| Tang-F@199:", round(100 * np.mean(z7["Tang-F@199"]), 2), flush=True)

# ===================== 2. transfer Kural =====================
def kural_v4():
    allp = np.arange(N); B, sd = fit_v4(BP, D_I2, allp); Z = (BP - D_I2 @ B) / sd; m = ek.nested_lr().fit(np.hstack([Z, AS]), Y)
    L = k12.kural_labels(); Fk = k12.kural_features(); D = L.join(Fk); D = D[D.ok]; yk = D.y.values; Xk = np.vstack(D.feat.values); ASk = D[["age", "male"]].values
    DK = np.column_stack([np.ones(len(yk)), ASk[:, 0], ASk[:, 1], np.zeros((len(yk), D_I2.shape[1] - 3))])
    p4 = m.predict_proba(np.hstack([(Xk - DK @ B) / sd, ASk]))[:, 1]
    m_c1 = ek.nested_lr().fit(np.hstack([BP, AS]), Y); pc = m_c1.predict_proba(np.hstack([Xk, ASk]))[:, 1]
    a4, ci4, pperm = k12.auc_ci_perm(yk, p4, 30201); ac = roc_auc_score(yk, pc)
    rng = np.random.default_rng(30202); d = []
    for _ in range(2000):
        i = rng.integers(0, len(yk), len(yk))
        if len(np.unique(yk[i])) == 2:
            d.append(roc_auc_score(yk[i], p4[i]) - roc_auc_score(yk[i], pc[i]))
    d = np.array(d); pp = max(min(1, 2 * min((d <= 0).mean(), (d >= 0).mean())), 1 / 2000)
    return dict(n=len(yk), auc_I2V4=a4, lo=ci4[0], hi=ci4[1], p_perm_vs_0_5=pperm, auc_C1=ac, diff_I2V4_C1=a4 - ac, d_lo=np.percentile(d, 2.5), d_hi=np.percentile(d, 97.5), p_vs_C1=pp)
K4 = kural_v4(); pd.DataFrame([K4]).to_csv(f"{OUT}/eval2_kural.csv", index=False); print("Kural I2-adj:", {k: round(float(x), 4) for k, x in K4.items()}, flush=True)

# ===================== 3. A5, A6, A7a, A8 =====================
# --- A5: z V4 out-of-fold, Cohen d
Zs = np.zeros_like(BP, dtype=np.float64); cnt = np.zeros(N)
for r, k, tr, te in J:
    z = nz4(BP, D_I2, tr); Zs[te] += z[te]; cnt[te] += 1
Zm = Zs / cnt[:, None]; names = list(fe.NAMES)
def cohend(Zq, y):
    a_, b_ = Zq[y == 1], Zq[y == 0]; sp = np.sqrt(((len(a_) - 1) * a_.var(0, ddof=1) + (len(b_) - 1) * b_.var(0, ddof=1)) / (len(a_) + len(b_) - 2)); return (a_.mean(0) - b_.mean(0)) / sp
dd = cohend(Zm, Y); rng = np.random.default_rng(30500); ep, ne = np.where(Y == 1)[0], np.where(Y == 0)[0]; bt = []
for _ in range(2000):
    ii = np.r_[rng.choice(ep, len(ep)), rng.choice(ne, len(ne))]; bt.append(cohend(Zm[ii], Y[ii]))
bt = np.array(bt); lo, hi = np.percentile(bt, 2.5, 0), np.percentile(bt, 97.5, 0)
E5 = pd.DataFrame(dict(feature=names, cohen_d=dd, d_lo=lo, d_hi=hi)); E5["band"] = E5.feature.str.rsplit("_", n=1).str[0]; E5.to_csv(f"{OUT}/eval2_A5_efek_V4.csv", index=False)
g5 = E5.groupby("band").apply(lambda x: pd.Series(dict(mean_d=x.cohen_d.mean(), n_ci_excl0=int(((x.d_lo > 0) | (x.d_hi < 0)).sum()), n=len(x)))); g5.to_csv(f"{OUT}/eval2_A5_ringkas_pita.csv"); print(g5.round(3).to_string(), flush=True)

# --- A6: sharp transients 20% with I2-adj
Xw, wr, rec, first, fw, pos = v.first_session_windows(); fr = rec[first]
with Pool(12) as pool:
    kk = dict(pool.map(v.kurt_windows, fr.h5_path, chunksize=2))
K = np.full(len(Xw), np.nan)
for h in fr.h5_path:
    wi = np.where(wr == pos[h])[0]; K[wi] = kk[h]
X20 = v.trimmed_bp(Xw, fw, K, 0.2)
p_t = c17.to_matrix(J, Parallel(n_jobs=12)(delayed(lr4)(X20, tr, te) for _, _, tr, te in J))
F2t = (p_t + P3V4 + NMV4) / 3; fa_t, fa_i = c17.fold_aucs(J, p_t), c17.fold_aucs(J, I2V4); fa_f2t, fa_f2 = c17.fold_aucs(J, F2t), c17.fold_aucs(J, F2V4)
d1 = S.nb_diff(fa_t, fa_i); d2 = S.nb_diff(fa_f2t, fa_f2)
A6 = pd.DataFrame([dict(test="I2-adj tanpa 20% jendela tajam - I2-adj", diff=d1[0], lo=d1[1], hi=d1[2], p=d1[3]), dict(test="F2-adj (I2 tanpa 20% jendela tajam) - F2-adj", diff=d2[0], lo=d2[1], hi=d2[2], p=d2[3])]); A6.to_csv(f"{OUT}/eval2_A6_transien.csv", index=False); print(A6.round(4).to_string(index=False), flush=True)

# --- A7a: later sessions with I2-adj
Z26 = np.load(f"{A.ek.D}/round26/features.npz", allow_pickle=True)
ses = pd.DataFrame(dict(subject=Z26["ses_subject"], session=Z26["ses_session"], era=Z26["ses_era"], age=Z26["ses_age"], first=Z26["ses_first"])); ses["p"] = ses.subject.map({s: i for i, s in enumerate(SUBJ)}); SX = Z26["SX"]
COL = v.COL
def a7_v4(tr, te):
    B, sd = fit_v4(BP, D_I2, tr); F = np.hstack([(BP - D_I2 @ B) / sd, AS]); m = ek.nested_lr().fit(F[tr], Y[tr])
    tsub = set(SUBJ[te]); rows = ses[ses.subject.isin(tsub) & ~ses["first"] & ses.era.isin(["<=2009", "2010", "2011", "2012", "2013"])]
    if rows.empty:
        return {}
    pidx = rows.p.values; Dl = np.zeros((len(rows), D_I2.shape[1])); Dl[:, 0] = 1; Dl[:, 1] = rows.age.values; Dl[:, 2] = AS[pidx, 1]
    for j, e in enumerate(rows.era.values):
        if e in COL:
            Dl[j, COL[e]] = 1
    Fl = np.hstack([(SX[rows.index.values] - Dl @ B) / sd, np.column_stack([rows.age.values, AS[pidx, 1]])]); pr = m.predict_proba(Fl)[:, 1]
    return pd.Series(pr).groupby(pidx).mean().to_dict()
LP = np.full((10, N), np.nan)
for (r, k, tr, te), dct in zip(J, Parallel(n_jobs=12)(delayed(a7_v4)(tr, te) for _, _, tr, te in J)):
    for i, val in dct.items():
        LP[r, i] = val
elig = ~np.isnan(LP[0]); auc_l = np.mean([roc_auc_score(Y[elig], LP[r, elig]) for r in range(10)]); auc_f = np.mean([roc_auc_score(Y[elig], I2V4[r, elig]) for r in range(10)])
rho = stats.spearmanr(np.nanmean(LP, 0)[elig], I2V4.mean(0)[elig]).correlation
A7 = pd.DataFrame([dict(n_eligible=int(elig.sum()), n_epilepsy=int(Y[elig].sum()), n_non=int((1 - Y[elig]).sum()), auc_later=auc_l, auc_first=auc_f, spearman=rho)]); A7.to_csv(f"{OUT}/eval2_A7a_V4.csv", index=False); print(A7.round(4).to_string(index=False), flush=True)

# --- A8: hard patients with F2-adj
wrong = ((F2V4 > 0.5).astype(int) != Y[None, :]).sum(0); hard = wrong >= 8
vm = {"I2": I2V4, "P3": P3V4, "NM": NMV4}; wrong_mean = {k: ((m.mean(0) > 0.5).astype(int) != Y) for k, m in vm.items()}
universal = hard & wrong_mean["I2"] & wrong_mean["P3"] & wrong_mean["NM"]
drowsy = np.array([(Z26["win_aidx"][Z26["win_patient"] == i] < 0).mean() for i in range(N)]); phot = np.array([Z26["win_photic"][Z26["win_patient"] == i].mean() for i in range(N)])
nwin = np.bincount(c17.__dict__.get("PID", np.zeros(0, int)), minlength=N) if False else np.array([len(fw[s]) for s in SUBJ])
vars_ = dict(age=AS[:, 0], male=AS[:, 1], era_le2009=(ERA == "<=2009").astype(float), n_windows=nwin, drowsy_frac=drowsy, photic_frac=phot); rows = []
for lab, nm_ in ((1, "epilepsy"), (0, "non-epilepsy")):
    a_, b_ = universal & (Y == lab), (~universal) & (Y == lab)
    for vn, x in vars_.items():
        if vn in ("male", "era_le2009"):
            tab = [[int(x[a_].sum()), int(a_.sum() - x[a_].sum())], [int(x[b_].sum()), int(b_.sum() - x[b_].sum())]]; p = stats.fisher_exact(tab).pvalue; h_, o_ = x[a_].mean(), x[b_].mean()
        else:
            p = stats.mannwhitneyu(x[a_], x[b_]).pvalue if a_.sum() > 0 else np.nan; h_, o_ = np.median(x[a_]), np.median(x[b_])
        rows.append(dict(cls=nm_, variable=vn, n_hard=int(a_.sum()), hard=float(h_), others=float(o_), p=p))
A8 = pd.DataFrame(rows); A8["q_bh"] = stats.false_discovery_control(np.nan_to_num(A8.p.values, nan=1.0)); A8.to_csv(f"{OUT}/eval2_A8_pasien_sulit.csv", index=False)
print(f"hard patients (universal): {int(universal.sum())} (epilepsy {int((universal & (Y == 1)).sum())}, non-epilepsy {int((universal & (Y == 0)).sum())}); F2-adj wrong >= 8/10: {int(hard.sum())}", flush=True); print(A8.round(4).to_string(index=False))
json.dump(dict(smoke=SMOKE, code_sha256=hashlib.sha256(open(__file__, "rb").read()).hexdigest()), open(f"{OUT}/meta_eval2.json", "w")); print("done")
