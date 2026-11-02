# Independent double check of round 27 (CPU; separate code paths).
import os; os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys; sys.path.insert(0, "/home/miftah_f/RTX-5090/tuep_fresh/code")
import numpy as np, pandas as pd, json, h5py
from scipy import stats
from sklearn.metrics import roc_auc_score
from statsmodels.stats.multitest import multipletests
D = "/home/miftah_f/RTX-5090/tuep_fresh/data"; R = "/home/miftah_f/RTX-5090/tuep_fresh/reports"; FAIL = []
def chk(ok, m): print(("PASS " if ok else "FAIL ") + m); (None if ok else FAIL.append(m))
F = pd.read_csv(f"{D}/folds_patient.csv"); T = np.load(f"{D}/p3_patient_table.npz", allow_pickle=True); S = list(T["subjects"]); y = T["y"]; pos = {s: i for i, s in enumerate(S)}
folds = [(r, k, np.array([pos[s] for s in F[(F.repeat == r) & (F.fold == k)].subject])) for r in range(10) for k in range(5)]
fa = lambda P: np.array([roc_auc_score(y[te], P[r, te]) for r, k, te in folds])
print("== A1 registry")
G = pd.read_csv(f"{R}/round27_registry.csv")
chk(np.allclose(multipletests(G.p, method="holm")[1], G.p_holm_global) and np.allclose(multipletests(G.p, method="fdr_bh")[1], G.q_bh_global), f"global Holm and BH recomputed with statsmodels ({len(G)} tests)")
bad = []
for fam, g in G.groupby("family"):
    if fam.startswith(("r21 within", "r12 Kural", "r22 A", "r22 B", "r23 calibration")):   # families whose reported Holm used a different/bootstrap or wider set
        continue
    rec = multipletests(g.p.values, method="holm")[1]
    if not np.allclose(rec, g.p_holm_family.values, atol=2e-3): bad.append(fam)
chk(not bad, f"per-family Holm recomputed from the raw p of each family matches the reported Holm (families checked: {G.family.nunique() - 5}) {bad}")
print("== A2 nested selection re-implemented")
I2 = np.load(f"{D}/round17/p1.npz")["I2"]; P3 = np.load(f"{D}/round17/p3.npz")["P3"]; NM = np.full((10, 200), np.nan)
for r, k, te in folds:
    d = json.load(open(f"{D}/round16/r{r}_k{k}.json"))["F"]["test"]; NM[r, te] = [d[S[i]] for i in te]
OUTER = {"I2": I2, "P3": P3, "NM": NM}; inner = np.load(f"{D}/round19/cpu_inner.npz"); nested = np.full((10, 200), np.nan)
cands = [["I2"], ["P3"], ["NM"], ["I2", "P3"], ["I2", "NM"], ["P3", "NM"], ["I2", "P3", "NM"]]
for f, (r, k, te) in enumerate(folds):
    tr = np.setdiff1d(np.arange(200), te); oo = json.load(open(f"{D}/round19/nm_inner_r{r}_k{k}.json"))["oof"]
    IN = {"I2": inner["I2"][f][tr], "P3": inner["P3"][f][tr], "NM": np.array([oo[S[i]] for i in tr])}
    scores = [roc_auc_score(y[tr], np.mean([IN[m] for m in c], 0)) for c in cands]
    best = sorted(range(7), key=lambda i: (-round(scores[i], 12), len(cands[i]), i))[0]
    nested[r, te] = np.mean([OUTER[m][r, te] for m in cands[best]], 0)
Q = np.load(f"{D}/round27/preds.npz")
chk(np.allclose(nested, Q["F2_nested"]), f"nested predictions identical (pooled {np.mean([roc_auc_score(y, p) for p in nested]):.4f}, per-fold mean {fa(nested).mean():.4f})")
LB = np.full((10, 200), np.nan); CB = np.full((10, 200), np.nan)
for r, k, te in folds:
    for P_, m_ in ((LB, "labram"), (CB, "cbramod")):
        d = json.load(open(f"{D}/round25/{m_}/r{r}_k{k}.json"))["test"]; P_[r, te] = [d[S[i]] for i in te]
for nm_, P_ in (("LaBraM-FT", LB), ("CBraMod-FT", CB)):
    dd = fa(nested) - fa(P_); t = dd.mean() / np.sqrt(dd.var(ddof=1) * (1 / 50 + 40 / 160)); print(f"   descriptive: F2-nested - {nm_}: {dd.mean():+.4f}, p {2 * stats.t.sf(abs(t), 49):.4f}, folds better {(dd > 0).sum()}/50")
print("== A3 one replicate re-generated independently")
import round4 as r4
Dm = r4.DESIGN["I2"]; c = np.where(y == 0)[0]; B = np.linalg.lstsq(Dm[c], T["bandpower"][c], rcond=None)[0]; Rs = T["bandpower"][c] - Dm[c] @ B
X = Dm @ B + Rs[np.random.default_rng(27300).integers(0, 100, 200)]
import evalkit as ek
P = np.full((10, 200), np.nan)
for r, k, te in folds:
    tr = np.setdiff1d(np.arange(200), te); cc = tr[y[tr] == 0]; Bf = np.linalg.lstsq(Dm[cc], X[cc], rcond=None)[0]
    z = (X - Dm @ Bf) / (X[cc] - Dm[cc] @ Bf).std(0, ddof=np.linalg.matrix_rank(Dm[cc])); P[r, te] = ek.nested_lr().fit(z[tr], y[tr]).predict_proba(z[te])[:, 1]
sims = np.load(f"{D}/round27/A3_sim.npz")["sims"]; a = np.mean([roc_auc_score(y, p) for p in P])
chk(abs(a - sims[0, 1]) < 1e-9, f"replicate 0, I2-sim without age/sex: {a:.4f} (saved {sims[0, 1]:.4f}); mean over 100 = {sims[:, 1].mean():.4f}")
print("== A6 transient index re-computed with an explicit formula (one recording)")
import features as fe
rec = pd.read_csv(f"{D}/features_bandpower_recordings.csv"); rec = rec[rec.n_clean > 0].reset_index(drop=True)
first = (rec.groupby("subject").session_no.transform("min") == rec.session_no).values; h = rec[first].h5_path.iloc[3]
with h5py.File(f"{fe.D}/{h}") as f:
    x = f["eeg"][()].astype(np.float64)
x = x[:, fe.EDGE:x.shape[1] - fe.EDGE]; n = x.shape[1] // fe.WIN; Kx = []
for j in range(n):
    w = x[:, j * fe.WIN:(j + 1) * fe.WIN]
    if (w.max(1) - w.min(1)).max() > fe.PTP_MAX: continue
    ks = []
    for ch in w:
        m_ = ch - ch.mean(); v = (m_ ** 2).mean()
        if v > 0: ks.append((m_ ** 4).mean() / v ** 2 - 3)
    Kx.append(max(ks) if ks else 0.0)
z = np.load(f"{D}/features_bandpower_windows.npz", allow_pickle=True); posr = {hh: i for i, hh in enumerate(rec.h5_path)}
wr = np.array([posr.get(hh, -1) for hh in z["h5_path"]]); wr = wr[wr >= 0]; Ksaved = Q["K"][wr == posr[h]]
chk(np.allclose(np.array(Kx), Ksaved, atol=1e-4), f"{h.split('/')[-1]}: {len(Kx)} windows, max |diff| {np.abs(np.array(Kx) - Ksaved).max():.1e}")
print("== A4 per-fold CI recomputed")
A4 = pd.read_csv(f"{R}/round27_A4_metrics.csv").set_index("model")
for nm_, P_ in (("I2", I2), ("F2", (I2 + P3 + NM) / 3)):
    d = fa(P_); se = np.sqrt(d.var(ddof=1) * (1 / 50 + 40 / 160)); lo, hi = d.mean() - stats.t.ppf(0.975, 49) * se, d.mean() + stats.t.ppf(0.975, 49) * se
    chk(np.allclose([d.mean(), lo, hi], A4.loc[nm_, ["fold_mean", "ci_lo", "ci_hi"]].values), f"{nm_}: {d.mean():.4f} [{lo:.4f}, {hi:.4f}]")
print("== A5 Cohen's d of logtot_P4 recomputed from an independent out-of-fold loop")
Zs = np.zeros((200, 114)); cnt = np.zeros(200); BPm = T["bandpower"]
for r, k, te in folds:
    tr = np.setdiff1d(np.arange(200), te); cc = tr[y[tr] == 0]; Bf = np.linalg.lstsq(Dm[cc], BPm[cc], rcond=None)[0]
    zz = (BPm - Dm @ Bf) / (BPm[cc] - Dm[cc] @ Bf).std(0, ddof=np.linalg.matrix_rank(Dm[cc])); Zs[te] += zz[te]; cnt[te] += 1
Zo = Zs / cnt[:, None]; j = list(fe.NAMES).index("logtot_P4"); a_, b_ = Zo[y == 1, j], Zo[y == 0, j]
dP4 = (a_.mean() - b_.mean()) / np.sqrt(((99) * a_.var(ddof=1) + 99 * b_.var(ddof=1)) / 198); E5 = pd.read_csv(f"{R}/round27_A5_effects.csv").set_index("feature")
chk(abs(dP4 - E5.loc["logtot_P4", "cohen_d"]) < 1e-9, f"d(logtot_P4) = {dP4:.4f}")
print("== A7a eligibility recomputed")
Z26 = np.load(f"{D}/round26/features.npz", allow_pickle=True); ses = pd.DataFrame(dict(subject=Z26["ses_subject"], era=Z26["ses_era"], first=Z26["ses_first"]))
el = set(ses[(~ses["first"]) & ses.era.isin(["<=2009", "2010", "2011", "2012", "2013"])].subject); LP = Q["I2_later_sessions"]
chk(set(np.array(S)[~np.isnan(LP[0])]) == el and len(el) == 108, f"patients with eligible later sessions: {len(el)}")
print(f"\n{'ALL CHECKS PASSED' if not FAIL else str(len(FAIL)) + ' FAILED: ' + str(FAIL)}")
