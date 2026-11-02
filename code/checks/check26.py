# Independent double check of round 26 (CPU). Different code paths from round26_run.py / round26_eval.py.
import os; os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys; sys.path.insert(0, "/home/miftah_f/RTX-5090/tuep_fresh/code")
import numpy as np, pandas as pd, json, glob
from scipy import stats
from sklearn.metrics import roc_auc_score
D = "/home/miftah_f/RTX-5090/tuep_fresh/data"; FAIL = []
def chk(ok, m): print(("PASS " if ok else "FAIL ") + m); (None if ok else FAIL.append(m))
F = pd.read_csv(f"{D}/folds_patient.csv"); T = np.load(f"{D}/p3_patient_table.npz", allow_pickle=True); S = list(T["subjects"]); y = T["y"]; pos = {s: i for i, s in enumerate(S)}
folds = [(r, k, np.array([pos[s] for s in F[(F.repeat == r) & (F.fold == k)].subject])) for r in range(10) for k in range(5)]
P = np.load(f"{D}/round26/preds.npz"); I2 = np.load(f"{D}/round17/p1.npz")["I2"]
fa = lambda M: np.array([roc_auc_score(y[te], M[r, te]) for r, k, te in folds])
def nb(a, b):
    d = fa(a) - fa(b); t = d.mean() / np.sqrt(d.var(ddof=1) * (1 / 50 + 40 / 160)); return d.mean(), 2 * stats.t.sf(abs(t), 49)
print("== 1. primary tests recomputed (folds from CSV, own t-test, own Holm)")
res = [nb(P["WP"], I2), nb(P["SM"], I2)]; p = np.array([x[1] for x in res]); o = np.argsort(p); h = np.empty(2); run = 0
for j, i in enumerate(o): run = max(run, min(1, p[i] * (2 - j))); h[i] = run
prim = pd.read_csv("/home/miftah_f/RTX-5090/tuep_fresh/reports/round26_primary.csv")
chk(np.allclose([x[0] for x in res], prim.dAUC) and np.allclose(p, prim.p) and np.allclose(h, prim.p_holm), f"dAUC {np.round([x[0] for x in res], 4).tolist()}, p {np.round(p, 4).tolist()}, Holm {np.round(h, 4).tolist()}")
chk(np.array_equal(P["I2"], I2), "I2 predictions of round 26 identical to round 17")
print("== 2. E1 estimator re-implemented with explicit subject dummies (no demeaning), folds r0k0 and r7k3")
import round26_run as R26
for r, k in [(0, 0), (7, 3)]:
    te = [x[2] for x in folds if x[0] == r and x[1] == k][0]; tr = np.setdiff1d(np.arange(200), te); c = tr[y[tr] == 0]
    B = np.linalg.lstsq(R26.D_I2[c], R26.BP[c], rcond=None)[0]; delta, sup, used = R26.wp_delta(tr, B[1], R26.SX)
    Sx = R26.SES[R26.SES.p.isin(c)]; multi = Sx.groupby("subject").era.nunique(); Sx = Sx[Sx.subject.isin(multi[multi > 1].index)]
    subs = sorted(Sx.subject.unique()); eras = [e for e in R26.ERAS if e != "2012" and (Sx.era == e).any()]
    X = np.column_stack([(Sx.subject.values == s).astype(float) for s in subs] + [(Sx.era.values == e).astype(float) for e in eras])
    yy = R26.SX[Sx.index.values] - np.outer(Sx.age.values, B[1]); coef = np.linalg.lstsq(X, yy, rcond=None)[0][len(subs):]
    same = all(np.allclose(delta[e], coef[eras.index(e)], atol=1e-8) for e in used if e != "2012")
    chk(same, f"fold r{r}k{k}: within-person era effects identical with subject-dummy OLS (eras {used})")
print("== 3. E2 features recomputed from the raw window table with independent code (20 patients)")
z = np.load(f"{D}/round26/features.npz", allow_pickle=True); wp, X, ph, ai = z["win_patient"], z["win_X"], z["win_photic"], z["win_aidx"]
names = list(np.load(f"{D}/features_bandpower_windows.npz", allow_pickle=True)["names"])
ai2 = np.mean([X[:, names.index(f"rel_alpha_{c}")] - X[:, names.index(f"rel_theta_{c}")] for c in ["O1", "O2", "P3", "P4", "PZ"]], 0)
chk(np.allclose(ai, ai2), "posterior alpha index recomputed")
ok = True
for i in np.random.default_rng(0).choice(200, 20, replace=False):
    w = np.where(wp == i)[0]; w = w[~ph[w]] if (~ph[w]).any() else w
    if len(w) >= 20:
        k = int(np.ceil(len(w) / 2)); top = sorted(w, key=lambda j: (-ai2[j], j))[:k]
    else:
        top = w
    ok &= np.allclose(np.median(X[top], 0), z["BP_SM"][i])
chk(ok, "BP_SM equals the median of each patient's most-alert half of non-photic windows (20 random patients)")
print("== 4. photic blocks re-derived for 3 more files with an independent rule implementation")
import mne; mne.set_log_level("ERROR"); import h5py, features as fe
st = pd.read_csv(f"{D}/round26/photic_status.csv"); idx = pd.read_csv(f"{D}/index.csv").set_index("file")
edf = {os.path.basename(p): p for p in glob.glob("/home/miftah_f/RTX-5090/EEG-TUEP-v2.0.1/0*_*epilepsy/**/*.edf", recursive=True)}
rec = pd.read_csv(f"{D}/features_bandpower_recordings.csv"); rec = rec[rec.n_clean > 0]
zw = np.load(f"{D}/features_bandpower_windows.npz", allow_pickle=True)["h5_path"]
for _, r in st[st.photic_status == "active"].sample(3, random_state=5).iterrows():
    raw = mne.io.read_raw_edf(edf[r.file], preload=False); ch = [c for c in raw.ch_names if "PHOTIC" in c.upper()][0]; fs = raw.info["sfreq"]
    x = raw.get_data(picks=[ch])[0]; on = [t for t in range(int(len(x) // fs)) if x[int(t * fs):int((t + 1) * fs)].max() != x[int(t * fs):int((t + 1) * fs)].min()]
    lead = float(idx.loc[r.file, "calib_lead_s"])
    with h5py.File(f"{fe.D}/{r.h5_path}") as f:
        e = f["eeg"][()].astype(np.float32)
    e = e[:, fe.EDGE:e.shape[1] - fe.EDGE]; n = e.shape[1] // fe.WIN; ww = e[:, :n * fe.WIN].reshape(19, n, fe.WIN).transpose(1, 0, 2)
    keep = np.where((ww.max(2) - ww.min(2)).max(1) <= fe.PTP_MAX)[0]
    blocks = []; 
    for t in on:
        if blocks and t - blocks[-1][1] - 1 <= 30: blocks[-1][1] = t
        else: blocks.append([t, t])
    flag = np.array([any(lead + 10 + 10 * j < b + 1 + 10 and lead + 20 + 10 * j > a for a, b in blocks) for j in keep])
    chk(int(flag.sum()) == int(r.photic_windows) and len(keep) == int(r.windows), f"{r.file}: {flag.sum()} photic windows (saved {r.photic_windows}); {len(keep)} windows")
print("== 5. determinism: one fold of WP and SM re-run")
r, k, te = folds[12]; tr = np.setdiff1d(np.arange(200), te)
chk(np.array_equal(R26.pred("WP", tr, te), P["WP"][r, te]) and np.array_equal(R26.pred("SM", tr, te), P["SM"][r, te]), f"fold r{r}k{k}: re-run predictions identical to the saved ones")
print("== 6. 169-patient tests recomputed")
J169 = R26.c17.J169; f169 = lambda M: np.array([roc_auc_score(y[te], M[r, te]) for r, k, tr, te in J169])
d = f169(P["WP_169"]) - f169(P["I2_169"]); t = d.mean() / np.sqrt(d.var(ddof=1) * (1 / len(d) + 34 / 135)); a169 = pd.read_csv("/home/miftah_f/RTX-5090/tuep_fresh/reports/round26_169.csv")
chk(abs(d.mean() - a169.dAUC.iloc[0]) < 1e-12 and abs(2 * stats.t.sf(abs(t), len(d) - 1) - a169.p.iloc[0]) < 1e-9, f"I2-WP - I2 (169): dAUC {d.mean():+.4f}")
print(f"\n{'ALL CHECKS PASSED' if not FAIL else str(len(FAIL)) + ' CHECKS FAILED: ' + str(FAIL)}")
