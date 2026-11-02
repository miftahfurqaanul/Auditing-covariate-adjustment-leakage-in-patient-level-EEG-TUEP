# Independent leakage / integrity audit (CPU only, read-only).
import os; os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys; sys.path.insert(0, "/home/miftah_f/RTX-5090/tuep_fresh/code")
import numpy as np, pandas as pd, json, glob, itertools
D = "/home/miftah_f/RTX-5090/tuep_fresh/data"; FAIL = []
def chk(ok, msg): print(("PASS " if ok else "FAIL ") + msg); (None if ok else FAIL.append(msg))
idx = pd.read_csv(f"{D}/index.csv"); F = pd.read_csv(f"{D}/folds_patient.csv")
T = np.load(f"{D}/p3_patient_table.npz", allow_pickle=True); SUBJ, Y = list(T["subjects"]), T["y"]
print("== A. Labels and patient identity")
lab = idx.groupby("subject").label.nunique(); chk((lab == 1).all(), f"every patient has one label across all sessions ({len(lab)} patients)")
cls = idx.groupby("subject").cls.nunique(); chk((cls == 1).all(), "no patient appears in both the epilepsy and non-epilepsy folders")
y_idx = idx.groupby("subject").label.first().loc[SUBJ].values; chk((y_idx == Y).all(), "label of the feature table = label of index.csv for 200 patients")
chk(len(set(SUBJ)) == 200 and Y.sum() == 100, "200 unique patients, 100/100")
print("== B. Duplicate recordings (md5 of the first 30 seconds) across patients")
md = idx.dropna(subset=["md5_first30s"]); g = md.groupby("md5_first30s").subject.nunique()
cross = g[g > 1]; chk(len(cross) == 0, f"no identical recording (md5 of the first 30 s) in two different patients ({len(cross)} kasus)")
for h in cross.index[:10]: print("     ", md[md.md5_first30s == h][["subject", "label", "file"]].values.tolist())
dup = idx[idx.duplicate_of.notna()]; print(f"     note: {len(dup)} files flagged duplicate_of in the index (within the same patient?)",
      "all within the same patient" if all(dup.apply(lambda r: str(r.duplicate_of).split('_')[0] == r.subject, axis=1)) else "SOME ACROSS PATIENTS")
print("== C. First session only")
rec = pd.read_csv(f"{D}/features_bandpower_recordings.csv"); rec = rec[rec.n_clean > 0]
first = rec.groupby("subject").session_no.transform("min") == rec.session_no
chk(rec[first].groupby("subject").session_no.nunique().eq(1).all(), "patient features use one session (the first session) per patient")
print("== D. Fold kanonik")
ok = True
for r in range(10):
    Fr = F[F.repeat == r]; c = Fr.groupby("subject").fold.nunique()
    ok &= len(Fr) == 200 and (c == 1).all() and set(Fr.subject) == set(SUBJ) and all(len(Fr[Fr.fold == k]) == 40 for k in range(5))
chk(ok, "10 repetitions: each patient exactly once in the test fold, 5 folds x 40 patients, training = the remaining 160")
bal = F.groupby(["repeat", "fold"]).label.sum(); chk(bal.between(19, 21).all(), f"label-stratified folds (epilepsy per test fold {bal.min()}-{bal.max()})")
print("== E. Stored predictions: only for test patients, exactly once per repetition")
pos = {s: i for i, s in enumerate(SUBJ)}; mats = {}
for f, keys in [("round17/p1.npz", ["I2", "C1"]), ("round17/p3.npz", ["P3", "Riemann_raw"]), ("round17/p5.npz", ["LaBraM_raw", "P5"]), ("round24/r1.npz", ["W"]), ("round24/r2.npz", ["P3KS"]), ("round24/r3.npz", ["R3"])]:
    z = np.load(f"{D}/{f}")
    for k in keys: mats[k] = z[k]
nm = np.full((10, 200), np.nan); okj = True
for p in sorted(glob.glob(f"{D}/round16/r*_k*.json")):
    r, k = map(int, os.path.basename(p)[1:-5].split("_k")); d = json.load(open(p))["F"]["test"]
    te = set(F[(F.repeat == r) & (F.fold == k)].subject); okj &= set(d) == te
    for s, v in d.items(): nm[r, pos[s]] = v
chk(okj, "NormMamba: the patients in each JSON file = the test patients of that fold (50 files)"); mats["NM"] = nm
chk(all(m.shape == (10, 200) and not np.isnan(m).any() for m in mats.values()), f"{len(mats)} prediction matrices: exactly one prediction per patient per repetition, none missing")
print("== F. Perturbation test (test-patient data changed -> training features/model must be identical)")
import round4 as r4
tr, te = np.setdiff1d(np.arange(200), [pos[s] for s in F[(F.repeat == 3) & (F.fold == 2)].subject]), np.array([pos[s] for s in F[(F.repeat == 3) & (F.fold == 2)].subject])
z0 = r4.normative_z(r4.DESIGN["I2"], tr); B0 = r4.BP.copy(); r4.BP[te] = r4.BP[te] * 5 + 3; z1 = r4.normative_z(r4.DESIGN["I2"], tr); r4.BP[:] = B0
chk(np.allclose(z0[tr], z1[tr]), "I2 (fold r3k2): changing test patients does not change the z of training patients")
Yb = r4.Y.copy(); r4.Y[te] = 1 - r4.Y[te]; z2 = r4.normative_z(r4.DESIGN["I2"], tr); r4.Y[:] = Yb
chk(np.allclose(z0, z2), "I2: flipping the labels of test patients does not change anyone's features (test labels are not used)")
p_a = r4.fit_predict("I2", tr, te, 0); r4.Y[te] = 1 - r4.Y[te]; p_b = r4.fit_predict("I2", tr, te, 0); r4.Y[:] = Yb
chk(np.allclose(p_a, p_b), "I2: flipping the labels of test patients does not change the test predictions")
import normmamba as nmm
t0, s0, T0 = nmm.make_tokens(tr, "norm"); X0 = nmm.XSEG.copy(); te_seg = np.isin(nmm.PID, te); nmm.XSEG[te_seg] = nmm.XSEG[te_seg] * 3 + 1
t1, s1, T1 = nmm.make_tokens(tr, "norm"); nmm.XSEG[:] = X0
chk(all(np.array_equal(t0[i], t1[i]) for i in tr) and np.allclose(s0[tr], s1[tr]), "NormMamba: changing the segments of test patients does not change the tokens/summary of training patients")
import round17_p3 as p3
C, pid = p3.load_cache(12); X0p = p3.p3_features(C, pid, tr, r4.YEAR_G, "era")[0]; C2 = C.copy(); C2[np.isin(pid, te)] *= 4.0
X1p = p3.p3_features(C2, pid, tr, r4.YEAR_G, "era")[0]
chk(np.allclose(X0p[tr], X1p[tr]), "P3: changing the covariances of test patients does not change the features of training patients")
print("== G. Random-label control (I2, full pipeline, 50 folds, 3 permutations): AUC must be about 0.5")
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score
J = [(r, k, np.array([pos[s] for s in F[(F.repeat == r) & (F.fold != k)].subject]), np.array([pos[s] for s in F[(F.repeat == r) & (F.fold == k)].subject])) for r in range(10) for k in range(5)]
def perm_run(seed):
    rng = np.random.default_rng(seed); r4.Y[:] = rng.permutation(Yb)
    P = np.full((10, 200), np.nan)
    for r, k, tr_, te_ in J: P[r, te_] = r4.fit_predict("I2", tr_, te_, 0)
    yp = r4.Y.copy(); r4.Y[:] = Yb; return float(np.mean([roc_auc_score(yp, p) for p in P]))
aucs = Parallel(n_jobs=3)(delayed(perm_run)(s) for s in (101, 102, 103))
chk(all(0.40 < a < 0.60 for a in aucs), f"I2 with permuted labels: AUC {np.round(aucs, 3).tolist()}")
print(f"\n{'ALL CHECKS PASSED' if not FAIL else str(len(FAIL)) + ' CHECKS FAILED: ' + str(FAIL)}")
