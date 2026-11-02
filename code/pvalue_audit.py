"""Audit of the reported p-values (requested 2026-10-10). Recomputes them from fold-level AUCs with an independently written
Nadeau-Bengio test, recomputes permutation p-values from the saved null distributions, and applies Holm across the four
pre-registered comparator tests (Tang-F, Rohira-F set A, EEG-Bench-F BENDR, EEG-Bench-F LaBraM vs C1). CPU only."""
import glob, json, numpy as np, pandas as pd, warnings
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import evalkit as ek, tang_fixed as tf, eegbench_fixed_eval as ebf, rohira_features as rf, rohira_eval as re_
warnings.filterwarnings("ignore"); R = "/home/miftah_f/RTX-5090/tuep_fresh/reports"


def nb_test(a, b, n_train, n_test):
    """Nadeau & Bengio (2003) corrected resampled t-test, written independently of evalkit."""
    d = np.asarray(a, float) - np.asarray(b, float); J = d.size; m = d.mean(); s2 = ((d - m) ** 2).sum() / (J - 1)
    t = m / np.sqrt((1.0 / J + n_test / n_train) * s2); return m, 2 * stats.t.sf(abs(t), J - 1)


T = np.load(f"{ek.D}/p3_patient_table.npz", allow_pickle=True); S, Y = T["subjects"], T["y"]; C1X = np.hstack([T["bandpower"], T["agesex"]])
rows = []
# ---- Tang-F vs C1 on the 199 patients
F = tf.folds_199(); keep = ~np.isin(S, list(tf.EXCLUDE)); s199, y199, X199 = S[keep], Y[keep], C1X[keep]; pos = {s: i for i, s in enumerate(s199)}
fa_t, fa_c = [], []
for r in range(10):
    for k in range(5):
        te = np.array([pos[s] for s in F[(F.repeat == r) & (F.fold == k)].subject]); tr = np.setdiff1d(np.arange(len(y199)), te)
        d = json.load(open(f"{tf.OUT}/fixed/r{r}_f{k}.json")); fa_t.append(roc_auc_score(y199[te], [d["test"][s] for s in s199[te]]))
        fa_c.append(roc_auc_score(y199[te], ek.nested_lr().fit(X199[tr], y199[tr]).predict_proba(X199[te])[:, 1]))
rows.append(("Tang-F vs C1 (199 pts)", 0.014, *nb_test(fa_t, fa_c, 160, 40)))
# ---- EEG-Bench-F (variant D) vs C1 on 200 patients
_, fa_c1 = ebf.p3(C1X)
for enc, rep in [("bendr", 0.028), ("labram", 0.553)]:
    z = np.load(f"{ek.D}/eegbench_rep_D.npz"); rec = z["rec"]; X = z[enc].astype(np.float32); s = ebf.subj_r[rec]
    _, fa = ebf.p3(np.vstack([X[ebf.first[rec] & (s == p)].mean(0) for p in S])); rows.append((f"EEG-Bench-F {enc} vs C1", rep, *nb_test(fa, fa_c1, 160, 40)))
# ---- Rohira-F set A vs C1 (deterministic re-run of rohira_fixed.py set A, fold AUCs were not saved)
import threshold_select as ts
z = np.load(f"{rf.OUT}/features.npz"); idx = pd.read_csv(f"{rf.OUT}/index.csv"); idx["sno"] = idx.sess.str.extract(r"s(\d+)_")[0].astype(int)
first = (idx.sno == idx.groupby("subj").sno.transform("min")).values; w = first[z["file_idx"]]
ws, wy = idx.subj.values[z["file_idx"]][w], idx.label.values[z["file_idx"]][w]
Xr = z["X"][w][:, [rf.CH.index(c) for c in re_.SETS["A"]], :].reshape(w.sum(), -1); ylab = pd.Series(Y, index=S); posS = {s: i for i, s in enumerate(S)}
Fc = ek.folds(); fa_r, rep_auc = [], []
for r in range(10):
    P = np.zeros(len(Y))
    for k in range(5):
        te = np.array([posS[s] for s in Fc[(Fc.repeat == r) & (Fc.fold == k)].subject]); tr = np.setdiff1d(np.arange(len(Y)), te)
        tr_p, va_p = tf.split_val(list(S[tr]), ylab, 100 * r + k)
        m = RandomForestClassifier(n_estimators=100, max_depth=1000, class_weight="balanced", n_jobs=12, random_state=1000 * r + k).fit(Xr[np.isin(ws, tr_p)], wy[np.isin(ws, tr_p)])
        sel = np.isin(ws, S[te]); pt = pd.Series(m.predict_proba(Xr[sel])[:, 1]).groupby(ws[sel]).mean()
        P[te] = pt.loc[S[te]].values; fa_r.append(roc_auc_score(Y[te], P[te]))
    rep_auc.append(roc_auc_score(Y, P))
saved = pd.read_csv(f"{R}/rohira_fixed_setA.csv")["Rohira-F|AUC"].values
print(("PASS" if np.allclose(rep_auc, saved, atol=1e-12) else "FAIL") + " Rohira-F set A re-run reproduces the saved per-repeat AUCs exactly")
rows.append(("Rohira-F set A vs C1", 0.010, *nb_test(fa_r, fa_c1, 160, 40)))
# ---- Holm across the four pre-registered comparator tests
D = pd.DataFrame(rows, columns=["test", "reported_p", "dAUC", "recomputed_p"]); o = np.argsort(D.recomputed_p.values); h = np.empty(len(D)); run = 0
for rank, i in enumerate(o):
    run = max(run, min(1.0, D.recomputed_p.values[i] * (len(D) - rank))); h[i] = run
D["holm_p"] = h
print("\n== comparator tests vs C1 (corrected resampled t-test, 50 folds; independent implementation) ==")
print(D.round(4).to_string(index=False))
# ---- permutation p-values recomputed from the saved null distributions
print("\n== permutation p-values from saved nulls ==")
nul = np.load(f"{R}/matched_null.npy"); obs = pd.read_csv(f"{R}/matched_results.csv", index_col=0).loc["M-EEG", "AUC"]
print(f"matched M-EEG: observed {obs:.4f}, null max {nul.max():.4f}, p = {(1 + (nul >= obs).sum()) / (1 + len(nul)):.4f} (smallest possible with {len(nul)} permutations: {1 / (1 + len(nul)):.4f})")
cn = pd.read_csv(f"{R}/cbramod_permutation_null.csv"); cp = pd.read_csv(f"{R}/cbramod_p3.csv").set_index("config")
for c in ["C1", "ENS", "CBraMod"]:
    print(f"{c:8s}: observed {cp.AUC[c]:.4f}, null max {cn[c].max():.4f}, p = {(1 + (cn[c] >= cp.AUC[c]).sum()) / (1 + len(cn)):.4f} (smallest possible {1 / (1 + len(cn)):.4f})")
D.to_csv(f"{R}/pvalue_audit.csv", index=False)
