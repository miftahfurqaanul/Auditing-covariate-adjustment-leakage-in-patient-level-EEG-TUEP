"""Round 7 (reports/round7_prediction.md, commit 39843db): direct tests of I2 against every comparator on the same patients and folds
(Tang-F on 199 patients; Rohira-F set A, DNM-F, EEG-Bench-F BENDR/LaBraM, acquisition-metadata floor on 200). Holm over 6. CPU only."""
import os, json, numpy as np, pandas as pd, warnings
os.environ.setdefault("OMP_NUM_THREADS", "1")
from joblib import Parallel, delayed
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import evalkit as ek, round4 as r4, round6_eval as r6, tang_fixed as tf, eegbench_fixed_eval as ebf, rohira_features as rf, rohira_eval as re_
warnings.filterwarnings("ignore"); R = "/home/miftah_f/RTX-5090/tuep_fresh/reports"
SUBJ, Y = r4.SUBJ, r4.Y; POS = {s: i for i, s in enumerate(SUBJ)}; F = ek.folds()
with np.load(f"{ek.D}/p3_patient_table.npz", allow_pickle=True) as _T:
    ACQ = _T["acq"]


def nb_test(a, b, n_train=160, n_test=40):
    """Second, independent implementation of the Nadeau-Bengio corrected resampled t-test (as in pvalue_audit.py)."""
    d = np.asarray(a, float) - np.asarray(b, float); J = d.size; m = d.mean(); s2 = ((d - m) ** 2).sum() / (J - 1)
    t = m / np.sqrt((1.0 / J + n_test / n_train) * s2); return m, 2 * stats.t.sf(abs(t), J - 1)


def jobs(exclude=()):
    keep = np.array([s not in exclude for s in SUBJ]); out = []
    for r in range(10):
        for k in range(5):
            te = np.array([POS[s] for s in F[(F.repeat == r) & (F.fold == k)].subject if s not in exclude])
            out.append((r, k, np.setdiff1d(np.where(keep)[0], te), te))
    return out


J200, J199 = jobs(), jobs(tf.EXCLUDE)


def lr_on(X, tr, te):
    return ek.nested_lr().fit(X[tr], Y[tr]).predict_proba(X[te])[:, 1]


def rohira_job(r, k, tr, te, Xr, ws, wy, ylab):
    tr_p, _ = tf.split_val(list(SUBJ[tr]), ylab, 100 * r + k)        # identical to rohira_fixed.py / pvalue_audit.py (model trained on 80%)
    m = RandomForestClassifier(n_estimators=100, max_depth=1000, class_weight="balanced", n_jobs=1, random_state=1000 * r + k)
    m.fit(Xr[np.isin(ws, tr_p)], wy[np.isin(ws, tr_p)]); sel = np.isin(ws, SUBJ[te])
    return pd.Series(m.predict_proba(Xr[sel])[:, 1]).groupby(ws[sel]).mean().loc[SUBJ[te]].values


def rohira_inputs():
    z = np.load(f"{rf.OUT}/features.npz"); idx = pd.read_csv(f"{rf.OUT}/index.csv"); idx["sno"] = idx.sess.str.extract(r"s(\d+)_")[0].astype(int)
    first = (idx.sno == idx.groupby("subj").sno.transform("min")).values; w = first[z["file_idx"]]
    ws, wy = idx.subj.values[z["file_idx"]][w], idx.label.values[z["file_idx"]][w]
    Xr = z["X"][w][:, [rf.CH.index(c) for c in re_.SETS["A"]], :].reshape(w.sum(), -1)
    return Xr, ws, wy, pd.Series(Y, index=SUBJ)


def predictions():
    P = {}
    P["I2"] = Parallel(n_jobs=12)(delayed(r4.fit_predict)("I2", tr, te, 0) for _, _, tr, te in J200)
    P["C1"] = Parallel(n_jobs=12)(delayed(r4.fit_predict)("C1", tr, te, 0) for _, _, tr, te in J200)
    P["I2@199"] = Parallel(n_jobs=12)(delayed(r4.fit_predict)("I2", tr, te, 0) for _, _, tr, te in J199)
    P["C1@199"] = Parallel(n_jobs=12)(delayed(r4.fit_predict)("C1", tr, te, 0) for _, _, tr, te in J199)
    P["Tang-F@199"] = [np.array([json.load(open(f"{tf.OUT}/fixed/r{r}_f{k}.json"))["test"][s] for s in SUBJ[te]]) for r, k, _, te in J199]
    P["DNM-F"] = Parallel(n_jobs=12)(delayed(r6.predict)("DNM-F", tr, te) for _, _, tr, te in J200)
    P["Floor"] = Parallel(n_jobs=12)(delayed(lr_on)(ACQ, tr, te) for _, _, tr, te in J200)
    z = np.load(f"{ek.D}/eegbench_rep_D.npz"); rec = z["rec"]; s = ebf.subj_r[rec]
    for enc, name in [("bendr", "BENDR"), ("labram", "LaBraM")]:
        X = z[enc].astype(np.float32); Xp = np.vstack([X[ebf.first[rec] & (s == p)].mean(0) for p in SUBJ])
        P[name] = Parallel(n_jobs=12)(delayed(lambda tr, te: ebf.lr_fp(Xp[tr], Y[tr], Xp[te]))(tr, te) for _, _, tr, te in J200)
    args = rohira_inputs()
    P["Rohira-F"] = Parallel(n_jobs=12)(delayed(rohira_job)(r, k, tr, te, *args) for r, k, tr, te in J200)
    return P


def summarise(out, J):
    per = []
    for r in range(10):
        idx = np.concatenate([te for (rr, _, _, te) in J if rr == r]); p = np.concatenate([o for (rr, _, _, _), o in zip(J, out) if rr == r])
        yy = Y[idx]; per.append(dict(auc=roc_auc_score(yy, p), **r4.binm(yy, p > 0.5), brier=np.mean((p - yy) ** 2)))
    D = pd.DataFrame(per); return D.mean(), D.auc.std(ddof=1), [roc_auc_score(Y[te], o) for (_, _, _, te), o in zip(J, out)]


if __name__ == "__main__":
    P = predictions(); S = {m: summarise(o, J199 if "@199" in m else J200) for m, o in P.items()}
    chk = [("C1 = 0.66627", abs(S["C1"][0].auc - 0.66627) < 1e-5), ("I2 = 0.71210", abs(S["I2"][0].auc - 0.71210) < 5e-5),
           ("Tang-F@199 = 0.549", abs(S["Tang-F@199"][0].auc - 0.549) < 5e-4), ("C1@199 = 0.672", abs(S["C1@199"][0].auc - 0.672) < 5e-4),
           ("DNM-F = 0.6109", abs(S["DNM-F"][0].auc - 0.6109) < 5e-5), ("Floor = 0.648", abs(S["Floor"][0].auc - 0.648) < 5e-4),
           ("BENDR = 0.553", abs(S["BENDR"][0].auc - 0.553) < 5e-4), ("LaBraM = 0.640", abs(S["LaBraM"][0].auc - 0.640) < 5e-4)]
    saved = pd.read_csv(f"{R}/rohira_fixed_setA.csv")["Rohira-F|AUC"].values
    rep = [roc_auc_score(Y[np.concatenate([te for (rr, _, _, te) in J200 if rr == r])],
                         np.concatenate([o for (rr, _, _, _), o in zip(J200, P["Rohira-F"]) if rr == r])) for r in range(10)]
    chk.append(("Rohira-F per-repeat AUC == rohira_fixed_setA.csv", np.allclose(rep, saved, atol=1e-12)))
    for n, ok in chk:
        print(("PASS " if ok else "FAIL ") + n)
    order = ["I2", "C1", "Tang-F@199", "I2@199", "C1@199", "Rohira-F", "DNM-F", "BENDR", "LaBraM", "Floor"]
    D = pd.DataFrame({m: S[m][0] for m in order}).T; D["auc_sd"] = [S[m][1] for m in order]
    print("\n== strict P3, threshold 0.5, mean over 10 repeats ==")
    print((100 * D[["auc", "auc_sd", "acc", "sens", "spec", "prec", "f1", "bac"]]).round(2).join(D[["brier"]].round(3)).to_string())
    tests = [("I2 vs Tang-F (199)", "I2@199", "Tang-F@199", "from scratch"), ("I2 vs Rohira-F", "I2", "Rohira-F", "from scratch"),
             ("I2 vs DNM-F", "I2", "DNM-F", "from scratch"), ("I2 vs BENDR", "I2", "BENDR", "pretrained"),
             ("I2 vs LaBraM", "I2", "LaBraM", "pretrained"), ("I2 vs metadata floor", "I2", "Floor", "non-EEG")]
    rows = []
    for n, a, b, g in tests:
        d1, p1 = ek.corrected_ttest(S[a][2], S[b][2]); d2, p2 = nb_test(S[a][2], S[b][2])
        rows.append(dict(test=n, group=g, dAUC_folds=d1, p=p1, p_check=p2, dAUC_pooled=S[a][0].auc - S[b][0].auc, wins=int(np.sum(np.array(S[a][2]) > np.array(S[b][2])))))
    T = pd.DataFrame(rows); o = np.argsort(T.p.values); h = np.empty(len(T)); run = 0
    for rank, i in enumerate(o):
        run = max(run, min(1, T.p.values[i] * (len(T) - rank))); h[i] = run
    T["p_holm"] = h
    print(("PASS" if np.allclose(T.p, T.p_check, rtol=1e-10) else "FAIL") + " two implementations of the corrected t-test agree")
    print("\n== I2 minus comparator: corrected t-test 160/40 on 50 fold AUCs, Holm over 6 ==")
    print(T.drop(columns="p_check").round(4).to_string(index=False))
    D.to_csv(f"{R}/round7_results.csv"); T.to_csv(f"{R}/round7_tests.csv", index=False)
    np.savez(f"{R}/round7_fold_aucs.npz", **{m: np.array(S[m][2]) for m in order})
