"""Rohira-F (reports/rohira_fixed_prediction.md, commit 213718d): Rohira et al. 2023 features and random forest with the
evaluation gaps closed. First session, one prediction per patient, canonical 10x5 folds, class-balanced RF, threshold from
validation patients, compared with C1 and the metadata floor on the same patients and folds. CPU only."""
import numpy as np, pandas as pd, warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import evalkit as ek, rohira_features as rf, rohira_eval as re_, threshold_select as ts, tang_fixed as tf
warnings.filterwarnings("ignore"); R = "/home/miftah_f/RTX-5090/tuep_fresh/reports"


def full(y, p, thr):
    q = p > thr; tp = (q & (y == 1)).sum(); fp = (q & (y == 0)).sum(); se = q[y == 1].mean(); sp = 1 - q[y == 0].mean(); pr = tp / max(tp + fp, 1)
    return dict(acc=(q == y).mean(), sens=se, spec=sp, prec=pr, f1=2 * pr * se / max(pr + se, 1e-9), bac=(se + sp) / 2)


z = np.load(f"{rf.OUT}/features.npz"); idx = pd.read_csv(f"{rf.OUT}/index.csv")
idx["sno"] = idx.sess.str.extract(r"s(\d+)_")[0].astype(int); first = (idx.sno == idx.groupby("subj").sno.transform("min")).values
w = first[z["file_idx"]]; wsubj = idx.subj.values[z["file_idx"]][w]; wy = idx.label.values[z["file_idx"]][w]
T = np.load(f"{ek.D}/p3_patient_table.npz", allow_pickle=True); subjects, y = T["subjects"], T["y"]; pos = {s: i for i, s in enumerate(subjects)}
ylab = pd.Series(y, index=subjects); assert set(np.unique(wsubj)) == set(subjects)
F = ek.folds(); BASE = {"C1 band power + age/sex": np.hstack([T["bandpower"], T["agesex"]]), "acquisition-metadata floor": T["acq"]}
print(f"first-session windows {w.sum()} ({wy.sum()} epilepsy / {(1 - wy).sum()} non), patients {len(subjects)}", flush=True)
out = []
for set_name, chans in re_.SETS.items():
    X = z["X"][w][:, [rf.CH.index(c) for c in chans], :].reshape(w.sum(), -1)
    rows, fa = [], {k: [] for k in ["Rohira-F"] + list(BASE)}
    for r in range(10):
        P = {k: np.zeros(len(y)) for k in fa}; Q = np.zeros(len(y), bool); thrs = []
        for k in range(5):
            te = np.array([pos[s] for s in F[(F.repeat == r) & (F.fold == k)].subject]); tr = np.setdiff1d(np.arange(len(y)), te)
            tr_p, va_p = tf.split_val(list(subjects[tr]), ylab, 100 * r + k)
            m = RandomForestClassifier(n_estimators=100, max_depth=1000, class_weight="balanced", n_jobs=12, random_state=1000 * r + k)
            m.fit(X[np.isin(wsubj, tr_p)], wy[np.isin(wsubj, tr_p)])
            def pat(ps):
                sel = np.isin(wsubj, ps); return pd.Series(m.predict_proba(X[sel])[:, 1]).groupby(wsubj[sel]).mean()
            pv = pat(va_p); thr, _ = ts.best_threshold(ylab.loc[pv.index].values, pv.values); thrs.append(thr)
            pt = pat(subjects[te]); P["Rohira-F"][te] = pt.loc[subjects[te]].values; Q[te] = P["Rohira-F"][te] > thr
            for name, Xb in BASE.items():
                P[name][te] = ek.nested_lr().fit(Xb[tr], y[tr]).predict_proba(Xb[te])[:, 1]
            for name in fa:
                fa[name].append(roc_auc_score(y[te], P[name][te]))
        q = Q; tp = (q & (y == 1)).sum(); fp = (q & (y == 0)).sum(); se = q[y == 1].mean(); sp = 1 - q[y == 0].mean(); pr = tp / max(tp + fp, 1)
        row = {"Rohira-F|AUC": roc_auc_score(y, P["Rohira-F"]), "Rohira-F|acc": (q == y).mean(), "Rohira-F|sens": se, "Rohira-F|spec": sp,
               "Rohira-F|prec": pr, "Rohira-F|f1": 2 * pr * se / max(pr + se, 1e-9), "Rohira-F|bac@val-thr": (se + sp) / 2,
               "Rohira-F|bac@0.5": full(y, P["Rohira-F"], 0.5)["bac"]}
        for name in BASE:
            b = full(y, P[name], 0.5); row.update({f"{name}|AUC": roc_auc_score(y, P[name]), **{f"{name}|{k}@0.5": v for k, v in b.items()}})
        rows.append(row)
    D = pd.DataFrame(rows); D.to_csv(f"{R}/rohira_fixed_set{set_name}.csv", index=False)
    print(f"\n== set {set_name} ({', '.join(chans)}), 200 patients, canonical folds, 10 repeats ==")
    for c in D.columns:
        print(f"{c:52s} {D[c].mean():.3f} ± {D[c].std():.3f}")
    for name in BASE:
        d, p = ek.corrected_ttest(fa["Rohira-F"], fa[name]); print(f"Rohira-F minus {name}: dAUC {d:+.3f}, corrected p = {p:.3f} (50 folds)")
