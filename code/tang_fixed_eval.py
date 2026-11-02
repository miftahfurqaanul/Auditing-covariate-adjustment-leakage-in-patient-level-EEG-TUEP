"""Evaluate Tang-F against C1 and the acquisition-metadata floor on the same 199 patients and folds
(reports/tang_fixed_prediction.md, commit cbd7995). CPU only."""
import glob, json, os, numpy as np, pandas as pd, warnings
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, recall_score
import evalkit as ek, tang_fixed as tf
warnings.filterwarnings("ignore")
R = "/home/miftah_f/RTX-5090/tuep_fresh/reports"
T = np.load(f"{ek.D}/p3_patient_table.npz", allow_pickle=True); keep = ~np.isin(T["subjects"], list(tf.EXCLUDE))
subjects, y = T["subjects"][keep], T["y"][keep]; pos = {s: i for i, s in enumerate(subjects)}
X = {"C1 band power + age/sex": np.hstack([T["bandpower"], T["agesex"]])[keep], "acquisition-metadata floor": T["acq"][keep]}
F = tf.folds_199(); done = {(int(os.path.basename(p)[1]), int(os.path.basename(p)[4])): json.load(open(p)) for p in glob.glob(f"{tf.OUT}/fixed/r*_f*.json")}
reps = sorted({r for r, _ in done if all((r, k) in done for k in range(5))})
rows, fold_auc = [], {k: [] for k in ["Tang-F"] + list(X)}
for r in reps:
    P = {k: np.zeros(len(y)) for k in fold_auc}; Q = np.zeros(len(y), bool)
    for k in range(5):
        te = np.array([pos[s] for s in F[(F.repeat == r) & (F.fold == k)].subject]); tr = np.setdiff1d(np.arange(len(y)), te); d = done[(r, k)]
        P["Tang-F"][te] = [d["test"][s] for s in subjects[te]]; Q[te] = P["Tang-F"][te] > d["val_threshold"]
        for name, Xm in X.items():
            P[name][te] = ek.nested_lr().fit(Xm[tr], y[tr]).predict_proba(Xm[te])[:, 1]
        for name in fold_auc:
            fold_auc[name].append(roc_auc_score(y[te], P[name][te]))
    rows.append(dict(repeat=r, **{f"{n}|AUC": roc_auc_score(y, P[n]) for n in P}, **{f"{n}|BAC@0.5": balanced_accuracy_score(y, P[n] > 0.5) for n in P},
                     **{"Tang-F|BAC@val-thr": balanced_accuracy_score(y, Q), "Tang-F|SENS@val-thr": recall_score(y, Q), "Tang-F|SPEC@val-thr": recall_score(1 - y, 1 - Q)}))
D = pd.DataFrame(rows); D.to_csv(f"{R}/tang_fixed_results.csv", index=False)
print(f"== Tang-F on 199 patients (first session), canonical folds: complete repeats {reps} ==")
for c in D.columns[1:]:
    print(f"{c:42s} {D[c].mean():.3f} ± {D[c].std():.3f}")
for name in X:
    d, p = ek.corrected_ttest(fold_auc["Tang-F"], fold_auc[name])
    print(f"Tang-F minus {name}: dAUC {d:+.3f}, corrected p = {p:.3f} ({len(fold_auc[name])} folds)")
ep = [v["best_epoch"] for v in done.values()]; print(f"best epoch chosen on validation: median {np.median(ep):.0f}, range {min(ep)}-{max(ep)}")
