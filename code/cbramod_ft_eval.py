"""Evaluate the CBraMod fine-tuning runs (reports/cbramod_ft_prediction.md). CPU only."""
import glob, json, os, numpy as np, pandas as pd, warnings
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, recall_score
import evalkit as ek, cbramod_eval as ce, threshold_select as ts
warnings.filterwarnings("ignore")
D = ek.D; OUT = f"{D}/cbramod_ft"

_, _, _, m4, _ = ts.tables()
test_subj = ek.eegbench_test_subjects(); keep = (m4.duration_s >= 60).values
subj_r, y_r = m4.subject.values, m4.label.values
te_rec = np.where(keep & np.isin(subj_r, list(test_subj)))[0]
yt, st = y_r[te_rec], subj_r[te_rec]; rng = np.random.default_rng(0); us = np.unique(st); rows_of = {s: np.where(st == s)[0] for s in us}
def boot(pred):
    b = []
    for _ in range(2000):
        i = np.concatenate([rows_of[s] for s in rng.choice(us, len(us))])
        if len(np.unique(yt[i])) == 2:
            b.append(balanced_accuracy_score(yt[i], pred[i]))
    return np.percentile(b, [2.5, 97.5])
runs = [json.load(open(p)) for p in sorted(glob.glob(f"{OUT}/p4_seed*.json"))]
print(f"== P4 EEG-Bench split: {len(runs)} seed(s) ==")
for i, r in enumerate(runs):
    p = np.array([r["test"][str(j)] for j in te_rec])
    print(f"seed {i}: best epoch {r['best_epoch']}, val AUROC {r['val_auc']:.3f} | test AUC {roc_auc_score(yt, p):.3f}, BAC@0.5 {balanced_accuracy_score(yt, p > 0.5):.3f}, BAC@val-thr({r['val_threshold']:.3f}) {balanced_accuracy_score(yt, p > r['val_threshold']):.3f}")
if runs:
    ens = np.mean([[r["test"][str(j)] for j in te_rec] for r in runs], 0); thr = float(np.mean([r["val_threshold"] for r in runs]))
    for tag, t in [("thr 0.5", 0.5), (f"thr val-mean {thr:.3f}", thr)]:
        pred = ens > t; lo, hi = boot(pred)
        print(f"ENSEMBLE {tag:18s}: AUC {roc_auc_score(yt, ens):.3f}  BAC {balanced_accuracy_score(yt, pred):.3f} (95% CI {lo:.3f}-{hi:.3f})  SENS {recall_score(yt, pred):.3f} SPEC {recall_score(1 - yt, 1 - pred):.3f}")
print("reference: BENDR 0.740 (EEG-Bench); our best P4 so far 0.704 (band power + age/sex, training threshold)")

T = np.load(f"{D}/p3_patient_table.npz", allow_pickle=True)
subjects, y = T["subjects"], T["y"]; pos = {s: i for i, s in enumerate(subjects)}; F = ek.folds()
X1 = np.hstack([T["bandpower"], T["agesex"]])
done = {(int(os.path.basename(p)[4]), int(os.path.basename(p)[7])): json.load(open(p)) for p in glob.glob(f"{OUT}/p3_r*_f*.json")}
reps = sorted({r for r, _ in done if all((r, k) in done for k in range(5))})
print(f"\n== P3 strict protocol: complete repeats {reps} ==")
fa_ft, fa_c1, out = [], [], []
for r in reps:
    Fr = F[F.repeat == r]; P_ft = np.zeros(len(y)); P_c1 = np.zeros(len(y)); Pt = np.zeros(len(y), bool)
    for k in range(5):
        te = np.array([pos[s] for s in Fr[Fr.fold == k].subject]); tr = np.setdiff1d(np.arange(len(y)), te)
        d = done[(r, k)]; P_ft[te] = [d["test"][s] for s in subjects[te]]; Pt[te] = P_ft[te] > d["val_threshold"]
        P_c1[te] = ce.fp(X1, tr, te, y)
        fa_ft.append(roc_auc_score(y[te], P_ft[te])); fa_c1.append(roc_auc_score(y[te], P_c1[te]))
    out.append((roc_auc_score(y, P_ft), balanced_accuracy_score(y, P_ft > 0.5), balanced_accuracy_score(y, Pt), roc_auc_score(y, P_c1), balanced_accuracy_score(y, P_c1 > 0.5)))
if reps:
    o = np.array(out); d, p = ek.corrected_ttest(fa_ft, fa_c1)
    print(f"CBraMod fine-tuned: AUC {o[:, 0].mean():.3f}  BAC@0.5 {o[:, 1].mean():.3f}  BAC@val-thr {o[:, 2].mean():.3f}")
    print(f"C1 same folds     : AUC {o[:, 3].mean():.3f}  BAC@0.5 {o[:, 4].mean():.3f}")
    print(f"paired: dAUC {d:+.3f}, corrected p = {p:.3f} ({len(fa_ft)} folds)")
    print("sequential rule: extend to 50 folds only if AUC >= 0.69 ->", "EXTEND" if o[:, 0].mean() >= 0.69 else "stop")
