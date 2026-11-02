"""Evaluate the saved CNN predictions (reports/cnn_prediction.md). CPU only."""
import json, os, numpy as np, pandas as pd, warnings
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, recall_score
import evalkit as ek, cbramod_eval as ce, threshold_select as ts
warnings.filterwarnings("ignore")
D, R = ek.D, "/home/miftah_f/RTX-5090/tuep_fresh/reports"

T = np.load(f"{D}/p3_patient_table.npz", allow_pickle=True)
subjects, y = T["subjects"], T["y"]; pos = {s: i for i, s in enumerate(subjects)}
F = ek.folds()
X1 = np.hstack([T["bandpower"], T["agesex"]])
c1_fa, c1_rep = [], []
for r in range(10):
    Fr = F[F.repeat == r]; P = np.zeros(len(y))
    for k in range(5):
        te = np.array([pos[s] for s in Fr[Fr.fold == k].subject]); tr = np.setdiff1d(np.arange(len(y)), te)
        P[te] = ce.fp(X1, tr, te, y); c1_fa.append(roc_auc_score(y[te], P[te]))
    c1_rep.append(ek.metrics(y, P))
print(f"C1 band power + age/sex: AUC {np.mean([m['AUC'] for m in c1_rep]):.3f}  BAC {np.mean([m['BAC'] for m in c1_rep]):.3f}")
for arch in ["shallow", "deep4"]:
    path = f"{D}/cnn_p3_{arch}.json"
    if not os.path.exists(path):
        continue
    done = json.load(open(path))
    if len(done) < 50:
        print(f"{arch}: P3 incomplete ({len(done)}/50 folds)"); continue
    fa, rep = [], []
    for r in range(10):
        P = np.zeros(len(y))
        for k in range(5):
            d = done[f"{r}-{k}"]; idx = np.array([pos[s] for s in d]); P[idx] = list(d.values())
            fa.append(roc_auc_score(y[idx], P[idx]))
        rep.append(ek.metrics(y, P))
    d_, p_ = ek.corrected_ttest(fa, c1_fa)
    a = np.array([m["AUC"] for m in rep]); b = np.array([m["BAC"] for m in rep])
    print(f"P3 {arch:8s}: AUC {a.mean():.3f} ± {a.std():.3f}  BAC {b.mean():.3f} ± {b.std():.3f}  vs C1 dAUC {d_:+.3f} p={p_:.3f}")

_, _, _, m4, _ = ts.tables()
test_subj = ek.eegbench_test_subjects(); keep = (m4.duration_s >= 60).values
subj_r, y_r = m4.subject.values, m4.label.values
tr_rec = np.where(keep & ~np.isin(subj_r, list(test_subj)))[0]; te_rec = np.where(keep & np.isin(subj_r, list(test_subj)))[0]
yt, st = y_r[te_rec], subj_r[te_rec]; rng = np.random.default_rng(0); us = np.unique(st); rows_of = {s: np.where(st == s)[0] for s in us}
def boot(pred):
    b = []
    for _ in range(2000):
        i = np.concatenate([rows_of[s] for s in rng.choice(us, len(us))])
        if len(np.unique(yt[i])) == 2:
            b.append(balanced_accuracy_score(yt[i], pred[i]))
    return np.percentile(b, [2.5, 97.5])
for arch in ["shallow", "deep4"]:
    path = f"{D}/cnn_p4_{arch}.json"
    if not os.path.exists(path):
        continue
    done = json.load(open(path)); seeds = [k for k in done if k.startswith("seed")]
    if len(seeds) < 5 or sum(k.startswith("inner") for k in done) < 5:
        print(f"{arch}: P4 incomplete ({list(done)})"); continue
    inner = {}
    for k in done:
        if k.startswith("inner"):
            inner.update({int(r): v for r, v in done[k].items()})
    ir = np.array(sorted(inner)); thr, inner_bac = ts.best_threshold(y_r[ir], np.array([inner[i] for i in ir]))
    S = np.array([[done[s][str(i)] for i in te_rec] for s in seeds])
    per_seed = [balanced_accuracy_score(yt, s > 0.5) for s in S]; ens = S.mean(0)
    for tag, t in [("thr 0.5", 0.5), (f"thr inner {thr:.3f}", thr)]:
        pred = ens > t; lo, hi = boot(pred)
        print(f"P4 {arch:8s} 5-seed ensemble {tag:16s}: AUC {roc_auc_score(yt, ens):.3f}  BAC {balanced_accuracy_score(yt, pred):.3f} (95% CI {lo:.3f}-{hi:.3f})  SENS {recall_score(yt, pred):.3f} SPEC {recall_score(1 - yt, 1 - pred):.3f}")
    print(f"   per-seed BAC at 0.5: {np.round(per_seed, 3)} | inner OOF BAC at chosen threshold {inner_bac:.3f}")
print("targets: P4 BENDR 0.740 (EEG-Bench); our best so far P4 0.704, P3 AUC 0.666")
