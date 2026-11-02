"""Step 2a: linear probes on frozen foundation-model embeddings (pre-specified design).

Representations: bandpower (step 1), LaBraM, BENDR, and their concatenation ("fusion").
P3: first session, mean embedding per patient, canonical 10x5 folds, nested-CV logistic regression,
    each variant with and without age+sex; paired comparison vs the step-1 baseline
    (bandpower + age+sex) by a corrected resampled t-test on the 50 fold-level AUCs.
P4: EEG-Bench split, mean embedding per recording (>= 60 s), scored per file.
"""
import json, numpy as np, pandas as pd, warnings
from sklearn.model_selection import GroupKFold
import evalkit as ek
warnings.filterwarnings("ignore")
D, R = ek.D, "/home/miftah_f/RTX-5090/tuep_fresh/reports"

meta = pd.read_csv(f"{D}/features_bandpower_recordings.csv")
meta = meta[meta.n_clean > 0].reset_index(drop=True)
hdr = pd.read_csv(f"{D}/edf_headers.csv")
hdr["age"] = hdr.patient.str.extract(r"Age:(-?\d+)")[0].astype(float)
hdr["male"] = (hdr.patient.str.extract(r"^\w+\s+([MF])\s")[0] == "M").astype(float)
meta = meta.merge(hdr[["file", "age", "male", "duration_s"]], on="file", how="left")
reps = {}
z = np.load(f"{D}/features_bandpower_windows.npz", allow_pickle=True)
pos = {h: i for i, h in enumerate(meta.h5_path)}
wr = np.array([pos.get(h, -1) for h in z["h5_path"]]); reps["bandpower"] = (z["X"][wr >= 0], wr[wr >= 0])
for k in ["labram", "bendr"]:
    e = np.load(f"{D}/emb_{k}_windows.npz", allow_pickle=True)
    assert (e["h5_path"] == meta.h5_path.values).all()
    reps[k] = (e["E"].astype(np.float32), e["rec"])
assert all(len(v[1]) == len(reps["bandpower"][1]) for v in reps.values())

first = (meta.groupby("subject").session_no.transform("min") == meta.session_no).values
T = np.load(f"{D}/p3_patient_table.npz", allow_pickle=True)
subjects, y, agesex = T["subjects"], T["y"], T["agesex"]

def pool(E, rec, rec_mask_per_unit):
    return np.vstack([E[np.isin(rec, ids)].mean(0) for ids in rec_mask_per_unit])
pt_recs = [np.where(first & (meta.subject.values == s))[0] for s in subjects]
P3 = {k: pool(E, rec, pt_recs) for k, (E, rec) in reps.items()}
P3["fusion"] = np.hstack([P3["bandpower"], P3["labram"], P3["bendr"]])

def lr_fp(Xtr, ytr, Xte, groups=None):
    m = ek.nested_lr()
    if groups is not None:
        m.set_params(logisticregressioncv__cv=list(GroupKFold(5).split(Xtr, ytr, groups)))
    return m.fit(Xtr, ytr).predict_proba(Xte)[:, 1]

print("== P3: first session, per patient (n=200), canonical 10x5 folds ==")
rows, fold_aucs, oof = [], {}, {}
for k in ["bandpower", "labram", "bendr", "fusion"]:
    for with_as in [False, True]:
        name = k + (" + age+sex" if with_as else "")
        X = np.hstack([P3[k], agesex]) if with_as else P3[k]
        m, P, fa = ek.cv_patient(X, subjects, y, lr_fp, return_folds=True)
        rows.append(ek.summarize(name, m)); fold_aucs[name] = fa; oof[name] = P
ref = "bandpower + age+sex"
print(f"\npaired vs {ref} (corrected resampled t-test, 50 folds):")
for name, fa in fold_aucs.items():
    if name != ref:
        d, p = ek.corrected_ttest(fa, fold_aucs[ref])
        print(f"  {name:28s} dAUC {d:+.3f}  p={p:.3f}")
        next(r for r in rows if r["name"] == name).update(dAUC_vs_ref=d, p_vs_ref=p)

# ------------------------------------------------------------------ P4
test_subj = ek.eegbench_test_subjects()
keep = (meta.duration_s >= 60).values
subj_r, y_r = meta.subject.values, meta.label.values
trm, tem = keep & ~np.isin(subj_r, list(test_subj)), keep & np.isin(subj_r, list(test_subj))
recs = [np.array([i]) for i in range(len(meta))]
R4 = {k: np.vstack([E[rec == i].mean(0) for i in range(len(meta))]) for k, (E, rec) in reps.items()}
R4["fusion"] = np.hstack([R4["bandpower"], R4["labram"], R4["bendr"]])
print(f"\n== P4: EEG-Bench split (train {trm.sum()} files / 150 pts, test {tem.sum()} files / 50 pts), per file ==")
p4 = []
for k in ["bandpower", "labram", "bendr", "fusion"]:
    m, _ = ek.p4_eval(R4[k], subj_r, y_r, trm, tem, lr_fp)
    p4.append(dict(name=k, **m))
    print(f"{k:12s} AUC {m['AUC']:.3f}  BAC {m['BAC']:.3f} (95% CI {m['BAC_lo']:.3f}-{m['BAC_hi']:.3f})  SENS {m['SENS']:.3f}  SPEC {m['SPEC']:.3f}")
print("reference (EEG-Bench README, BAC): BENDR 0.740, NeuroGPT 0.734, LaBraM 0.565, SVM/LDA 0.531")

pd.DataFrame(rows).to_csv(f"{R}/step2a_p3.csv", index=False); pd.DataFrame(p4).to_csv(f"{R}/step2a_p4.csv", index=False)
np.savez(f"{D}/oof_step2a.npz", subjects=subjects, y=y, **{k.replace(' ', '_').replace('+', 'p'): v for k, v in oof.items()})
