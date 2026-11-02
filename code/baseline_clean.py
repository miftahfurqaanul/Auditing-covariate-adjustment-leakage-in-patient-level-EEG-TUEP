"""Step 1: the clean classical baseline.

Only clinically legitimate covariates (age, sex) may enter the diagnostic model.
Acquisition metadata (device, year, sampling rate, montage, duration) is reported only as the
shortcut floor, never as a model input.
"""
import json, numpy as np, pandas as pd, warnings
from sklearn.model_selection import GroupKFold
import evalkit as ek
warnings.filterwarnings("ignore")
D, R = ek.D, "/home/miftah_f/RTX-5090/tuep_fresh/reports"

# ------------------------------------------------------------ recording / window features
z = np.load(f"{D}/features_bandpower_windows.npz", allow_pickle=True)
Xw, win_h5 = z["X"], z["h5_path"]
rec = pd.read_csv(f"{D}/features_bandpower_recordings.csv"); rec = rec[rec.n_clean > 0].reset_index(drop=True)
hdr = pd.read_csv(f"{D}/edf_headers.csv")
hdr["age"] = hdr.patient.str.extract(r"Age:(-?\d+)")[0].astype(float)
hdr["male"] = (hdr.patient.str.extract(r"^\w+\s+([MF])\s")[0] == "M").astype(float)
hdr["device"] = hdr.recording.str.extract(r"XXX\s+(.*)$")[0].str.strip().fillna("NA")
hdr.loc[hdr.device.str.contains("#"), "device"] = "masked"
hdr["log_dur"] = np.log1p(hdr.duration_s)
rec = rec.merge(hdr[["file", "age", "male", "fs_eeg", "n_sig", "log_dur", "year", "montage", "device", "duration_s"]], on="file")
pos = {h: i for i, h in enumerate(rec.h5_path)}
wr = np.array([pos.get(h, -1) for h in win_h5]); Xw, wr = Xw[wr >= 0], wr[wr >= 0]

# ------------------------------------------------------------ P3 patient table (first session)
first = (rec.groupby("subject").session_no.transform("min") == rec.session_no).values
F = ek.folds(); subjects = np.array(sorted(F.subject.unique()))
lab = rec.groupby("subject").label.first(); y = lab.loc[subjects].values
eeg = np.vstack([np.median(Xw[np.isin(wr, np.where(first & (rec.subject.values == s))[0])], axis=0) for s in subjects])
fr = rec[first].groupby("subject")
agesex = np.column_stack([fr.age.first().loc[subjects].values, fr.male.first().loc[subjects].values])
acq = fr.agg(fs=("fs_eeg", lambda x: x.mode()[0]), nsig=("n_sig", lambda x: x.mode()[0]), dur=("duration_s", lambda x: np.log1p(x.sum())),
             year=("year", "first"), montage=("montage", lambda x: x.mode()[0]), device=("device", lambda x: x.mode()[0])).loc[subjects]
acq_X = np.hstack([acq[["fs", "nsig", "dur", "year"]].values.astype(float), pd.get_dummies(acq[["montage", "device"]].astype(str), dtype=float).values])
np.savez(f"{D}/p3_patient_table.npz", subjects=subjects, y=y, bandpower=eeg, agesex=agesex, acq=acq_X)

def lr_fp(Xtr, ytr, Xte, groups=None):
    m = ek.nested_lr()
    if groups is not None:                     # inner CV grouped by patient when rows are recordings
        m.set_params(logisticregressioncv__cv=list(GroupKFold(5).split(Xtr, ytr, groups)))
    return m.fit(Xtr, ytr).predict_proba(Xte)[:, 1]

print("== P3: first session, per patient, canonical 10x5 folds, nested-CV logistic regression ==")
out, oof = [], {}
sets = {"acquisition metadata (shortcut floor)": acq_X, "age+sex": agesex, "bandpower EEG": eeg, "bandpower EEG + age+sex": np.hstack([eeg, agesex])}
per_rep = {}
for name, X in sets.items():
    m, P = ek.cv_patient(X, subjects, y, lr_fp); out.append(ek.summarize(name, m)); per_rep[name] = m; oof[name] = P

# paired: does EEG add to age+sex? (same folds, per-repeat difference) + permutation of EEG rows
d = per_rep["bandpower EEG + age+sex"].AUC.values - per_rep["age+sex"].AUC.values
rng = np.random.default_rng(0); null = []
for _ in range(300):
    Xp = np.hstack([eeg[rng.permutation(len(y))], agesex])
    mp, _ = ek.cv_patient(Xp, subjects, y, lr_fp, n_repeats=1); null.append(mp.AUC.iloc[0] - per_rep["age+sex"].AUC.iloc[0])
p_add = (1 + np.sum(np.array(null) >= d.mean())) / (1 + len(null))
print(f"EEG adds to age+sex: dAUC {d.mean():+.3f} (repeats > 0: {(d > 0).sum()}/10), permutation p = {p_add:.4f}")

# ------------------------------------------------------------ P4: EEG-Bench split, per file
test_subj = ek.eegbench_test_subjects()
keep = (rec.duration_s >= 60).values
rec_eeg = np.vstack([np.median(Xw[wr == i], axis=0) for i in range(len(rec))])
rec_as = rec[["age", "male"]].values.astype(float)
subj_r, y_r = rec.subject.values, rec.label.values
trm, tem = keep & ~np.isin(subj_r, list(test_subj)), keep & np.isin(subj_r, list(test_subj))
print(f"\n== P4: EEG-Bench split (train {trm.sum()} files / {len(set(subj_r[trm]))} pts, test {tem.sum()} files / {len(set(subj_r[tem]))} pts), per file ==")
p4 = []
for name, X in {"age+sex": rec_as, "bandpower EEG": rec_eeg, "bandpower EEG + age+sex": np.hstack([rec_eeg, rec_as])}.items():
    m, _ = ek.p4_eval(X, subj_r, y_r, trm, tem, lr_fp)
    p4.append(dict(name=name, **m))
    print(f"{name:34s} AUC {m['AUC']:.3f}  BAC {m['BAC']:.3f} (95% CI {m['BAC_lo']:.3f}-{m['BAC_hi']:.3f})  SENS {m['SENS']:.3f}  SPEC {m['SPEC']:.3f}")
print("reference: BENDR 0.740, NeuroGPT 0.734, LaBraM 0.565, SVM/LDA 0.531 (EEG-Bench README, balanced accuracy)")

pd.DataFrame(out).to_csv(f"{R}/step1_p3.csv", index=False); pd.DataFrame(p4).to_csv(f"{R}/step1_p4.csv", index=False)
np.savez(f"{D}/oof_step1.npz", subjects=subjects, y=y, **{k.replace(' ', '_'): v for k, v in oof.items()})
json.dump(dict(dAUC_eeg_over_agesex=float(d.mean()), p_perm=float(p_add)), open(f"{R}/step1_tests.json", "w"), indent=1)
