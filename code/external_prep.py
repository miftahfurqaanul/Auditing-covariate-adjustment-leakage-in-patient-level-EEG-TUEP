"""Preprocess the Zenodo Emotiv dataset and the TUEP first sessions onto a shared 10-channel montage,
then extract band-power features per 10 s window. Design: reports/external_prediction.md."""
import numpy as np, pandas as pd, h5py, warnings
from multiprocessing import Pool
from fractions import Fraction
from scipy.signal import resample_poly, welch
import mne
warnings.filterwarnings("ignore"); mne.set_log_level("ERROR")
Z = "/home/miftah_f/RTX-5090/tuep_fresh/external/zenodo_1252141"
D = "/home/miftah_f/RTX-5090/tuep_fresh/data"
COMMON = ["F7", "F3", "F4", "F8", "T7", "T8", "P7", "P8", "O1", "O2"]
TUEP_NAME = {"T7": "T3", "T8": "T4", "P7": "T5", "P8": "T6"}
TUEP_CH = ["FP1", "FP2", "F7", "F3", "FZ", "F4", "F8", "T3", "C3", "CZ", "C4", "T4", "T5", "P3", "PZ", "P4", "T6", "O1", "O2"]
FS_IN, FS, WIN, PTP_MAX = 128, 200, 2000, 800.0
BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 45)}

def bandpower(w):
    f, P = welch(w, fs=FS, nperseg=400, axis=2)
    tot = P[:, :, (f >= 1) & (f < 45)].sum(2) + 1e-12
    rel = [np.log(P[:, :, (f >= lo) & (f < hi)].sum(2) / tot + 1e-12) for lo, hi in BANDS.values()]
    return np.concatenate(rel + [np.log(tot)], axis=1).astype(np.float32)

EXCLUDE_ALL = {"NG-508", "NG-515", "NG-565", "NG-548"}
EXCLUDE_CLOSED = {"NG-526", "NG-84", "NG-104", "NG-622"}

def closed_segment(row, n_sec):
    """Eyes-closed window, conservative against uncertain switch times (see external_prediction.md addendum)."""
    if row.sid in EXCLUDE_CLOSED:
        return 0, 0
    if row.sid == "NG-124":
        return 10, n_sec - 10
    if row.sid == "NG-600":
        return 60, 110
    if row.sid == "GB-58":
        return 220, n_sec - 10
    if str(row.eyes).lower().startswith("closed"):
        return 10, 110
    return 190, n_sec - 10

def prep_zenodo(row):
    d = pd.read_csv(f"{Z}/{row.file}", compression="gzip")
    x = d[COMMON].values.T.astype(np.float64)
    x = x - np.median(x, axis=1, keepdims=True)
    x = x - x.mean(0, keepdims=True)                                    # CAR over the 10 channels
    x = mne.filter.filter_data(x, FS_IN, 0.5, 45.0, method="fir", phase="zero", verbose="ERROR")
    gy = d[["GYROX", "GYROY"]].values.T.astype(np.float64)
    r = Fraction(FS, FS_IN); x = resample_poly(x, r.numerator, r.denominator, axis=1)
    gy = resample_poly(gy, r.numerator, r.denominator, axis=1)
    n_sec = x.shape[1] / FS
    row = row.copy()
    out = {}
    for seg, (a, b) in {"closed": closed_segment(row, n_sec), "whole": (10, n_sec - 10)}.items():
        a, b = int(a * FS), int(b * FS); xs, gs = x[:, a:b], gy[:, a:b]
        n = xs.shape[1] // WIN
        n = max(n, 0)
        w = xs[:, :n * WIN].reshape(10, n, WIN).transpose(1, 0, 2) if n else np.zeros((0, 10, WIN))
        g = gs[:, :n * WIN].reshape(2, n, WIN).transpose(1, 0, 2) if n else np.zeros((0, 2, WIN))
        out[seg] = (w.astype(np.float32), (g.max(2) - g.min(2)).max(1), (w.max(2) - w.min(2)).max(1))
    return out

def prep_tuep(h5):
    with h5py.File(f"{D}/{h5}") as f:
        x = f["eeg"][()].astype(np.float32)
    x = x[[TUEP_CH.index(TUEP_NAME.get(c, c)) for c in COMMON]]
    x = x - x.mean(0, keepdims=True)
    x = x[:, 2000:x.shape[1] - 2000]; n = x.shape[1] // WIN
    w = x[:, :n * WIN].reshape(10, n, WIN).transpose(1, 0, 2)
    return w[(w.max(2) - w.min(2)).max(1) <= PTP_MAX]

if __name__ == "__main__":
    idx = pd.read_csv(f"{Z}/zenodo_index.csv")
    idx = idx[idx.path.notna()].reset_index(drop=True)
    idx["sid"] = idx.country + "-" + idx["subject.id"].astype(str)
    idx = idx[~idx.file.str.endswith("signal-124-2.csv.gz") & ~idx.sid.isin(EXCLUDE_ALL)].reset_index(drop=True)
    assert idx.sid.is_unique, "one file per subject expected"
    print("subjects after exclusions:", len(idx), pd.crosstab(idx.country, idx.group).to_dict())
    with Pool(12) as pool:
        res = pool.map(prep_zenodo, [r for _, r in idx.iterrows()])
    def robust_thr(v):
        med = np.median(v); return med + 3 * 1.4826 * np.median(np.abs(v - med))
    thr = {}
    for c in ["GB", "NG"]:
        oc = [o for (_, r), o in zip(idx.iterrows(), res) if r.country == c]
        thr[c] = (robust_thr(np.concatenate([o["whole"][2] for o in oc])), robust_thr(np.concatenate([o["whole"][1] for o in oc])))
        print(f"{c}: peak-to-peak threshold {thr[c][0]:.0f} uV, gyro threshold {thr[c][1]:.1f} (median + 3 MAD, label-free)")
    for seg in ["closed", "whole"]:
        W, S, keep_stats = [], [], []
        for (_, r), o in zip(idx.iterrows(), res):
            w, g, p = o[seg]; k = (p <= thr[r.country][0]) & (g <= thr[r.country][1])
            keep_stats.append((len(w), int(k.sum()))); W.append(w[k]); S += [r.sid] * int(k.sum())
        W = np.concatenate(W); S = np.array(S)
        np.savez(f"{D}/ext_zenodo_{seg}_windows.npz", W=W.astype(np.float16), sid=S)
        np.savez(f"{D}/ext_zenodo_{seg}_bp.npz", X=bandpower(W), sid=S)
        ks = pd.DataFrame(keep_stats, columns=["n", "kept"]).assign(group=idx.group.values, country=idx.country.values)
        ks["frac"] = ks.kept / ks.n.clip(lower=1)
        print(f"\n[{seg}] windows kept {ks.kept.sum()} of {ks.n.sum()}; subjects with 0 kept: {(ks.kept == 0).sum()}")
        print(ks.groupby(["country", "group"])[["n", "kept", "frac"]].median().round(2))
    idx.to_csv(f"{D}/ext_zenodo_subjects.csv", index=False)
    meta = pd.read_csv(f"{D}/features_bandpower_recordings.csv"); meta = meta[meta.n_clean > 0].reset_index(drop=True)
    first = (meta.groupby("subject").session_no.transform("min") == meta.session_no).values
    fm = meta[first].reset_index(drop=True)
    with Pool(12) as pool:
        tw = pool.map(prep_tuep, fm.h5_path)
    S = np.concatenate([[s] * len(w) for s, w in zip(fm.subject, tw)])
    np.savez(f"{D}/ext_tuep10_first_bp.npz", X=bandpower(np.concatenate(tw)), sid=S)
    print(f"\nTUEP first sessions, 10-channel montage: {len(S)} windows from {len(set(S))} patients")
