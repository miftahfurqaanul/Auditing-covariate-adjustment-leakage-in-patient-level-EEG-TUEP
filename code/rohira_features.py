"""Rohira et al. 2023 reproduction (reports/rohira_repro_prediction.md): resample 250 Hz, high-pass 1 Hz, notch 60 Hz,
contiguous 10-s windows, Welch PSD band powers for 12 candidate channels (sets A and B are subsets). CPU only.
Output: data/rohira/features.npz (X: (n_windows, 12, 6) float32, file_idx: window -> file row) + data/rohira/index.csv."""
import os, numpy as np, pandas as pd, mne
from fractions import Fraction
from multiprocessing import Pool
from scipy.signal import resample_poly, welch
ROOT = "/home/miftah_f/RTX-5090/EEG-TUEP-v2.0.1"; D = "/home/miftah_f/RTX-5090/tuep_fresh/data"; OUT = f"{D}/rohira"
CH = ["FP1", "FP2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2", "T3", "T4"]
BANDS = [(1, 4), (4, 7.5), (7.5, 13), (13, 16), (16, 30), (30, 40)]
FS, WIN, CHUNK, PAD = 250, 10, 3600, 20                     # Hz, s, s, s
SUFFIX = {"01_tcp_ar": "REF", "02_tcp_le": "LE", "03_tcp_ar_a": "REF"}


def preprocess(x, fs):
    """x: (12, n) microvolts at fs -> (12, m) at 250 Hz, high-pass 1 Hz, notch 60 Hz."""
    if fs != FS:
        r = Fraction(FS, int(fs)).limit_denominator(1000); x = resample_poly(x, r.numerator, r.denominator, axis=1)
    x = mne.filter.filter_data(x, FS, 1.0, None, method="fir", fir_design="firwin", phase="zero", verbose="ERROR")
    return mne.filter.notch_filter(x, FS, 60.0, method="fir", phase="zero", verbose="ERROR")


def band_powers(x):
    """x: (12, n) at 250 Hz -> (n_windows, 12, 6) summed Welch PSD per band."""
    n = x.shape[1] // (WIN * FS)
    if n == 0:
        return np.zeros((0, len(CH), len(BANDS)), np.float32)
    w = x[:, : n * WIN * FS].reshape(len(CH), n, WIN * FS).transpose(1, 0, 2)
    f, P = welch(w, fs=FS, nperseg=2 * FS, noverlap=FS, window="hann", axis=2)
    return np.stack([P[..., (f >= lo) & (f < hi)].sum(-1) for lo, hi in BANDS], axis=-1).astype(np.float32)


def file_features(path, montage, chunk=CHUNK):
    raw = mne.io.read_raw_edf(path, preload=False, verbose="ERROR"); fs = raw.info["sfreq"]
    picks = [f"EEG {c}-{SUFFIX[montage]}" for c in CH]; n_sec = int(raw.n_times // fs); out = []
    for s0 in range(0, n_sec - n_sec % WIN, chunk):
        s1 = min(s0 + chunk, n_sec - n_sec % WIN)
        a, b = int(max(0, s0 - PAD) * fs), int(min(raw.n_times, (s1 + PAD) * fs))
        x = preprocess(raw.get_data(picks=picks, start=a, stop=b) * 1e6, fs)
        off = int(round((s0 * fs - a) * FS / fs))
        out.append(band_powers(x[:, off: off + (s1 - s0) * FS]))
    return np.concatenate(out) if out else np.zeros((0, len(CH), len(BANDS)), np.float32)


def job(r):
    return file_features(f"{ROOT}/{r['cls']}/{r['subj']}/{r['sess']}/{r['montage']}/{r['file']}", r["montage"])


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    h = pd.read_csv(f"{D}/edf_headers.csv"); h = h[h.duration_s >= WIN].reset_index(drop=True)
    h["label"] = (h.cls == "00_epilepsy").astype(int)
    with Pool(8) as pool:
        feats = pool.map(job, h.to_dict("records"), chunksize=2)
    h["n_win"] = [len(f) for f in feats]
    np.savez(f"{OUT}/features.npz", X=np.concatenate(feats), file_idx=np.repeat(np.arange(len(h)), h.n_win.values))
    h[["file", "subj", "sess", "montage", "fs_eeg", "label", "duration_s", "n_win"]].to_csv(f"{OUT}/index.csv", index=False)
    print(f"{len(h)} files, {h.n_win.sum()} windows ({h[h.label == 1].n_win.sum()} epilepsy / {h[h.label == 0].n_win.sum()} non)")
