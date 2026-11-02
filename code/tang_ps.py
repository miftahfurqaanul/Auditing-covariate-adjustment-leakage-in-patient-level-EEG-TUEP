"""Tang et al. 2024 reproduction: TCP bipolar loading, 0.5-50 Hz FIR, 1-s segments, order-2 path signature.
See reports/tang_repro_prediction.md for every choice and its source."""
import numpy as np, mne

# Official TUH montage definitions (DOCS/01_tcp_ar, 02_tcp_le, 03_tcp_ar_a); checked against the files in tang_check.py
TCP22 = [("FP1", "F7"), ("F7", "T3"), ("T3", "T5"), ("T5", "O1"), ("FP2", "F8"), ("F8", "T4"), ("T4", "T6"), ("T6", "O2"),
         ("A1", "T3"), ("T3", "C3"), ("C3", "CZ"), ("CZ", "C4"), ("C4", "T4"), ("T4", "A2"),
         ("FP1", "F3"), ("F3", "C3"), ("C3", "P3"), ("P3", "O1"), ("FP2", "F4"), ("F4", "C4"), ("C4", "P4"), ("P4", "O2")]
SUFFIX = {"01_tcp_ar": "REF", "02_tcp_le": "LE"}


def load_tcp(path, montage):
    """Return (22, T) bipolar TCP signal in microvolts and the sampling rate."""
    suf = SUFFIX[montage]
    raw = mne.io.read_raw_edf(path, preload=False, verbose="ERROR")
    names = sorted({c for pair in TCP22 for c in pair})
    picks = [f"EEG {c}-{suf}" for c in names]
    x = raw.get_data(picks=picks) * 1e6
    ix = {c: i for i, c in enumerate(names)}
    return np.stack([x[ix[a]] - x[ix[b]] for a, b in TCP22]), raw.info["sfreq"]


def bandpass(x, fs):
    return mne.filter.filter_data(x, fs, 0.5, 50.0, method="fir", fir_design="firwin", phase="zero", h_trans_bandwidth=5.0, verbose="ERROR")


def segments(x, fs):
    """Non-overlapping 1-s windows -> (n, 22, fs)."""
    w = int(round(fs)); n = x.shape[1] // w
    return x[:, :n * w].reshape(x.shape[0], n, w).transpose(1, 0, 2)


def signature2(seg):
    """Truncated (order 1 and 2) signature of the piecewise-linear path through the samples.
    seg: (n, d, T) -> (n, d + d*d); level-2 term (i, j) flattened as i*d + j (Signatory order)."""
    p = np.ascontiguousarray(seg.transpose(0, 2, 1), dtype=np.float64)   # (n, T, d)
    inc = np.diff(p, axis=1)                                              # (n, T-1, d)
    s1 = p[:, -1] - p[:, 0]
    prev = p[:, :-1] - p[:, :1]                                           # position before each increment, relative to start
    s2 = np.einsum("nti,ntj->nij", prev, inc) + 0.5 * np.einsum("nti,ntj->nij", inc, inc)
    return np.concatenate([s1, s2.reshape(len(p), -1)], axis=1)
