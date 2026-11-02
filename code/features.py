"""Window-level spectral features from the rebuilt recordings (no labels used).

Per 10 s window (first/last 10 s of each recording skipped for filter transients):
  log relative power in delta 1-4, theta 4-8, alpha 8-13, beta 13-30, low gamma 30-45 Hz (19 x 5)
  log total 1-45 Hz power (19)
Windows with any channel peak-to-peak > 800 uV are rejected as artifact.
"""
import os, numpy as np, pandas as pd, h5py
from multiprocessing import Pool
from scipy.signal import welch
D = "/home/miftah_f/RTX-5090/tuep_fresh/data"
FS, WIN, EDGE, PTP_MAX = 200, 2000, 2000, 800.0
BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 45)}
CH = ["FP1", "FP2", "F7", "F3", "FZ", "F4", "F8", "T3", "C3", "CZ", "C4", "T4", "T5", "P3", "PZ", "P4", "T6", "O1", "O2"]
NAMES = [f"rel_{b}_{c}" for b in BANDS for c in CH] + [f"logtot_{c}" for c in CH]

def extract(h5_path):
    with h5py.File(os.path.join(D, h5_path)) as f:
        x = f["eeg"][()].astype(np.float32)
    x = x[:, EDGE:x.shape[1] - EDGE]
    n = x.shape[1] // WIN
    if n == 0:
        return h5_path, np.zeros((0, len(NAMES)), np.float32), 0
    w = x[:, :n * WIN].reshape(19, n, WIN).transpose(1, 0, 2)          # [n, 19, WIN]
    ptp = w.max(2) - w.min(2)
    keep = ptp.max(1) <= PTP_MAX
    w = w[keep]
    if len(w) == 0:
        return h5_path, np.zeros((0, len(NAMES)), np.float32), n
    f_, P = welch(w, fs=FS, nperseg=400, axis=2)                       # [k, 19, F]
    tot = P[:, :, (f_ >= 1) & (f_ < 45)].sum(2) + 1e-12
    rel = [np.log(P[:, :, (f_ >= lo) & (f_ < hi)].sum(2) / tot + 1e-12) for lo, hi in BANDS.values()]
    feats = np.concatenate(rel + [np.log(tot)], axis=1).astype(np.float32)
    return h5_path, feats, n

if __name__ == "__main__":
    idx = pd.read_csv(f"{D}/index.csv")
    use = idx[(idx.status == "ok") & idx.duplicate_of.isna()]
    out_rows, mats = [], []
    with Pool(8) as pool:
        for h5p, feats, n in pool.imap_unordered(extract, use.h5_path, chunksize=4):
            out_rows.append(dict(h5_path=h5p, n_windows=n, n_clean=len(feats))); mats.append((h5p, feats))
    meta = pd.DataFrame(out_rows).merge(use[["h5_path", "file", "cls", "label", "subject", "session", "session_no"]], on="h5_path")
    order = {h: i for i, h in enumerate(meta.h5_path)}
    X = np.concatenate([m for _, m in sorted(mats, key=lambda t: order[t[0]])])
    win_file = np.concatenate([[h] * len(m) for h, m in sorted(mats, key=lambda t: order[t[0]])])
    np.savez_compressed(f"{D}/features_bandpower_windows.npz", X=X, h5_path=win_file, names=np.array(NAMES))
    meta.to_csv(f"{D}/features_bandpower_recordings.csv", index=False)
    print("recordings", len(meta), "| windows total", int(meta.n_windows.sum()), "| clean", int(meta.n_clean.sum()),
          "| recordings with 0 clean windows", int((meta.n_clean == 0).sum()))
    print(meta.groupby("cls")[["n_windows", "n_clean"]].sum())
