"""Window cache for the end-to-end CNNs: same clean 10 s windows as every other experiment, resampled to
100 Hz, clipped to +-800 uV, stored as float16 uV/100. Rows are in the order of features_bandpower_recordings."""
import numpy as np, pandas as pd, h5py
from multiprocessing import Pool
from scipy.signal import resample_poly
D = "/home/miftah_f/RTX-5090/tuep_fresh/data"
WIN, EDGE, PTP = 2000, 2000, 800.0

def rec_windows(h5):
    with h5py.File(f"{D}/{h5}") as f:
        x = f["eeg"][()].astype(np.float32)
    x = x[:, EDGE:x.shape[1] - EDGE]; n = x.shape[1] // WIN
    if n == 0:
        return np.zeros((0, 19, 1000), np.float16)
    w = x[:, :n * WIN].reshape(19, n, WIN).transpose(1, 0, 2)
    w = w[(w.max(2) - w.min(2)).max(1) <= PTP]
    w = resample_poly(w, 1, 2, axis=2)                       # 200 -> 100 Hz
    return (np.clip(w, -800, 800) / 100.0).astype(np.float16)

if __name__ == "__main__":
    meta = pd.read_csv(f"{D}/features_bandpower_recordings.csv"); meta = meta[meta.n_clean > 0].reset_index(drop=True)
    with Pool(12) as pool:
        ws = pool.map(rec_windows, meta.h5_path, chunksize=4)
    rec = np.concatenate([np.full(len(w), i, np.int32) for i, w in enumerate(ws)])
    W = np.concatenate(ws)
    np.save(f"{D}/cnn_windows100.npy", W); np.save(f"{D}/cnn_windows100_rec.npy", rec)
    print("windows", W.shape, W.dtype, f"{W.nbytes / 1e9:.1f} GB")
