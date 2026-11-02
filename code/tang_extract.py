"""Extract order-2 path-signature features (1-s segments, 22-ch TCP, 0.5-50 Hz) for all 250 Hz and 256 Hz TUEP files.
Output: data/tang/feat_<fs>.h5 (one float32 dataset per file, shape (n_seconds, 506)) + data/tang/index_<fs>.csv. CPU only."""
import os, sys, numpy as np, pandas as pd, h5py, mne
from multiprocessing import Pool
import tang_ps as tp
ROOT = "/home/miftah_f/RTX-5090/EEG-TUEP-v2.0.1"; D = "/home/miftah_f/RTX-5090/tuep_fresh/data"; OUT = f"{D}/tang"
CHUNK, PAD = 3600, 15                                   # seconds; PAD > half FIR length (<= 3.3 s), so chunking is exact


def features(path, montage, fs_expected, chunk=CHUNK):
    suf = tp.SUFFIX[montage]
    raw = mne.io.read_raw_edf(path, preload=False, verbose="ERROR")
    fs = raw.info["sfreq"]; assert fs == fs_expected, (path, fs)
    w = int(fs); n_sec = raw.n_times // w
    names = sorted({c for p in tp.TCP22 for c in p}); ix = {c: i for i, c in enumerate(names)}
    picks = [f"EEG {c}-{suf}" for c in names]; out = []
    for s0 in range(0, n_sec, chunk):
        s1 = min(s0 + chunk, n_sec)
        a, b = max(0, (s0 - PAD) * w), min(raw.n_times, (s1 + PAD) * w)
        x = raw.get_data(picks=picks, start=a, stop=b) * 1e6
        x = np.stack([x[ix[p]] - x[ix[q]] for p, q in tp.TCP22])
        x = tp.bandpass(x, fs)[:, s0 * w - a: s1 * w - a]
        out.append(tp.signature2(tp.segments(x, fs)).astype(np.float32))
    return np.concatenate(out) if out else np.zeros((0, 506), np.float32)


def job(r):
    path = f"{ROOT}/{r['cls']}/{r['subj']}/{r['sess']}/{r['montage']}/{r['file']}"
    return r["file"], features(path, r["montage"], r["fs_eeg"])


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    h = pd.read_csv(f"{D}/edf_headers.csv")
    for fs in [250.0, 256.0]:
        s = h[(h.fs_eeg == fs) & (h.montage.isin(tp.SUFFIX))].copy()
        s["label"] = (s.cls == "00_epilepsy").astype(int)
        print(f"fs {fs}: {len(s)} files ({(h.fs_eeg == fs).sum() - len(s)} excluded for montage)", flush=True)
        with h5py.File(f"{OUT}/feat_{int(fs)}.h5", "w") as f, Pool(8) as pool:
            n = {}
            for i, (name, F) in enumerate(pool.imap_unordered(job, s.to_dict("records"), chunksize=1)):
                f.create_dataset(name, data=F); n[name] = len(F)
                if i % 100 == 0:
                    print(f"  {i}/{len(s)}", flush=True)
        s["n_sec"] = s.file.map(n)
        s[["file", "subj", "sess", "montage", "label", "fs_eeg", "duration_s", "n_sec", "year"]].to_csv(f"{OUT}/index_{int(fs)}.csv", index=False)
        print(f"fs {fs}: done, {s.n_sec.sum()} one-second segments", flush=True)
