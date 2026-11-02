"""Round 16 (reports/round16_normmamba_prediction.md, commit e2b8b14): 4-s segment band-power tokens for the first session of every patient.
Same H5 signals and feature formula as features.py; segments in temporal order (recordings sorted by h5_path); max 600 clean segments per patient.
No labels are used. CPU only. Output: data/seg4s_features.npz"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np, pandas as pd, h5py
from multiprocessing import Pool
from scipy.signal import welch
import features as fe, round4 as r4
SEG, MAXSEG = 4 * fe.FS, 600


def seg_feats(w):
    """features.extract formula for windows w (n, 19, T)."""
    f_, P = welch(w, fs=fe.FS, nperseg=400, axis=2); tot = P[:, :, (f_ >= 1) & (f_ < 45)].sum(2) + 1e-12
    rel = [np.log(P[:, :, (f_ >= lo) & (f_ < hi)].sum(2) / tot + 1e-12) for lo, hi in fe.BANDS.values()]
    return np.concatenate(rel + [np.log(tot)], axis=1).astype(np.float32)


def patient(args):
    s, paths = args; out, nrej = [], 0
    for h in paths:
        with h5py.File(os.path.join(fe.D, h)) as f:
            x = f["eeg"][()].astype(np.float32)
        x = x[:, fe.EDGE:x.shape[1] - fe.EDGE]; k = x.shape[1] // SEG
        if k == 0:
            continue
        w = x[:, :k * SEG].reshape(19, k, SEG).transpose(1, 0, 2); keep = (w.max(2) - w.min(2)).max(1) <= fe.PTP_MAX
        nrej += int((~keep).sum()); w = w[keep]
        if len(w):
            out.append(seg_feats(w))
        if sum(len(o) for o in out) >= MAXSEG:
            break
    X = np.vstack(out)[:MAXSEG] if out else np.zeros((0, 114), np.float32)
    return s, X, nrej


if __name__ == "__main__":
    rec = pd.read_csv(f"{fe.D}/features_bandpower_recordings.csv"); rec = rec[rec.n_clean > 0]
    first = rec[rec.session_no == rec.groupby("subject").session_no.transform("min")]
    jobs = [(s, sorted(first[first.subject == s].h5_path)) for s in r4.SUBJ]
    with Pool(12) as pool:
        res = {s: (X, n) for s, X, n in pool.map(patient, jobs)}
    X = np.vstack([res[s][0] for s in r4.SUBJ]); pid = np.concatenate([np.full(len(res[s][0]), i) for i, s in enumerate(r4.SUBJ)])
    nseg = np.array([len(res[s][0]) for s in r4.SUBJ])
    # independent check on one recording: direct per-window loop must equal the vectorised features
    s0, p0 = jobs[0]
    with h5py.File(os.path.join(fe.D, p0[0])) as f:
        x = f["eeg"][()].astype(np.float32)[:, fe.EDGE:]
    w = x[:, :SEG]; fr, P = welch(w, fs=fe.FS, nperseg=400, axis=1); tot = P[:, (fr >= 1) & (fr < 45)].sum(1) + 1e-12
    ind = np.concatenate([np.log(P[:, (fr >= lo) & (fr < hi)].sum(1) / tot + 1e-12) for lo, hi in fe.BANDS.values()] + [np.log(tot)])
    ok_first = (w.max(1) - w.min(1)).max() <= fe.PTP_MAX
    print(("PASS" if (not ok_first) or np.allclose(ind, res[s0][0][0], atol=1e-5) else "FAIL") + " first segment of the first recording equals an independent per-channel computation")
    print(f"patients {len(nseg)}, segments total {len(X)}, per patient median {np.median(nseg):.0f} (min {nseg.min()}, max {nseg.max()}), "
          f"capped at {MAXSEG}: {(nseg == MAXSEG).sum()}; finite {np.isfinite(X).all()}; rejected segments total {sum(res[s][1] for s in r4.SUBJ)}")
    np.savez_compressed(f"{fe.D}/seg4s_features.npz", X=X, pid=pid, subjects=r4.SUBJ, nseg=nseg)
