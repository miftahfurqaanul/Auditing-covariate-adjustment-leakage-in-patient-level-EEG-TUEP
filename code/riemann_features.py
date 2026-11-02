"""C2: log-Euclidean tangent-space features of band-wise spatial covariance (no labels used).

Per clean 10 s window (same windows as features.py): band-pass in 5 bands, covariance of the 19-channel CAR
signal projected onto the 18-dimensional subspace orthogonal to the all-ones vector (CAR makes the full matrix
singular), shrinkage 1e-3 * trace/18. Matrices are averaged per recording; the patient matrix is the
window-count-weighted average over the patient's recordings; features = upper triangle of logm
(off-diagonals x sqrt(2)). Recording-level matrices are stored for the P4 analysis.
"""
import numpy as np, pandas as pd, h5py
from multiprocessing import Pool
from scipy.signal import butter, sosfiltfilt
from scipy.linalg import null_space
D = "/home/miftah_f/RTX-5090/tuep_fresh/data"
FS, WIN, EDGE, PTP = 200, 2000, 2000, 800.0
BANDS = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 45)]
U = null_space(np.ones((1, 19)))                       # 19 x 18 orthonormal basis orthogonal to the CAR direction
SOS = [butter(4, b, btype="bandpass", fs=FS, output="sos") for b in BANDS]
IU = np.triu_indices(18)
SCALE = np.where(IU[0] == IU[1], 1.0, np.sqrt(2.0))

def rec_cov(h5):
    with h5py.File(f"{D}/{h5}") as f:
        x = f["eeg"][()].astype(np.float64)
    x = x[:, EDGE:x.shape[1] - EDGE]; n = x.shape[1] // WIN
    if n == 0:
        return h5, None, 0
    w = x[:, :n * WIN].reshape(19, n, WIN).transpose(1, 0, 2)
    w = w[(w.max(2) - w.min(2)).max(1) <= PTP]
    if len(w) == 0:
        return h5, None, 0
    covs = []
    for sos in SOS:
        b = sosfiltfilt(sos, w, axis=2)
        b = np.einsum("ij,njt->nit", U.T, b)                # project onto the 18-d subspace
        c = np.einsum("nit,njt->nij", b, b) / WIN
        covs.append(c.mean(0))
    return h5, np.stack(covs), len(w)                    # [5, 18, 18], n windows

def logm_spd(C):
    C = C + 1e-3 * np.trace(C) / 18 * np.eye(18)
    e, V = np.linalg.eigh(C)
    return (V * np.log(e)) @ V.T

def tangent(Cb):
    return np.concatenate([logm_spd(C)[IU] * SCALE for C in Cb])

if __name__ == "__main__":
    meta = pd.read_csv(f"{D}/features_bandpower_recordings.csv"); meta = meta[meta.n_clean > 0].reset_index(drop=True)
    with Pool(12) as pool:
        res = pool.map(rec_cov, meta.h5_path, chunksize=4)
    covs = {h: (c, n) for h, c, n in res if c is not None}
    keep = meta.h5_path.isin(covs).values
    rec_feat = np.vstack([tangent(covs[h][0]) for h in meta.h5_path[keep]])
    np.savez(f"{D}/riemann_recordings.npz", X=rec_feat.astype(np.float32), h5_path=meta.h5_path[keep].values)
    first = (meta.groupby("subject").session_no.transform("min") == meta.session_no).values
    T = np.load(f"{D}/p3_patient_table.npz", allow_pickle=True)
    pt = []
    for s in T["subjects"]:
        hs = [h for h in meta.h5_path[first & (meta.subject.values == s)] if h in covs]
        w = np.array([covs[h][1] for h in hs], float)
        pt.append(tangent(np.tensordot(w / w.sum(), np.stack([covs[h][0] for h in hs]), axes=1)))
    np.savez(f"{D}/riemann_p3.npz", X=np.vstack(pt).astype(np.float32), subjects=T["subjects"])
    print("recordings:", keep.sum(), "| patients:", len(pt), "| features per unit:", rec_feat.shape[1])
