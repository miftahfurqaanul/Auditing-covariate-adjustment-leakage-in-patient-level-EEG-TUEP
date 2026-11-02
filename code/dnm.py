"""Round 6 ATM (reports/round6_atm_prediction.md, commit 4a7550e): dynamic network model (DNM) metrics per patient.
A per 500-ms sub-window (ridge least squares); neural fragility (Li et al. 2021; analytic minimum-norm rank-one row perturbation,
arXiv 2202.07026) and source/sink index (Gunnarsdottir et al. 2022). Same clean 10-s windows as the band-power features (features.py).
Usage: python dnm.py check | extract"""
import os, sys, numpy as np, pandas as pd, h5py
from multiprocessing import Pool
import features as fe
D = fe.D; SUB, MAXSUB, NW = 100, 1200, 64
OMEGAS = np.linspace(0, np.pi, NW)


def estimate_A(x):
    """x: (N, T) -> A with x(t+1) = A x(t), ridge lambda = 1e-3 * trace(XX^T)/N (CAR makes the channels rank-deficient)."""
    X, Yn = x[:, :-1], x[:, 1:]; G = X @ X.T; lam = 1e-3 * np.trace(G) / len(G)
    return Yn @ X.T @ np.linalg.inv(G + lam * np.eye(len(G)))


def fragility(A):
    """Minimum 2-norm of a real perturbation of row k that puts r = e^{jw} in the spectrum of A + e_k d^T, minimised over w.
    det(A - rI + e_k d^T) = 0  <=>  d^T u = -1 with u = (A - rI)^{-1} e_k  (matrix determinant lemma).
    Real d: [Re u, Im u]^T d = [-1, 0]  ->  minimum norm ||d|| = sqrt([(C^T C)^{-1}]_00), C = [Re u, Im u]."""
    N = len(A); I = np.eye(N); best = np.full(N, np.inf)
    Minv = np.linalg.inv(A[None] - np.exp(1j * OMEGAS)[:, None, None] * I[None])        # (W, N, N); column k = u_k
    re, im = Minv.real, Minv.imag
    g11, g22, g12 = (re ** 2).sum(1), (im ** 2).sum(1), (re * im).sum(1)                 # (W, N) per column k
    det = g11 * g22 - g12 ** 2
    real_case = g22 <= 1e-16 * g11                                   # r real (w = 0 or pi): Im u is numerically zero
    norm2 = np.where(real_case, 1.0 / g11, g22 / np.where(real_case, 1.0, det))
    return np.sqrt(norm2).min(0)


def source_sink(A):
    N = len(A); rr = (np.argsort(np.argsort(np.abs(A).sum(1))) + 1) / N; cr = (np.argsort(np.argsort(np.abs(A).sum(0))) + 1) / N
    sink = 2 - np.hypot(rr - 1, cr - 1 / N); source = 2 - np.hypot(rr - 1 / N, cr - 1)
    return source, sink


def subwindow_metrics(w):
    """w: (19, SUB) -> normalised fragility, source, sink (each length 19)."""
    A = estimate_A(w.astype(np.float64)); f = fragility(A); f = (f.max() - f) / f.max(); so, si = source_sink(A)
    return f, so, si


def patient_features(h5_paths):
    subs = []
    for h in h5_paths:
        with h5py.File(os.path.join(D, h)) as f:
            x = f["eeg"][()].astype(np.float32)
        x = x[:, fe.EDGE:x.shape[1] - fe.EDGE]; n = x.shape[1] // fe.WIN
        if n == 0:
            continue
        win = x[:, :n * fe.WIN].reshape(19, n, fe.WIN).transpose(1, 0, 2); win = win[(win.max(2) - win.min(2)).max(1) <= fe.PTP_MAX]
        for w in win:
            subs += [w[:, i:i + SUB] for i in range(0, fe.WIN, SUB)]
        if len(subs) >= MAXSUB:
            break
    subs = subs[:MAXSUB]; M = np.array([np.concatenate(subwindow_metrics(s)) for s in subs])
    return M.mean(0), len(subs)


def job(args):
    s, paths = args
    v, n = patient_features(paths); return s, v, n


if __name__ == "__main__":
    if sys.argv[1] == "check":
        rng = np.random.default_rng(0); ok = True
        for trial in range(20):
            A = rng.normal(size=(19, 19)) * 0.2; f = fragility(A); N = 19
            for k in [0, 7, 18]:
                ws = OMEGAS; best = None
                for w in ws:
                    u = np.linalg.solve(A - np.exp(1j * w) * np.eye(N), np.eye(N)[:, k]); C = np.column_stack([u.real, u.imag])
                    d = np.linalg.lstsq(C.T, np.array([-1.0, 0.0]), rcond=None)[0]
                    ev = np.linalg.eigvals(A + np.outer(np.eye(N)[k], d)); err = np.abs(ev - np.exp(1j * w)).min()
                    ok &= err < 1e-6 or abs(np.linalg.det(C.T @ C)) < 1e-12
                    nd = np.linalg.norm(d); best = nd if best is None else min(best, nd)
                    # any other feasible d' = d + n with n orthogonal to both constraints is longer
                    nvec = rng.normal(size=N); nvec -= C @ np.linalg.lstsq(C, nvec, rcond=None)[0]
                    ok &= np.linalg.norm(d + 0.1 * nvec) >= nd - 1e-12
                ok &= abs(best - f[k]) < 1e-6 * max(1, best)
        print(("PASS" if ok else "FAIL") + " fragility: r becomes an eigenvalue of A + e_k d^T, d is minimum-norm, closed form == explicit solve (20 random A)")
        A = rng.normal(size=(19, 19)) * 0.05; A[:, 3] *= 20; A[3, :] *= 0.05; A[11, :] *= 20; A[:, 11] *= 0.05
        # node 3: large column (influences others), small row -> ideal source; node 11: large row, small column -> ideal sink (Gunnarsdottir 2022)
        so, si = source_sink(A)
        print(("PASS" if so.argmax() == 3 and si.argmax() == 11 else "FAIL") + f" source-sink: strongest sender = source (argmax {so.argmax()}), strongest receiver = sink (argmax {si.argmax()})")
        x = rng.normal(size=(19, 3000)); A0 = rng.normal(size=(19, 19)) * 0.1
        for t in range(1, 3000):
            x[:, t] = A0 @ x[:, t - 1] + rng.normal(size=19)
        print(("PASS" if np.abs(estimate_A(x) - A0).max() < 0.1 else "FAIL") + f" A estimation recovers a simulated system (max abs error {np.abs(estimate_A(x) - A0).max():.3f})")
        sys.exit()
    rec = pd.read_csv(f"{D}/features_bandpower_recordings.csv"); rec = rec[rec.n_clean > 0]
    first = rec[rec.session_no == rec.groupby("subject").session_no.transform("min")]
    T = np.load(f"{D}/p3_patient_table.npz", allow_pickle=True); subj = T["subjects"]
    jobs = [(s, list(first[first.subject == s].sort_values("h5_path").h5_path)) for s in subj]
    with Pool(16) as pool:
        out = {s: (v, n) for s, v, n in pool.imap_unordered(job, jobs)}
    X = np.vstack([out[s][0] for s in subj]); n = np.array([out[s][1] for s in subj])
    np.savez(f"{D}/dnm_features.npz", subjects=subj, X=X, n_sub=n)
    print(f"patients {len(subj)}, features {X.shape}, finite {np.isfinite(X).all()}, sub-windows per patient: min {n.min()}, median {np.median(n):.0f}, max {n.max()}")
