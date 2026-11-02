"""Round 24 R1 (reports/round24_prediction.md, commit 48092a0): fourth view = debiased wPLI (Vinck et al. 2011) with a pre-registered gate.
Features (label-free): clean 10-s first-session windows (features.py definition) cut into 5 non-overlapping 2-s epochs, Hann taper, FFT;
per patient, channel pair (171) and frequency bin: debiased wPLI^2 over all epochs; mean over bins in alpha [8,13) and beta [13,30) -> 342.
Descriptive comparator PLV. Model W: normative z (I2 design, training controls) -> PCA 50 (training) + age/sex -> evalkit.nested_lr.
Gate: inner-OOF AUC >= 0.60 and Spearman(inner-OOF W, inner-OOF I2) <= 0.70 (means over 50 outer folds; training data only). CPU only.
Usage: python round24_wpli.py --extract | --dry-run | --run"""
import round17_common as c17
import os, sys, time, argparse, numpy as np, pandas as pd, h5py
from multiprocessing import Pool
from joblib import Parallel, delayed
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
import evalkit as ek, round4 as r4, features as fe, round19_fusion as f19

Y, N, AS, SUBJ = c17.Y, c17.N, c17.AS, c17.SUBJ
OUT = f"{ek.D}/round24"; os.makedirs(OUT, exist_ok=True); CACHE = f"{OUT}/wpli_features.npz"
EP = 400; FREQ = np.fft.rfftfreq(EP, 1 / fe.FS); BANDS = {"alpha": (8, 13), "beta": (13, 30)}
SEL = (FREQ >= 8) & (FREQ < 30); FS_ = FREQ[SEL]; IU = np.triu_indices(19, 1); NPC = 50
HANN = np.hanning(EP)


def patient_sums(paths):
    """Accumulate sum Im, sum Im^2, sum |Im| and sum of unit phasors of the cross-spectra over all epochs of the patient's clean windows."""
    s1 = s2 = s3 = 0.0; ph = 0.0; n = 0
    for h in paths:
        with h5py.File(f"{fe.D}/{h}") as f:
            x = f["eeg"][()].astype(np.float64)
        x = x[:, fe.EDGE:x.shape[1] - fe.EDGE]; k = x.shape[1] // fe.WIN
        if k == 0:
            continue
        w = x[:, :k * fe.WIN].reshape(19, k, fe.WIN).transpose(1, 0, 2); w = w[(w.max(2) - w.min(2)).max(1) <= fe.PTP_MAX]
        if len(w) == 0:
            continue
        e = w.reshape(len(w), 19, fe.WIN // EP, EP).transpose(0, 2, 1, 3).reshape(-1, 19, EP)        # epochs (n, 19, 400)
        X = np.fft.rfft(e * HANN, axis=2)[:, :, SEL]                                                  # (n, 19, nbins)
        S = X[:, IU[0], :] * np.conj(X[:, IU[1], :]); im = S.imag                                     # (n, 171, nbins)
        s1 = s1 + im.sum(0); s2 = s2 + (im ** 2).sum(0); s3 = s3 + np.abs(im).sum(0); ph = ph + (S / np.maximum(np.abs(S), 1e-300)).sum(0); n += len(e)
    return s1, s2, s3, ph, n


def patient_features(paths):
    s1, s2, s3, ph, n = patient_sums(paths)
    wpli = (s1 ** 2 - s2) / np.maximum(s3 ** 2 - s2, 1e-300); plv = np.abs(ph) / n
    band = lambda v: np.concatenate([v[:, (FS_ >= lo) & (FS_ < hi)].mean(1) for lo, hi in BANDS.values()])
    return band(wpli), band(plv), n


def first_session_paths():
    meta = pd.read_csv(f"{fe.D}/features_bandpower_recordings.csv"); meta = meta[meta.n_clean > 0]
    first = meta.session_no == meta.groupby("subject").session_no.transform("min")
    return [sorted(meta[first & (meta.subject == s)].h5_path) for s in SUBJ]


def wpli_manual_check(paths):
    """Second implementation (explicit loops, one pair, one bin) of debiased wPLI for the first patient."""
    i, j = 3, 15; b = int(np.where(SEL)[0][10]); ims = []
    for h in paths:
        with h5py.File(f"{fe.D}/{h}") as f:
            x = f["eeg"][()].astype(np.float64)
        x = x[:, fe.EDGE:x.shape[1] - fe.EDGE]
        for s0 in range(0, (x.shape[1] // fe.WIN) * fe.WIN, fe.WIN):
            w = x[:, s0:s0 + fe.WIN]
            if (w.max(1) - w.min(1)).max() > fe.PTP_MAX:
                continue
            for e0 in range(0, fe.WIN, EP):
                xi = np.fft.rfft(w[i, e0:e0 + EP] * HANN)[b]; xj = np.fft.rfft(w[j, e0:e0 + EP] * HANN)[b]; ims.append((xi * np.conj(xj)).imag)
    ims = np.array(ims); return (ims.sum() ** 2 - (ims ** 2).sum()) / (np.abs(ims).sum() ** 2 - (ims ** 2).sum())


def normative_z(X, design, tr):
    c = tr[Y[tr] == 0]; B = np.linalg.lstsq(design[c], X[c], rcond=None)[0]
    return (X - design @ B) / (X[c] - design[c] @ B).std(0, ddof=np.linalg.matrix_rank(design[c]))


def w_predict(X, tr, te, design=None):
    Z = normative_z(X, r4.DESIGN["I2"] if design is None else design, tr); pca = PCA(NPC, random_state=0).fit(Z[tr])
    F = np.hstack([pca.transform(Z), AS]); return ek.nested_lr().fit(F[tr], Y[tr]).predict_proba(F[te])[:, 1]


def inner_oof(X, r, k, tr):
    o = np.full(N, np.nan)
    for ia, ib in f19.inner_folds(r, k, tr):
        o[tr[ib]] = w_predict(X, tr[ia], tr[ib])
    return o


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); g = ap.add_mutually_exclusive_group(required=True)
    for f_ in ("--extract", "--dry-run", "--run"):
        g.add_argument(f_, action="store_true")
    ap.add_argument("--jobs", type=int, default=12); a = ap.parse_args(); c17.header("round24_wpli"); t0 = time.time()
    if a.extract:
        P = first_session_paths()
        with Pool(a.jobs) as pool:
            res = pool.map(patient_features, P, chunksize=2)
        W, PLV, NE = np.vstack([r[0] for r in res]), np.vstack([r[1] for r in res]), np.array([r[2] for r in res])
        man = wpli_manual_check(P[0]); s1, s2, s3, _, _ = patient_sums(P[0]); pair = [(int(a_), int(b_)) for a_, b_ in zip(*IU)].index((3, 15)); auto = ((s1 ** 2 - s2) / (s3 ** 2 - s2))[pair, 10]
        c17.pf(abs(man - auto) < 1e-9, f"debiased wPLI of pair (3,15), bin {FS_[10]} Hz, patient 0: vectorised {auto:.6f} == explicit-loop {man:.6f}")
        c17.pf(np.isfinite(W).all() and np.isfinite(PLV).all() and W.shape == (N, 342), f"features finite; shape {W.shape}; epochs per patient median {np.median(NE):.0f} (min {NE.min()})")
        np.savez(CACHE, W=W, PLV=PLV, n_epochs=NE, subjects=SUBJ); print(f"extracted in {(time.time() - t0) / 60:.1f} min"); sys.exit()
    z = np.load(CACHE, allow_pickle=True); assert list(z["subjects"]) == list(SUBJ); W, PLV = z["W"], z["PLV"]
    J = c17.JOBS; tr0 = J[0][2]; Z0 = normative_z(W, r4.DESIGN["I2"], tr0); c0 = tr0[Y[tr0] == 0]
    c17.pf(np.abs(Z0[c0].mean(0)).max() < 1e-8, f"W normative z of training controls has mean 0 ({np.abs(Z0[c0].mean(0)).max():.1e})")
    rng = np.random.default_rng(0); W2 = W.copy(); te0 = J[0][3]; W2[te0] += rng.normal(0, 1, W2[te0].shape)
    Zp = normative_z(W2, r4.DESIGN["I2"], tr0); c17.pf(np.allclose(Zp[tr0], Z0[tr0]), "leakage: perturbing test patients leaves W training features unchanged")
    if a.dry_run:
        r, k, tr, te = J[0]; o = inner_oof(W, r, k, tr)
        print(f"dry-run fold r{r}k{k}: W test AUC {roc_auc_score(Y[te], w_predict(W, tr, te)):.4f} | inner-OOF AUC {roc_auc_score(Y[tr], o[tr]):.4f} | {time.time() - t0:.0f} s"); sys.exit()
    # gate (training data only)
    O = np.vstack(Parallel(n_jobs=a.jobs)(delayed(inner_oof)(W, r, k, tr) for r, k, tr, _ in J)); I2in = np.load(f"{ek.D}/round19/cpu_inner.npz")["I2"]
    g_auc = np.array([roc_auc_score(Y[tr], O[f, tr]) for f, (_, _, tr, _) in enumerate(J)]); g_rho = np.array([spearmanr(O[f, tr], I2in[f, tr]).correlation for f, (_, _, tr, _) in enumerate(J)])
    gate = bool(g_auc.mean() >= 0.60 and g_rho.mean() <= 0.70)
    print(f"GATE: inner-OOF AUC mean {g_auc.mean():.4f} (>= 0.60: {g_auc.mean() >= 0.60}); Spearman with I2 inner-OOF mean {g_rho.mean():.4f} (<= 0.70: {g_rho.mean() <= 0.70}) -> {'PASSED' if gate else 'FAILED'}", flush=True)
    P = {"W": c17.to_matrix(J, Parallel(n_jobs=a.jobs)(delayed(w_predict)(W, tr, te) for _, _, tr, te in J)),
         "PLV": c17.to_matrix(J, Parallel(n_jobs=a.jobs)(delayed(w_predict)(PLV, tr, te) for _, _, tr, te in J))}
    print(f"W AUC {c17.pooled_auc(P['W']):.4f} | PLV AUC {c17.pooled_auc(P['PLV']):.4f}", flush=True)
    shuf = []
    for i in range(c17.N_SHUF):
        d = c17.year_design(c17.shuffled_year(i)); shuf.append(c17.pooled_auc(c17.to_matrix(J, Parallel(n_jobs=a.jobs)(delayed(w_predict)(W, tr, te, d) for _, _, tr, te in J))))
        print(f"   year-shuffle {i:2d}: W AUC {shuf[-1]:.4f}", flush=True)
    W169 = c17.to_matrix(c17.J169, Parallel(n_jobs=a.jobs)(delayed(w_predict)(W, tr, te) for _, _, tr, te in c17.J169))
    np.savez(f"{OUT}/r1.npz", W=P["W"], PLV=P["PLV"], W_169=W169, inner_oof=O, gate_auc=g_auc, gate_rho=g_rho, gate=gate, shuffle_auc=np.array(shuf))
    print(f"done in {(time.time() - t0) / 60:.1f} min")
