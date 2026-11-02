"""Round 17 P3 (reports/round17_P1-P5_prediction.md Bagian 5; choices in round17_implementation_notes.md): per-era re-centering of
band-wise spatial covariances with a reference taken from the non-epilepsy TRAINING patients only, then log-Euclidean tangent features.
Window covariances recomputed exactly as riemann_features.rec_cov but kept per window (first session). Reference = Euclidean mean of
training-control window covariances of the era (fallback (b): pyriemann not installed -> NOT a full Riemannian mean); era with < 10
training-control windows -> global training-control reference. C' = M^-1/2 C M^-1/2, tangent = riemann_features.logm_spd (upper triangle,
off-diagonal x sqrt 2), mean over the patient's windows (855) -> PCA 50 fitted on the training patients -> + age/sex -> evalkit.nested_lr.
Ablation P3-global (one reference for all eras); comparator Riemann without normative layer (riemann_p3.npz + age/sex, round 15).
Negative controls: year shuffle 20x, without <=2009. CPU only."""
import round17_common as c17
import os, time, numpy as np, pandas as pd, h5py
from multiprocessing import Pool
from joblib import Parallel, delayed
from scipy.signal import sosfiltfilt
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
import evalkit as ek, round4 as r4, riemann_features as rfe

Y, AS, N = c17.Y, c17.AS, c17.N
CACHE = f"{c17.OUT}/p3_wincov.npz"; NPC, MIN_WIN = 50, 10


def win_cov(h5):
    """riemann_features.rec_cov, but returning every clean window's covariance [n, 5, 18, 18] instead of their mean."""
    with h5py.File(f"{rfe.D}/{h5}") as f:
        x = f["eeg"][()].astype(np.float64)
    x = x[:, rfe.EDGE:x.shape[1] - rfe.EDGE]; n = x.shape[1] // rfe.WIN
    if n == 0:
        return h5, None
    w = x[:, :n * rfe.WIN].reshape(19, n, rfe.WIN).transpose(1, 0, 2)
    w = w[(w.max(2) - w.min(2)).max(1) <= rfe.PTP]
    if len(w) == 0:
        return h5, None
    covs = []
    for sos in rfe.SOS:
        b = sosfiltfilt(sos, w, axis=2); b = np.einsum("ij,njt->nit", rfe.U.T, b)
        covs.append(np.einsum("nit,njt->nij", b, b) / rfe.WIN)
    return h5, np.stack(covs, 1)


def build_cache(jobs):
    meta = pd.read_csv(f"{rfe.D}/features_bandpower_recordings.csv"); meta = meta[meta.n_clean > 0].reset_index(drop=True)
    first = (meta.groupby("subject").session_no.transform("min") == meta.session_no).values
    paths = {s: list(meta.h5_path[first & (meta.subject.values == s)]) for s in c17.SUBJ}
    with Pool(jobs) as pool:
        res = dict(pool.map(win_cov, [h for s in c17.SUBJ for h in paths[s]], chunksize=2))
    C, pid = [], []
    for i, s in enumerate(c17.SUBJ):
        for h in paths[s]:
            if res[h] is not None:
                C.append(res[h]); pid.append(np.full(len(res[h]), i))
    C, pid = np.concatenate(C), np.concatenate(pid)
    np.savez(CACHE, C=C, pid=pid); return C, pid


def load_cache(jobs):
    if os.path.exists(CACHE):
        z = np.load(CACHE); return z["C"], z["pid"]
    return build_cache(jobs)


def inv_sqrt(M):
    e, V = np.linalg.eigh(M); return (V / np.sqrt(e)) @ V.T, e.max() / e.min()


def tangent_windows(Cb):
    """riemann_features.logm_spd + upper triangle x SCALE, batched over windows (one band)."""
    S = Cb + (1e-3 * np.trace(Cb, axis1=1, axis2=2) / 18)[:, None, None] * np.eye(18)
    e, V = np.linalg.eigh(S); L = (V * np.log(e)[:, None, :]) @ V.transpose(0, 2, 1)
    return L[:, rfe.IU[0], rfe.IU[1]] * rfe.SCALE


def p3_features(C, pid, tr, yg, mode):
    """mode 'era': per-era reference from training-control windows; 'global': one reference; 'none': no re-centering."""
    c = tr[Y[tr] == 0]; cw = np.isin(pid, c); era = yg[pid] if mode == "era" else np.zeros(len(pid), int)
    out = np.empty((len(pid), 5 * 171)); conds, fallback = [], []
    for b in range(5):
        Cb = C[:, b]
        if mode == "none":
            out[:, b * 171:(b + 1) * 171] = tangent_windows(Cb); continue
        Mg = Cb[cw].mean(0); Rc = np.empty_like(Cb)
        for g in np.unique(era):
            sel = era == g; ref = cw & sel
            if ref.sum() < MIN_WIN:
                M = Mg; fallback.append((str(g), b))
            else:
                M = Cb[ref].mean(0)
            W, cond = inv_sqrt(M); conds.append(cond); Rc[sel] = W @ Cb[sel] @ W
        out[:, b * 171:(b + 1) * 171] = tangent_windows(Rc)
    cnt = np.bincount(pid, minlength=N); X = np.add.reduceat(out, np.r_[0, np.cumsum(cnt)[:-1]], axis=0) / cnt[:, None]
    return X, (max(conds) if conds else np.nan), fallback


def p3_predict(C, pid, tr, te, yg, mode):
    X, cond, fb = p3_features(C, pid, tr, yg, mode)
    pca = PCA(NPC, random_state=0).fit(X[tr]); F = np.hstack([pca.transform(X), AS])
    return ek.nested_lr().fit(F[tr], Y[tr]).predict_proba(F[te])[:, 1], cond, fb


def raw_riemann(tr, te, Xr):
    F = np.hstack([Xr, AS]); return ek.nested_lr().fit(F[tr], Y[tr]).predict_proba(F[te])[:, 1]


if __name__ == "__main__":
    a = c17.args(); c17.header("round17_p3"); t0 = time.time()
    C, pid = load_cache(a.jobs); cnt = np.bincount(pid, minlength=N)
    print(f"window covariances {C.shape} ({C.nbytes / 1e9:.2f} GB), patients with windows {int((cnt > 0).sum())}, per patient median {np.median(cnt):.0f} (min {cnt.min()}) | {time.time() - t0:.0f} s", flush=True)
    zr = np.load(f"{ek.D}/riemann_p3.npz", allow_pickle=True); assert list(zr["subjects"]) == list(c17.SUBJ); Xr = zr["X"].astype(np.float64)
    pm = np.add.reduceat(C, np.r_[0, np.cumsum(cnt)[:-1]], axis=0) / cnt[:, None, None, None]
    rec = np.vstack([rfe.tangent(m) for m in pm]); err = np.abs(rec - Xr).max()
    if not c17.pf(err < 1e-3, f"window covariances averaged per patient reproduce riemann_p3.npz (max |diff| {err:.1e}; stored as float32)"):
        raise SystemExit("STOP: covariance reconstruction failed")
    tr0 = c17.JOBS[0][2]; X0, cond0, fb0 = p3_features(C, pid, tr0, r4.YEAR_G, "era"); c = tr0[Y[tr0] == 0]
    print(f"   fold r0k0: max condition number of era references {cond0:.2e} (limit 1e8: {'OK' if cond0 < 1e8 else 'EXCEEDED'}); era fallbacks {fb0}")
    Xn, _, _ = p3_features(C, pid, tr0, r4.YEAR_G, "none")
    print(f"   check: per-window tangent without re-centering, patient mean vs tangent of patient mean (differ by construction; corr {np.corrcoef(Xn.ravel(), Xr.ravel())[0, 1]:.4f})")
    if a.dry_run:
        r, k, tr, te = c17.JOBS[0]; p, _, _ = p3_predict(C, pid, tr, te, r4.YEAR_G, "era"); pg, _, _ = p3_predict(C, pid, tr, te, r4.YEAR_G, "global")
        print(f"dry-run fold r{r}k{k}: P3 AUC {roc_auc_score(Y[te], p):.4f} | P3-global {roc_auc_score(Y[te], pg):.4f} | Riemann raw {roc_auc_score(Y[te], raw_riemann(tr, te, Xr)):.4f} | {time.time() - t0:.0f} s")
        raise SystemExit
    J = c17.JOBS; P = {}
    P["Riemann_raw"] = c17.to_matrix(J, Parallel(n_jobs=a.jobs)(delayed(raw_riemann)(tr, te, Xr) for _, _, tr, te in J))
    if not c17.pf(abs(c17.pooled_auc(P["Riemann_raw"]) - 0.61159) < 1e-4, f"Riemann without normative layer == 0.6116 (round 15) ({c17.pooled_auc(P['Riemann_raw']):.5f})"):
        raise SystemExit("STOP: reproduction failed")
    run = lambda yg, mode, J_=J: Parallel(n_jobs=a.jobs, max_nbytes="1M")(delayed(p3_predict)(C, pid, tr, te, yg, mode) for _, _, tr, te in J_)
    for name, mode in [("P3", "era"), ("P3_global", "global")]:
        out = run(r4.YEAR_G, mode); P[name] = c17.to_matrix(J, [o[0] for o in out])
        print(f"{name} AUC {c17.pooled_auc(P[name]):.5f} | max condition number over folds {max(o[1] for o in out):.2e} | folds with era fallback "
              f"{sum(1 for o in out if o[2])} ({sorted(set(f for o in out for f in o[2]))})", flush=True)
    shuf = []
    for i in range(c17.N_SHUF):
        shuf.append(c17.pooled_auc(c17.to_matrix(J, [o[0] for o in run(c17.shuffled_year(i), "era")]))); print(f"   year-shuffle {i:2d}: P3 AUC {shuf[-1]:.4f}", flush=True)
    shuf = np.array(shuf); a3 = c17.pooled_auc(P["P3"])
    print(f"year-shuffle null: mean {shuf.mean():.4f} sd {shuf.std(ddof=1):.4f} p95 {np.percentile(shuf, 95):.4f}; P3 > p95: {a3 > np.percentile(shuf, 95)}; "
          f"rank p = {(1 + (shuf >= a3).sum()) / (1 + len(shuf)):.3f}")
    J2 = c17.J169; P169 = {"P3": c17.to_matrix(J2, [o[0] for o in run(r4.YEAR_G, "era", J2)]),
                           "I2": c17.to_matrix(J2, Parallel(n_jobs=a.jobs)(delayed(r4.fit_predict)("I2", tr, te, 0) for _, _, tr, te in J2))}
    d, p = ek.corrected_ttest(c17.fold_aucs(J2, P169["P3"]), c17.fold_aucs(J2, P169["I2"]), 135, 34)
    print(f"without <=2009 (169): P3 {c17.pooled_auc(P169['P3']):.4f} | I2 {c17.pooled_auc(P169['I2']):.4f} | dAUC {d:+.4f} p {p:.4f}")
    np.savez(f"{c17.OUT}/p3.npz", P3=P["P3"], P3_global=P["P3_global"], Riemann_raw=P["Riemann_raw"], shuffle_auc=shuf, P3_169=P169["P3"], I2_169=P169["I2"])
    print(f"done in {(time.time() - t0) / 60:.1f} min")
