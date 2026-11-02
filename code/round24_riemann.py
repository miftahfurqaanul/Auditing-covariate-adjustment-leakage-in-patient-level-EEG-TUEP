"""Round 24 R2 (reports/round24_prediction.md, commit 48092a0): P3 with a Riemannian (Karcher) mean reference per era from the training-control
windows (pyriemann 0.12, geometry.mean.mean_riemann, tol 1e-8, maxiter 50) plus dispersion stretching without rotation (Rodrigues et al. 2019):
per era and band, tangent vectors are scaled by sqrt(d*/d_g), d_g = mean squared tangent norm of the era's training-control windows and
d* = the same over all training-control windows. Otherwise identical to round17_p3 (cache, tangent_windows, PCA 50, age/sex, nested_lr).
P3-K = Karcher without stretching (secondary). CPU only.  Usage: python round24_riemann.py --dry-run | --run"""
import round17_common as c17
import os, sys, time, argparse, warnings, numpy as np
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
import evalkit as ek, round4 as r4, round17_p3 as p3
from pyriemann.geometry.mean import mean_riemann

warnings.filterwarnings("ignore"); Y, N, AS = c17.Y, c17.N, c17.AS
OUT = f"{ek.D}/round24"; os.makedirs(OUT, exist_ok=True)


def ks_features(C, pid, tr, yg):
    """Returns patient-mean tangent features for P3-KS (Karcher + stretch) and P3-K (Karcher only), plus diagnostics."""
    c = tr[Y[tr] == 0]; cw = np.isin(pid, c); era = yg[pid]; cnt = np.bincount(pid, minlength=N); starts = np.r_[0, np.cumsum(cnt)[:-1]]
    outK = np.empty((len(pid), 5 * 171)); outKS = np.empty_like(outK); diag = []
    for b in range(5):
        Cb = C[:, b]; Mg = mean_riemann(Cb[cw]); T = np.empty((len(pid), 171))
        for g in np.unique(era):
            sel = era == g; ref = cw & sel; M = Mg if ref.sum() < p3.MIN_WIN else mean_riemann(Cb[ref])
            W, _ = p3.inv_sqrt(M); T[sel] = p3.tangent_windows(W @ Cb[sel] @ W)
        outK[:, b * 171:(b + 1) * 171] = T
        dstar = (T[cw] ** 2).sum(1).mean(); TS = T.copy()
        for g in np.unique(era):
            sel = era == g; ref = cw & sel
            if ref.sum() >= p3.MIN_WIN:
                s = np.sqrt(dstar / (T[ref] ** 2).sum(1).mean()); TS[sel] = T[sel] * s; diag.append((str(g), b, s))
        outKS[:, b * 171:(b + 1) * 171] = TS
    mean = lambda O: np.add.reduceat(O, starts, axis=0) / cnt[:, None]
    return mean(outKS), mean(outK), diag


def predict(X, tr, te):
    pca = PCA(p3.NPC, random_state=0).fit(X[tr]); F = np.hstack([pca.transform(X), AS])
    return ek.nested_lr().fit(F[tr], Y[tr]).predict_proba(F[te])[:, 1]


def fold(C, pid, tr, te, yg):
    XKS, XK, diag = ks_features(C, pid, tr, yg); return predict(XKS, tr, te), predict(XK, tr, te), diag


def checks(C, pid):
    from scipy.linalg import logm
    tr = c17.JOBS[0][2]; c = tr[Y[tr] == 0]; cw = np.isin(pid, c); era = r4.YEAR_G[pid]; ref = cw & (era == "2011"); Cb = C[ref, 2]
    M = mean_riemann(Cb); W, _ = p3.inv_sqrt(M); L = np.array([logm(W @ x @ W).real for x in Cb])
    ok = c17.pf(np.abs(L.mean(0)).max() < 1e-6, f"Karcher property: mean log of re-centred reference windows ~ 0 (max {np.abs(L.mean(0)).max():.1e}; era 2011, band 3, {len(Cb)} windows)")
    XKS, XK, diag = ks_features(C, pid, tr, r4.YEAR_G)
    print(f"   stretching factors fold r0k0: min {min(d[2] for d in diag):.3f}, max {max(d[2] for d in diag):.3f}, n {len(diag)}")
    C2 = C.copy(); C2[np.isin(pid, c17.JOBS[0][3])] *= 3.0; XKS2, _, _ = ks_features(C2, pid, tr, r4.YEAR_G)
    ok &= c17.pf(np.allclose(XKS2[tr], XKS[tr]), "leakage: perturbing test covariances leaves P3-KS training features unchanged")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); g = ap.add_mutually_exclusive_group(required=True); g.add_argument("--dry-run", action="store_true"); g.add_argument("--run", action="store_true")
    ap.add_argument("--jobs", type=int, default=12); a = ap.parse_args(); c17.header("round24_riemann (pyriemann 0.12)"); t0 = time.time()
    C, pid = p3.load_cache(a.jobs)
    if not checks(C, pid):
        sys.exit("STOP")
    if a.dry_run:
        r, k, tr, te = c17.JOBS[0]; t1 = time.time(); pks, pk, _ = fold(C, pid, tr, te, r4.YEAR_G)
        print(f"dry-run fold r{r}k{k}: P3-KS {roc_auc_score(Y[te], pks):.4f} | P3-K {roc_auc_score(Y[te], pk):.4f} | fold {time.time() - t1:.0f} s"); sys.exit()
    J = c17.JOBS; run = lambda yg, J_=J: Parallel(n_jobs=a.jobs, max_nbytes="1M")(delayed(fold)(C, pid, tr, te, yg) for _, _, tr, te in J_)
    out = run(r4.YEAR_G); P = {"P3KS": c17.to_matrix(J, [o[0] for o in out]), "P3K": c17.to_matrix(J, [o[1] for o in out])}
    sf = np.array([d[2] for o in out for d in o[2]]); print(f"P3-KS AUC {c17.pooled_auc(P['P3KS']):.4f} | P3-K AUC {c17.pooled_auc(P['P3K']):.4f} | stretch factors {sf.min():.3f}-{sf.max():.3f}", flush=True)
    shuf = []
    for i in range(c17.N_SHUF):
        shuf.append(c17.pooled_auc(c17.to_matrix(J, [o[0] for o in run(c17.shuffled_year(i))]))); print(f"   year-shuffle {i:2d}: P3-KS AUC {shuf[-1]:.4f}", flush=True)
    P169 = c17.to_matrix(c17.J169, [o[0] for o in run(r4.YEAR_G, c17.J169)])
    np.savez(f"{OUT}/r2.npz", P3KS=P["P3KS"], P3K=P["P3K"], P3KS_169=P169, shuffle_auc=np.array(shuf), stretch=sf)
    print(f"done in {(time.time() - t0) / 60:.1f} min")
