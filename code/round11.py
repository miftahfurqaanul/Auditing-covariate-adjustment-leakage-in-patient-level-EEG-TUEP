"""Round 11 (reports/round11_prediction.md, commit e2631e5; execution instructions in the project reports).
E1a LMM random-intercept year harmonisation, E1b ComBat fitted on training controls only, E2 specparam normative features,
E6 beta/gamma sensitivity, E11 number-of-controls curve, E12 temporal recalibration. CPU only, one process.
Usage: python round11.py --dry-run | --jobs 12"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import sys, time, warnings, argparse, numpy as np, pandas as pd, h5py
from joblib import Parallel, delayed
from scipy import stats, optimize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import evalkit as ek, round4 as r4, round9 as r9, round10 as r10, features as fe
warnings.filterwarnings("ignore"); R = "/home/miftah_f/RTX-5090/tuep_fresh/reports"
SUBJ, Y, BP, AGESEX = r4.SUBJ, r4.Y, r4.BP, r4.AGESEX; S = r4.S; YEAR = S.year.values.astype(int); N = len(Y)
POS = {s: i for i, s in enumerate(SUBJ)}; F = ek.folds()
JOBS = [(r, k, np.setdiff1d(np.arange(N), te), te) for r in range(10) for k in range(5)
        for te in [np.array([POS[s] for s in F[(F.repeat == r) & (F.fold == k)].subject])]]
BETA_GAMMA = list(range(57, 95)); NO_BG = [i for i in range(114) if i not in BETA_GAMMA]       # rel_beta 57-75, rel_gamma 76-94
assert fe.NAMES[57].startswith("rel_beta_") and fe.NAMES[76].startswith("rel_gamma_") and fe.NAMES[95].startswith("logtot_") and len(NO_BG) == 76
I1D = np.column_stack([np.ones(N), r4.AGE, r4.MALE])
LOG = lambda *a: print(*a, flush=True)


# ------------------------------------------------------------------ generic normative model (OLS on training controls)
def ols_z(X, design, fit_idx):
    B = np.linalg.lstsq(design[fit_idx], X[fit_idx], rcond=None)[0]
    sd = (X[fit_idx] - design[fit_idx] @ B).std(0, ddof=np.linalg.matrix_rank(design[fit_idx])); return (X - design @ B) / sd


def lr(X, tr, te):
    return ek.nested_lr().fit(X[tr], Y[tr]).predict_proba(X[te])[:, 1]


def c1_cols(tr, te, cols):
    return lr(np.hstack([BP[:, cols], AGESEX]), tr, te)


def i2_cols(tr, te, cols):
    c = tr[Y[tr] == 0]; return lr(np.hstack([ols_z(BP, r4.DESIGN["I2"], c)[:, cols], AGESEX]), tr, te)


# ------------------------------------------------------------------ E1a: random-intercept LMM per feature (controls only)
def lmm_z(X, years, c, return_info=False):
    import statsmodels.api as sm
    from statsmodels.tools.sm_exceptions import ConvergenceWarning
    Z = np.zeros_like(X); nfb = 0; shrunk = []
    for j in range(X.shape[1]):
        xc, yc = X[c, j], years[c]; ok = True
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                res = sm.MixedLM(xc, I1D[c], groups=yc).fit(reml=True)
                fe_ = np.asarray(res.fe_params); blup = {g: float(np.asarray(v)[0]) for g, v in res.random_effects.items()}
                if any(issubclass(x.category, ConvergenceWarning) for x in w) or not np.all(np.isfinite(fe_)):
                    ok = False
            except Exception:
                ok = False
        if not ok:                                   # fallback: manual EB shrinkage (method of moments), logged
            nfb += 1; fe_ = np.linalg.lstsq(I1D[c], xc, rcond=None)[0]; rr = xc - I1D[c] @ fe_
            g = np.unique(yc); nk = np.array([(yc == k).sum() for k in g]); mk = np.array([rr[yc == k].mean() for k in g])
            s2 = np.sum([((rr[yc == k] - rr[yc == k].mean()) ** 2).sum() for k in g]) / max(len(rr) - len(g), 1)
            tau2 = max(0.0, mk.var(ddof=1) - np.mean(s2 / nk)) if len(g) > 1 else 0.0
            blup = {k: (n * tau2 / (n * tau2 + s2) if tau2 > 0 else 0.0) * m for k, n, m in zip(g, nk, mk)}
        pred = I1D @ fe_ + np.array([blup.get(v, 0.0) for v in years])
        sd = (xc - pred[c]).std(ddof=I1D.shape[1]); Z[:, j] = (X[:, j] - pred) / sd
        if return_info:
            rr = xc - I1D[c] @ np.linalg.lstsq(I1D[c], xc, rcond=None)[0]; g = np.unique(yc)
            shrunk.append(np.var([blup.get(k, 0.0) for k in g]) <= np.var([rr[yc == k].mean() for k in g]) + 1e-15)
    return (Z, nfb, np.mean(shrunk) if shrunk else None) if return_info else Z


def e1a(tr, te, years=YEAR):
    c = tr[Y[tr] == 0]; return lr(np.hstack([lmm_z(BP, years, c), AGESEX]), tr, te)


# ------------------------------------------------------------------ E1b: ComBat (Johnson et al. 2007) fitted on training controls only
def combat_fit(Xc, bc, covc, tol=1e-4, max_iter=1000):
    batches = np.unique(bc); n = len(Xc); Bd = np.column_stack([(bc == b).astype(float) for b in batches]); D = np.hstack([Bd, covc])
    Bh = np.linalg.lstsq(D, Xc, rcond=None)[0]; nb = Bd.sum(0)
    grand = (nb / n) @ Bh[:len(batches)]; var_pooled = ((Xc - D @ Bh) ** 2).mean(0); bcov = Bh[len(batches):]
    s = (Xc - grand - covc @ bcov) / np.sqrt(var_pooled)
    g_hat = np.array([s[bc == b].mean(0) for b in batches])
    d_hat = np.array([s[bc == b].var(0, ddof=1) if (bc == b).sum() >= 2 else np.full(s.shape[1], np.nan) for b in batches])
    g_bar, t2 = g_hat.mean(1), g_hat.var(1, ddof=1)
    multi = [i for i, b in enumerate(batches) if (bc == b).sum() >= 2]
    m_, v_ = d_hat[multi].mean(1), d_hat[multi].var(1, ddof=1); a_p, b_p = (2 * v_ + m_ ** 2) / v_, (m_ * v_ + m_ ** 3) / v_
    a_d = {i: a for i, a in zip(multi, a_p)}; b_d = {i: b for i, b in zip(multi, b_p)}
    g_star, d_star = np.zeros_like(g_hat), np.ones_like(g_hat)
    for i, b in enumerate(batches):
        sb = s[bc == b]; nk = len(sb)
        if nk >= 2:
            g_old, d_old = g_hat[i].copy(), d_hat[i].copy()
            for _ in range(max_iter):
                g_new = (nk * t2[i] * g_hat[i] + d_old * g_bar[i]) / (nk * t2[i] + d_old)
                d_new = (0.5 * ((sb - g_new) ** 2).sum(0) + b_d[i]) / (nk / 2 + a_d[i] - 1)
                done = max(np.max(np.abs(g_new - g_old) / np.abs(g_old).clip(1e-12)), np.max(np.abs(d_new - d_old) / d_old)) < tol
                g_old, d_old = g_new, d_new
                if done:
                    break
            g_star[i], d_star[i] = g_old, d_old
        else:                                        # 1 control in this year: location only (delta* = 1), as pre-registered
            g_star[i] = (t2[i] * g_hat[i] + g_bar[i]) / (t2[i] + 1.0)
    return dict(batches=batches, grand=grand, var=var_pooled, bcov=bcov, g=g_star, d=d_star)


def combat_apply(P, X, b, cov):
    out = X.copy(); idx = {bb: i for i, bb in enumerate(P["batches"])}
    for j in range(len(X)):
        if b[j] in idx:                              # years without training controls are left unadjusted (gamma* = 0, delta* = 1)
            i = idx[b[j]]; m = P["grand"] + cov[j] @ P["bcov"]
            out[j] = (X[j] - m) / np.sqrt(P["var"]) - P["g"][i]
            out[j] = out[j] / np.sqrt(P["d"][i]) * np.sqrt(P["var"]) + m
    return out


def e1b_features(tr, years=YEAR):
    c = tr[Y[tr] == 0]; P = combat_fit(BP[c], years[c], AGESEX[c]); H = combat_apply(P, BP, years, AGESEX)
    return np.hstack([ols_z(H, I1D, c), AGESEX]), H, c


def e1b(tr, te, years=YEAR):
    X, _, _ = e1b_features(tr, years); return lr(X, tr, te)


# ------------------------------------------------------------------ E2: specparam features (first-session windows, frozen settings)
POST = [fe.CH.index(c) for c in ["O1", "O2", "P3", "P4"]]


def specparam_patient(paths, max_windows=None):
    from specparam import SpectralModel
    from scipy.signal import welch
    sm_ = SpectralModel(aperiodic_mode="fixed", peak_width_limits=[1, 8], max_n_peaks=8, min_peak_height=0.1, verbose=False)
    ex, off, acf, apw = [], [], [], []; nwin = 0
    for h in paths:
        with h5py.File(os.path.join(fe.D, h)) as f:
            x = f["eeg"][()].astype(np.float32)
        x = x[:, fe.EDGE:x.shape[1] - fe.EDGE]; n = x.shape[1] // fe.WIN
        if n == 0:
            continue
        w = x[:, :n * fe.WIN].reshape(19, n, fe.WIN).transpose(1, 0, 2); w = w[(w.max(2) - w.min(2)).max(1) <= fe.PTP_MAX]
        for win in w:
            if max_windows is not None and nwin >= max_windows:
                break
            fr, P = welch(win, fs=fe.FS, nperseg=400, axis=1); nwin += 1; e_ch, o_ch = [], []
            for ch in range(19):
                try:
                    sm_.fit(fr, P[ch], [1, 45]); o, e = sm_.get_params("aperiodic"); e_ch.append(e); o_ch.append(o)
                except Exception:
                    pass
            ex.append(np.nanmedian(e_ch) if e_ch else np.nan); off.append(np.nanmedian(o_ch) if o_ch else np.nan)
            try:
                sm_.fit(fr, P[POST].mean(0), [1, 45]); pk = np.atleast_2d(sm_.get_params("periodic"))
                pk = pk[np.isfinite(pk[:, 0]) & (pk[:, 0] >= 8) & (pk[:, 0] <= 13)]
                if len(pk):
                    best = pk[np.argmax(pk[:, 1])]; acf.append(best[0]); apw.append(best[1])
            except Exception:
                pass
    med = lambda v: float(np.nanmedian(v)) if len(v) and np.isfinite(v).any() else np.nan
    return [med(ex), med(off), med(acf), med(apw)], nwin


def specparam_table(jobs, max_windows=None):
    rec = pd.read_csv(f"{fe.D}/features_bandpower_recordings.csv"); rec = rec[rec.n_clean > 0]
    first = rec[rec.session_no == rec.groupby("subject").session_no.transform("min")]
    paths = [list(first[first.subject == s].sort_values("h5_path").h5_path) for s in SUBJ]
    out = Parallel(n_jobs=jobs)(delayed(specparam_patient)(p, max_windows) for p in paths)
    return np.array([o[0] for o in out]), np.array([o[1] for o in out])


def e2(tr, te, SP):
    c = tr[Y[tr] == 0]; SPi = SP.copy()
    for j in range(SP.shape[1]):                        # NaN -> median of training controls (inside the fold)
        SPi[np.isnan(SPi[:, j]), j] = np.nanmedian(SP[c, j])
    X = np.hstack([BP, SPi]); return lr(np.hstack([ols_z(X, r4.DESIGN["I2"], c), AGESEX]), tr, te)


# ------------------------------------------------------------------ E11: number of training controls per year group
def e11_job(r, k, tr, te, n, seed):
    rng = np.random.default_rng(100000 * seed + 1000 * n + 10 * r + k); c = tr[Y[tr] == 0]
    sub = np.concatenate([rng.permutation(c[r4.YEAR_G[c] == g])[:n] for g in np.unique(r4.YEAR_G[c])])   # all if group < n
    return lr(np.hstack([ols_z(BP, r4.DESIGN["I2"], sub), AGESEX]), tr, te)


# ------------------------------------------------------------------ evaluation helpers
def per_repeat(preds, jobs=JOBS):
    P = np.zeros((10, N))
    for (r, _, _, te), p in zip(jobs, preds):
        P[r, te] = p
    return P


def summary(P):
    rows = []
    for p in P:
        ci, cs = r4.calib(Y, p); rows.append(dict(auc=roc_auc_score(Y, p), **r4.binm(Y, p > 0.5), brier=np.mean((p - Y) ** 2), slope=cs))
    return pd.DataFrame(rows).mean()


def fold_aucs(preds, jobs=JOBS):
    return np.array([roc_auc_score(Y[te], p) for (_, _, _, te), p in zip(jobs, preds)])


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); h = np.empty(len(p)); run = 0
    for rank, i in enumerate(o):
        run = max(run, min(1, p[i] * (len(p) - rank))); h[i] = run
    return h


def run_method(fn, jobs, n_jobs, **kw):
    return Parallel(n_jobs=n_jobs)(delayed(fn)(tr, te, **kw) for _, _, tr, te in jobs)


# ------------------------------------------------------------------ E12 temporal recalibration
def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6); return np.log(p / (1 - p))


def intercept_only(p, y):
    o = logit(p); f = lambda a: -np.sum(y * (a + o) - np.log1p(np.exp(a + o)))
    return optimize.minimize_scalar(f, bounds=(-10, 10), method="bounded").x


def e12(n_splits, rng_seed):
    tr = np.where(YEAR <= 2011)[0]; te = np.where(YEAR > 2011)[0]
    g = r4.YEAR_G; present = sorted(set(g[tr])); ref = pd.Series(g[tr]).value_counts().idxmax()
    D = np.column_stack([np.ones(N), r4.AGE, r4.MALE] + [(g == l).astype(float) for l in present if l != ref])
    c = tr[Y[tr] == 0]; p = lr(np.hstack([ols_z(BP, D, c), AGESEX]), tr, te); yt = Y[te]
    LOG(f"   E12: train <=2011 n={len(tr)} (controls {len(c)}), test >2011 n={len(te)}; reference group {ref}; test AUC {roc_auc_score(yt, p):.4f}")
    rng = np.random.default_rng(rng_seed); rows = []; mono_ok = True; platt_nonmono = 0
    for ncal in (30, 50):
        for sp in range(n_splits):
            cal = np.concatenate([rng.choice(np.where(yt == cl)[0], int(round(ncal * (yt == cl).mean())), replace=False) for cl in (0, 1)])
            rest = np.setdiff1d(np.arange(len(te)), cal); yc_perm = rng.permutation(yt[cal])
            variants = {"none": p}
            for tag, ycal in [("", yt[cal]), ("_permuted", yc_perm)]:
                a = intercept_only(p[cal], ycal); variants[f"intercept{tag}"] = 1 / (1 + np.exp(-(a + logit(p))))
                m = LogisticRegression(penalty=None).fit(logit(p[cal])[:, None], ycal); variants[f"platt{tag}"] = m.predict_proba(logit(p)[:, None])[:, 1]
                if tag == "" and m.coef_[0, 0] <= 0:
                    platt_nonmono += 1
            base_auc = roc_auc_score(yt[rest], p[rest])
            mono_ok &= abs(roc_auc_score(yt[rest], variants["intercept"][rest]) - base_auc) < 1e-12
            for name, q in variants.items():
                qr, yr = q[rest], yt[rest]; ci_, cs_ = r4.calib(yr, qr); bm = r4.binm(yr, qr > 0.5)
                rows.append(dict(n_cal=ncal, split=sp, method=name, brier=np.mean((qr - yr) ** 2), slope=cs_, intercept=ci_, sens=bm["sens"], spec=bm["spec"], auc_rest=roc_auc_score(yr, qr)))
    LOG(("PASS" if mono_ok else "FAIL") + " recalibration leaves every fold AUC unchanged (monotone) [intercept-only, all splits]; "
        f"Platt with b <= 0 in {platt_nonmono} of {2 * n_splits} splits")
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ main
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--dry-run", action="store_true"); ap.add_argument("--jobs", type=int, default=12); a = ap.parse_args()
    t0 = time.time(); NJ = a.jobs; dry = a.dry_run; J = JOBS[:1] if dry else JOBS
    LOG(f"round11 {'DRY-RUN (fold r0k0 only)' if dry else 'FULL'}; jobs {NJ}; CUDA_VISIBLE_DEVICES='{os.environ.get('CUDA_VISIBLE_DEVICES')}'")
    import specparam; LOG(f"specparam {specparam.__version__}; years with training controls (fold 0): "
                          f"{pd.Series(YEAR[JOBS[0][2][Y[JOBS[0][2]] == 0]]).value_counts().sort_index().to_dict()}")
    # ---- checks on fold 0
    tr0 = JOBS[0][2]; c0 = tr0[Y[tr0] == 0]
    z = ols_z(BP, r4.DESIGN["I2"], c0); LOG(("PASS" if np.abs(z[c0].mean(0)).max() < 1e-8 else "FAIL") + f" normative z of training controls has mean 0 (I2-style OLS, max |mean| {np.abs(z[c0].mean(0)).max():.1e})")
    LOG(("PASS" if np.allclose(z, r4.normative_z(r4.DESIGN["I2"], tr0)) else "FAIL") + " ols_z == round4.normative_z")
    t = time.time(); Za, nfb, shr = lmm_z(BP, YEAR, c0, return_info=True)
    LOG(f"   E1a fold 0: {time.time() - t:.1f} s; MixedLM fallbacks {nfb}/114; features whose between-year spread shrank vs OLS year means {100 * shr:.0f}%; "
        f"max |mean z| of training controls {np.abs(Za[c0].mean(0)).max():.2e} (GLS weighting: not exactly 0, reported)")
    Xb, H, _ = e1b_features(tr0); LOG(("PASS" if np.abs(Xb[c0, :114].mean(0)).max() < 1e-8 else "FAIL") + f" E1b: z of training controls after ComBat has mean 0 (max {np.abs(Xb[c0, :114].mean(0)).max():.1e})")
    unadj = ~np.isin(YEAR, np.unique(YEAR[c0])); LOG(("PASS" if np.allclose(H[unadj], BP[unadj]) else "FAIL") + f" E1b: {unadj.sum()} patients in years without training controls left unadjusted")
    cases0 = tr0[Y[tr0] == 1]; tr_half = np.sort(np.concatenate([c0, np.random.default_rng(0).choice(cases0, len(cases0) // 2, replace=False)]))
    zc, zcase = r4.normative_z(r4.DESIGN["I2"], tr0), r4.normative_z(r4.DESIGN["I2"], tr_half)            # E11 method control
    LOG(("PASS" if np.array_equal(zc, zcase) else "FAIL") + f" E11 method control: removing {len(cases0) - len(cases0) // 2} of {len(cases0)} training cases leaves the normative fit unchanged")
    # ---- reproduction (fold-level in dry run, pooled in full run)
    ref = np.load(f"{R}/round7_fold_aucs.npz")
    P = {}; FA = {}
    for name, fn, kw in [("C1", c1_cols, dict(cols=list(range(114)))), ("I2", i2_cols, dict(cols=list(range(114))))]:
        out = run_method(fn, J, NJ, **kw); FA[name] = fold_aucs(out, J)
        if dry:
            LOG(("PASS" if abs(FA[name][0] - ref[name][0]) < 1e-12 else "FAIL") + f" {name} fold r0k0 AUC {FA[name][0]:.4f} == round 7 {ref[name][0]:.4f}")
        else:
            P[name] = per_repeat(out)
    if not dry:
        sC, sI = summary(P["C1"]), summary(P["I2"])
        okC, okI = abs(sC.auc - 0.66627) < 1e-5, abs(sI.auc - 0.71210) < 1e-5
        LOG(("PASS" if okC else "FAIL") + f" C1 reproduces 0.66627 ({sC.auc:.5f})"); LOG(("PASS" if okI else "FAIL") + f" I2 reproduces 0.71210 ({sI.auc:.5f})")
        if not (okC and okI):
            LOG("STOP: reproduction failed"); sys.exit(1)
    # ---- E2 features
    t = time.time(); SP, nw = specparam_table(NJ, max_windows=5 if dry else None)
    LOG(f"   E2 specparam: {time.time() - t:.0f} s; windows per patient median {np.median(nw):.0f} (min {nw.min()}); NaN per feature {np.isnan(SP).sum(0).tolist()}")
    LOG(f"   E2 feature medians over patients: exponent {np.nanmedian(SP[:, 0]):.3f}, offset {np.nanmedian(SP[:, 1]):.3f}, alpha CF {np.nanmedian(SP[:, 2]):.2f} Hz, alpha PW {np.nanmedian(SP[:, 3]):.3f}")
    rho = stats.spearmanr(SP[tr0, 0], r4.AGE[tr0], nan_policy="omit")[0]; rho_all = stats.spearmanr(SP[:, 0], r4.AGE, nan_policy="omit")[0]
    LOG(("PASS" if rho < 0 else "FAIL") + f" E2 sanity: Spearman(exponent, age) training fold 0 = {rho:.3f}; all 200 = {rho_all:.3f}")
    # ---- methods
    methods = {"E1a": (e1a, {}), "E1b": (e1b, {}), "E2": (e2, dict(SP=SP)),
               "E6a-I2": (i2_cols, dict(cols=NO_BG)), "E6a-C1": (c1_cols, dict(cols=NO_BG)),
               "E6b-I2": (i2_cols, dict(cols=BETA_GAMMA)), "E6b-C1": (c1_cols, dict(cols=BETA_GAMMA))}
    tm = {}
    for name, (fn, kw) in methods.items():
        t = time.time(); out = run_method(fn, J, NJ, **kw); tm[name] = time.time() - t; FA[name] = fold_aucs(out, J)
        if not dry:
            P[name] = per_repeat(out)
        LOG(f"   {name}: {tm[name]:.1f} s" + (f"; fold r0k0 AUC {FA[name][0]:.4f}" if dry else f"; pooled AUC {summary(P[name]).auc:.4f}"))
    # ---- null (E1a, E1b): permuted calendar years, full 50 folds
    per_perm = (tm["E1a"] + tm["E1b"]) * (50 if dry else 1)          # dry run: one fold -> scale to 50 folds
    n_perm = 100 if per_perm * 100 / 3600 <= 8 else 20
    LOG(f"   null cost estimate: {per_perm * 100 / 3600:.1f} h for 100 permutations -> using {n_perm}" + (" (DEVIATION: reduced from 100, pre-registered rule)" if n_perm == 20 else ""))
    # ---- E11 / E12 / outputs
    if dry:
        t = time.time(); out = Parallel(n_jobs=NJ)(delayed(e11_job)(r, k, tr, te, 5, s) for s in range(2) for r, k, tr, te in J)
        LOG(f"   E11 dry: {time.time() - t:.1f} s for 2 seeds x 1 fold; fold AUCs {[round(roc_auc_score(Y[J[0][3]], o), 4) for o in out]}")
        D12 = e12(2, 1212); LOG(D12.groupby("method")[["brier", "slope", "auc_rest"]].mean().round(4).to_string())
        LOG(f"DRY-RUN done in {time.time() - t0:.0f} s"); sys.exit(0)
    rng = np.random.default_rng(1111); null = {"E1a": [], "E1b": []}
    for i in range(n_perm):
        yp = rng.permutation(YEAR)
        for name, fn in [("E1a", e1a), ("E1b", e1b)]:
            null[name].append(summary(per_repeat(run_method(fn, JOBS, NJ, years=yp))).auc)
        if (i + 1) % 10 == 0:
            LOG(f"      null {i + 1}/{n_perm}: E1a mean {np.mean(null['E1a']):.4f}, E1b mean {np.mean(null['E1b']):.4f}")
    # E11
    rows11 = []; e11P = {}
    for n in (5, 10, 15, 20, 30):
        seedP = []
        for s in range(20):
            Pp = per_repeat(Parallel(n_jobs=NJ)(delayed(e11_job)(r, k, tr, te, n, s) for r, k, tr, te in JOBS)); seedP.append(Pp)
            rows11.append(dict(n_controls=n, seed=s, auc_within_yxd=r9.strat_auc(Y, Pp.mean(0), r9.STRATA["year x device"]), auc_pooled=summary(Pp).auc))
        e11P[n] = np.mean(seedP, 0)
        LOG(f"   E11 n={n}: within year x device {np.mean([x['auc_within_yxd'] for x in rows11 if x['n_controls'] == n]):.4f}")
    E11 = pd.DataFrame(rows11)
    for n in e11P:
        W = r10.within({"E11": e11P[n], "I2": P["I2"]}, np.arange(N), {"year x device": r9.STRATA["year x device"]}, [("E11", "I2")], 11000 + n)
        E11 = pd.concat([E11, pd.DataFrame([dict(n_controls=n, seed="mean", auc_within_yxd=W.auc_E11.iloc[0], diff_vs_I2=W["diff"].iloc[0], lo=W.lo.iloc[0], hi=W.hi.iloc[0])])])
    rho11 = stats.spearmanr(E11[E11.seed != "mean"].n_controls.astype(float), E11[E11.seed != "mean"].auc_within_yxd.astype(float))
    LOG(f"   E11 trend: Spearman(n, within-stratum AUC) = {rho11[0]:.3f}, p = {rho11[1]:.4f} (100 points)")
    E11.to_csv(f"{R}/round11_E11.csv", index=False)
    # E12
    D12 = e12(50, 1212); D12.to_csv(f"{R}/round11_E12.csv", index=False)
    for ncal in (30, 50):
        d = D12[D12.n_cal == ncal].pivot_table(index="split", columns="method", values=["brier", "slope"])
        for mth in ["intercept", "platt", "intercept_permuted", "platt_permuted"]:
            for met in ["brier", "slope"]:
                diff = (d[(met, mth)] - d[(met, "none")]).values; bs = [np.random.default_rng(i).choice(diff, len(diff)).mean() for i in range(2000)]
                LOG(f"   E12 n_cal={ncal} {mth:18s} {met}: none {d[(met, 'none')].mean():.4f} -> {d[(met, mth)].mean():.4f} (sd {d[(met, mth)].std():.4f}); "
                    f"diff {diff.mean():+.4f} [{np.percentile(bs, 2.5):+.4f}, {np.percentile(bs, 97.5):+.4f}]")
    # ---- statistics and outputs
    order = ["I2", "C1", "E1a", "E1b", "E2", "E6a-I2", "E6a-C1", "E6b-I2", "E6b-C1"]
    rows = []
    for m in order:
        s = summary(P[m]); row = dict(method=m, **s.to_dict())
        for refm in ["I2", "C1"]:
            if m != refm:
                d_, p_ = ek.corrected_ttest(FA[m], FA[refm]); row[f"dAUC_vs_{refm}"], row[f"p_vs_{refm}"] = d_, p_
        rows.append(row)
    T = pd.DataFrame(rows).set_index("method")
    main = ["E1a", "E1b", "E2"]; T["p_holm"] = np.nan; T.loc[main, "p_holm"] = holm(T.loc[main, "p_vs_I2"].values)
    if (T.loc[main, "p_holm"] < 0.05).any():
        sec = ["E1a", "E2"]; T["p_holm_secondary_vs_C1"] = np.nan; T.loc[sec, "p_holm_secondary_vs_C1"] = holm(T.loc[sec, "p_vs_C1"].values)
        LOG("   main family has p_holm < 0.05 -> secondary family vs C1 evaluated")
    else:
        LOG("   main family: no p_holm < 0.05 -> secondary family vs C1 not evaluated (pre-registered rule)")
    for name in ["E1a", "E1b"]:
        nl = np.array(null[name]); obs = T.loc[name, "auc"]
        T.loc[name, "null_mean"], T.loc[name, "null_p95"], T.loc[name, "null_p"] = nl.mean(), np.percentile(nl, 95), (1 + (nl >= obs).sum()) / (1 + len(nl))
    T.to_csv(f"{R}/round11_results.csv")
    LOG("\n== round 11, strict P3, 200 patients, threshold 0.5 (mean over 10 repeats) =="); LOG(T.round(4).to_string())
    np.savez(f"{R}/round11_fold_aucs.npz", **{m: FA[m] for m in order})
    W = r10.within({m: P[m] for m in ["I2", "E1a", "E1b", "E2", "E6a-I2", "E6b-I2"]}, np.arange(N), r9.STRATA,
                   [("E1a", "I2"), ("E1b", "I2"), ("E2", "I2"), ("E6a-I2", "I2"), ("E6b-I2", "I2")], 11111)
    W.to_csv(f"{R}/round11_within.csv", index=False); LOG("\n== within-stratum AUC (round10 estimator, Holm over the table) =="); LOG(W.round(4).to_string(index=False))
    LOG(f"\nFULL RUN done in {(time.time() - t0) / 3600:.2f} h")
