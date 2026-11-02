"""Round 27 A2-A8 (reports/round27_prediction.md, commit 23f825b): nested estimate of the fusion selection (A2), null-era simulation (A3),
per-fold metric table (A4), interpretability of I2 (A5), sharp-transient sensitivity (A6), between-session stability (A7), hard patients (A8).
No new model variants; everything that uses labels or controls is fitted on the training patients of the fold. CPU only.
Usage: python round27_validity.py --dry-run | --run"""
import round17_common as c17
import os, sys, json, time, numpy as np, pandas as pd, h5py
from multiprocessing import Pool
from joblib import Parallel, delayed
from scipy import stats
from sklearn.metrics import roc_auc_score
import evalkit as ek, round4 as r4, features as fe, round18_fusion as f18

J, Y, N, SUBJ, AS = c17.JOBS, c17.Y, c17.N, np.asarray(c17.SUBJ), c17.AS
BP, D_I2, YG = r4.BP, r4.DESIGN["I2"], r4.YEAR_G
OUT = f"{ek.D}/round27"; os.makedirs(OUT, exist_ok=True); R = c17.R
CANDS = [("I2",), ("P3",), ("NM",), ("I2", "P3"), ("I2", "NM"), ("P3", "NM"), ("I2", "P3", "NM")]
COL = {"2010": 3, "2011": 4, "2013": 5, "<=2009": 6}                     # r4.dummies column of each era (reference 2012)


def nz(X, design, tr):
    c = tr[Y[tr] == 0]; B = np.linalg.lstsq(design[c], X[c], rcond=None)[0]
    return (X - design @ B) / (X[c] - design[c] @ B).std(0, ddof=np.linalg.matrix_rank(design[c]))


def lr_pred(F, tr, te):
    return ek.nested_lr().fit(F[tr], Y[tr]).predict_proba(F[te])[:, 1]


def fold_ci(fa):
    d = np.asarray(fa); se = np.sqrt(d.var(ddof=1) * (1 / len(d) + 40 / 160)); t = stats.t.ppf(0.975, len(d) - 1)
    return d.mean(), d.mean() - t * se, d.mean() + t * se


# ------------------------------------------------------------------ A2 nested fusion selection
def inner_members(f, r, k, tr, te, INNER):
    nm = np.full(N, np.nan); d = json.load(open(f"{ek.D}/round19/nm_inner_r{r}_k{k}.json"))["oof"]; nm[tr] = [d[s] for s in SUBJ[tr]]
    I = {"I2": INNER["I2"][f], "P3": INNER["P3"][f], "NM": nm}
    assert all(np.isnan(v[te]).all() and not np.isnan(v[tr]).any() for v in I.values()), "inner OOF must cover exactly the training patients"
    return I


def a2_nested(M):
    INNER = np.load(f"{ek.D}/round19/cpu_inner.npz"); P = np.full((10, N), np.nan); log = []
    for f, (r, k, tr, te) in enumerate(J):
        I = inner_members(f, r, k, tr, te, INNER)
        sc = [roc_auc_score(Y[tr], np.mean([I[m][tr] for m in c], 0)) for c in CANDS]
        best = max(range(len(CANDS)), key=lambda i: (round(sc[i], 12), -len(CANDS[i]), -i))
        P[r, te] = np.mean([M[m][r, te] for m in CANDS[best]], 0); log.append(dict(r=r, k=k, chosen="+".join(CANDS[best]), **{"+".join(c): s for c, s in zip(CANDS, sc)}))
    return P, pd.DataFrame(log)


# ------------------------------------------------------------------ A3 null-era simulation
CTRL = np.where(Y == 0)[0]; B_TRUE = np.linalg.lstsq(D_I2[CTRL], BP[CTRL], rcond=None)[0]; RES = BP[CTRL] - D_I2[CTRL] @ B_TRUE


def sim_features(seed, era=True):
    rng = np.random.default_rng(seed); B = B_TRUE.copy()
    if not era:
        B[3:] = 0
    return D_I2 @ B + RES[rng.integers(0, len(CTRL), N)]


def sim_fold(X, tr, te):
    z = nz(X, D_I2, tr)
    return (lr_pred(X, tr, te), lr_pred(z, tr, te), lr_pred(np.hstack([X, AS]), tr, te), lr_pred(np.hstack([z, AS]), tr, te))


def sim_rep(seed, era=True):
    X = sim_features(seed, era); out = Parallel(n_jobs=12)(delayed(sim_fold)(X, tr, te) for _, _, tr, te in J)
    return [c17.pooled_auc(c17.to_matrix(J, [o[i] for o in out])) for i in range(4)]


# ------------------------------------------------------------------ A6 transient index (label-free)
def kurt_windows(h5_path):
    with h5py.File(f"{fe.D}/{h5_path}") as f:
        x = f["eeg"][()].astype(np.float32)
    x = x[:, fe.EDGE:x.shape[1] - fe.EDGE]; n = x.shape[1] // fe.WIN
    w = x[:, :n * fe.WIN].reshape(19, n, fe.WIN).transpose(1, 0, 2); w = w[(w.max(2) - w.min(2)).max(1) <= fe.PTP_MAX].astype(np.float64)
    kz = stats.kurtosis(w, axis=2, fisher=True, bias=True)                # flat (zeroed bad) channels give NaN: ignore them
    return h5_path, np.where(np.isnan(kz).all(1), 0.0, np.nanmax(np.where(np.isnan(kz), -np.inf, kz), 1))


def first_session_windows():
    z = np.load(f"{ek.D}/features_bandpower_windows.npz", allow_pickle=True); Xw, win_h5 = z["X"], z["h5_path"]
    rec = pd.read_csv(f"{ek.D}/features_bandpower_recordings.csv"); rec = rec[rec.n_clean > 0].reset_index(drop=True)
    pos = {h: i for i, h in enumerate(rec.h5_path)}; wr = np.array([pos.get(h, -1) for h in win_h5]); Xw, wr = Xw[wr >= 0], wr[wr >= 0]
    first = (rec.groupby("subject").session_no.transform("min") == rec.session_no).values
    fw = {s: np.where(np.isin(wr, np.where(first & (rec.subject.values == s))[0]))[0] for s in SUBJ}
    return Xw, wr, rec, first, fw, pos


def trimmed_bp(Xw, fw, K, q):
    out = []
    for s in SUBJ:
        wi = fw[s]; drop = int(np.floor(q * len(wi))); keep = wi[np.argsort(K[wi], kind="stable")[:len(wi) - drop]]; out.append(np.median(Xw[keep], 0))
    return np.vstack(out)


# ------------------------------------------------------------------ A7 later sessions
def a7_fold(tr, te, ses, SX):
    c = tr[Y[tr] == 0]; B = np.linalg.lstsq(D_I2[c], BP[c], rcond=None)[0]; sd = (BP[c] - D_I2[c] @ B).std(0, ddof=np.linalg.matrix_rank(D_I2[c]))
    F = np.hstack([(BP - D_I2 @ B) / sd, AS]); m = ek.nested_lr().fit(F[tr], Y[tr])
    tsub = set(SUBJ[te]); rows = ses[ses.subject.isin(tsub) & ~ses["first"] & ses.era.isin(["<=2009", "2010", "2011", "2012", "2013"])]
    if rows.empty:
        return {}
    pidx = rows.p.values; Dl = np.zeros((len(rows), D_I2.shape[1])); Dl[:, 0] = 1; Dl[:, 1] = rows.age.values; Dl[:, 2] = AS[pidx, 1]
    for j, e in enumerate(rows.era.values):
        if e in COL:
            Dl[j, COL[e]] = 1
    Fl = np.hstack([(SX[rows.index.values] - Dl @ B) / sd, np.column_stack([rows.age.values, AS[pidx, 1]])])
    pr = m.predict_proba(Fl)[:, 1]; return pd.Series(pr).groupby(pidx).mean().to_dict()


if __name__ == "__main__":
    a = c17.args(); c17.header("round27_validity"); t0 = time.time()
    M = f18.load_members(); M["C1"] = np.load(f"{c17.OUT}/p1.npz")["C1"]; F2 = f18.fuse(M, ["I2", "P3", "NM"])
    ok = [c17.pf(abs(c17.pooled_auc(M["I2"]) - 0.71210) < 1e-5, "I2 reproduces 0.71210"), c17.pf(abs(c17.pooled_auc(F2) - 0.72709) < 1e-5, "F2 reproduces 0.72709")]
    INNER = np.load(f"{ek.D}/round19/cpu_inner.npz")
    ok.append(c17.pf(all(np.isnan(INNER["I2"][f][te]).all() and not np.isnan(INNER["I2"][f][tr]).any() for f, (_, _, tr, te) in enumerate(J)), "inner OOF of I2: NaN exactly at the 40 test patients of every fold"))
    ok.append(c17.pf(D_I2.shape[1] == 7 and np.abs(B_TRUE[3:]).max() > 0 and np.allclose(sim_features(1, era=False) - RES[np.random.default_rng(1).integers(0, len(CTRL), N)], D_I2[:, :3] @ B_TRUE[:3]),
                     "null simulation: era rows of the generating model are non-zero; the no-era version keeps only intercept/age/sex + resampled control residuals"))
    if not all(ok):
        sys.exit("STOP")
    if a.dry_run:
        P, lg = a2_nested(M); print(f"dry-run A2 nested pooled AUC {c17.pooled_auc(P):.4f}; chosen {lg.chosen.value_counts().to_dict()}")
        print("dry-run A3 one replicate [C1-sim-noAS, I2-sim-noAS, C1-sim, I2-sim]:", np.round(sim_rep(27300), 4).tolist(), f"| {time.time() - t0:.0f} s"); sys.exit()
    res = {}
    # A2
    PN, lg = a2_nested(M); res["F2_nested"] = PN; lg.to_csv(f"{R}/round27_A2_nested_choices.csv", index=False)
    FA = {m: c17.fold_aucs(J, P) for m, P in {"F2": F2, "I2": M["I2"], "F2_nested": PN}.items()}
    d1, p1 = ek.corrected_ttest(FA["F2_nested"], FA["I2"]); d2, _ = ek.corrected_ttest(FA["F2"], FA["F2_nested"])
    print(f"\n== A2 nested fusion selection ==\nnested pooled AUC {c17.pooled_auc(PN):.4f} (F2 always {c17.pooled_auc(F2):.4f}); per-fold mean {FA['F2_nested'].mean():.4f} (F2 {FA['F2'].mean():.4f})"
          f"\noptimism (F2 - nested, per fold) {d2:+.4f}; nested - I2 {d1:+.4f} (p {p1:.3f}); chosen: {lg.chosen.value_counts().to_dict()}", flush=True)
    # A3
    sims = np.array([sim_rep(27300 + s) for s in range(100)]); sims0 = np.array([sim_rep(27900 + s, era=False) for s in range(10)])
    np.savez(f"{OUT}/A3_sim.npz", sims=sims, sims_noera=sims0)
    S3 = pd.DataFrame([dict(pipeline=n, mean=sims[:, i].mean(), sd=sims[:, i].std(ddof=1), p5=np.percentile(sims[:, i], 5), p95=np.percentile(sims[:, i], 95), mean_no_era=sims0[:, i].mean())
                       for i, n in enumerate(["C1-sim (no age/sex)", "I2-sim (no age/sex)", "C1-sim (+age/sex)", "I2-sim (+age/sex)"])])
    print("\n== A3 null-era simulation (100 replicates; label kept, no disease signal) =="); print(S3.round(4).to_string(index=False)); S3.to_csv(f"{R}/round27_A3_simulation.csv", index=False)
    # A6 transient index
    Xw, wr, rec, first, fw, pos = first_session_windows(); fr = rec[first]
    with Pool(12) as pool:
        kk = dict(pool.map(kurt_windows, fr.h5_path, chunksize=2))
    K = np.full(len(Xw), np.nan)
    for h in fr.h5_path:
        wi = np.where(wr == pos[h])[0]; assert len(wi) == len(kk[h]); K[wi] = kk[h]
    BP0 = np.vstack([np.median(Xw[fw[s]], 0) for s in SUBJ]); assert np.array_equal(BP0, BP), "first-session features must reproduce BP"
    TR = {q: trimmed_bp(Xw, fw, K, q) for q in (0.1, 0.2, 0.3)}
    for q, X in TR.items():
        res[f"I2_noTrans{int(q * 100)}"] = c17.to_matrix(J, Parallel(n_jobs=12)(delayed(lr_pred)(np.hstack([nz(X, D_I2, tr), AS]), tr, te) for _, _, tr, te in J))
    res["F2_noTrans20"] = f18.fuse({**M, "I2": res["I2_noTrans20"]}, ["I2", "P3", "NM"])
    pk = np.array([np.mean(K[fw[s]]) for s in SUBJ])
    rows = []
    for v in ("I2_noTrans10", "I2_noTrans20", "I2_noTrans30"):
        d, p = ek.corrected_ttest(c17.fold_aucs(J, res[v]), FA["I2"]); rows.append(dict(test=f"{v} - I2", dAUC=d, p=p, pooled=c17.pooled_auc(res[v])))
    d, p = ek.corrected_ttest(c17.fold_aucs(J, res["F2_noTrans20"]), FA["F2"]); rows.append(dict(test="F2_noTrans20 - F2", dAUC=d, p=p, pooled=c17.pooled_auc(res["F2_noTrans20"])))
    S6 = pd.DataFrame(rows); print("\n== A6 sharp-transient sensitivity (primary: I2_noTrans20 - I2) =="); print(S6.round(4).to_string(index=False))
    print(f"patient mean transient index (max-channel excess kurtosis): epilepsy median {np.median(pk[Y == 1]):.3f}, non-epilepsy {np.median(pk[Y == 0]):.3f}, Mann-Whitney p {stats.mannwhitneyu(pk[Y == 1], pk[Y == 0]).pvalue:.4f}")
    S6.to_csv(f"{R}/round27_A6_transients.csv", index=False)
    # A5 interpretability
    Zs = np.zeros_like(BP, dtype=np.float64); cnt = np.zeros(N); coefs = []
    for r, k, tr, te in J:
        z = nz(BP, D_I2, tr); Zs[te] += z[te]; cnt[te] += 1
        mdl = ek.nested_lr().fit(np.hstack([z, AS])[tr], Y[tr]); coefs.append(mdl[-1].coef_[0][:BP.shape[1]])
    Z = Zs / cnt[:, None]; coefs = np.array(coefs); names = list(fe.NAMES)
    def cohend(Zm):
        a_, b_ = Zm[Y == 1], Zm[Y == 0]; sp = np.sqrt(((len(a_) - 1) * a_.var(0, ddof=1) + (len(b_) - 1) * b_.var(0, ddof=1)) / (len(a_) + len(b_) - 2)); return (a_.mean(0) - b_.mean(0)) / sp
    dd = cohend(Z); rng = np.random.default_rng(27500); ep, ne = np.where(Y == 1)[0], np.where(Y == 0)[0]; boots = []
    for _ in range(2000):
        ii = np.r_[rng.choice(ep, len(ep)), rng.choice(ne, len(ne))]; Zb = Z[ii]; yb = Y[ii]
        a_, b_ = Zb[yb == 1], Zb[yb == 0]; sp = np.sqrt(((len(a_) - 1) * a_.var(0, ddof=1) + (len(b_) - 1) * b_.var(0, ddof=1)) / (len(a_) + len(b_) - 2)); boots.append((a_.mean(0) - b_.mean(0)) / sp)
    boots = np.array(boots); lo, hi = np.percentile(boots, 2.5, 0), np.percentile(boots, 97.5, 0)
    E5 = pd.DataFrame(dict(feature=names, cohen_d=dd, d_lo=lo, d_hi=hi, coef_mean=coefs.mean(0), coef_sd=coefs.std(0), sign_consistency=np.maximum((coefs > 0).mean(0), (coefs < 0).mean(0))))
    E5.to_csv(f"{R}/round27_A5_effects.csv", index=False)
    print("\n== A5 out-of-fold normative z: largest effects (Cohen's d, epilepsy - non-epilepsy) ==")
    print(E5.reindex(E5.cohen_d.abs().sort_values(ascending=False).index).head(12).round(3).to_string(index=False))
    bands = ["delta", "theta", "alpha", "beta", "gamma", "logtot"]; dm = dd.reshape(6, 19)
    print("per band: mean d over 19 channels [min, max]; channels with CI excluding 0:")
    for b, row, l_, h_ in zip(bands, dm, lo.reshape(6, 19), hi.reshape(6, 19)):
        print(f"   {b:7s} {row.mean():+.3f} [{row.min():+.3f}, {row.max():+.3f}]  {int(((l_ > 0) | (h_ < 0)).sum())}/19")
    print(f"LR coefficients: features with the same sign in >= 90% of folds: {(E5.sign_consistency >= 0.9).sum()}/114")
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 3.6)); v = np.abs(dm).max(); im = ax.imshow(dm, cmap="RdBu_r", vmin=-v, vmax=v, aspect="auto")
    ax.set_yticks(range(6)); ax.set_yticklabels(["delta rel", "theta rel", "alpha rel", "beta rel", "gamma rel", "log total"]); ax.set_xticks(range(19)); ax.set_xticklabels(fe.CH, rotation=90)
    for (i, j), val in np.ndenumerate(dm):
        if lo.reshape(6, 19)[i, j] > 0 or hi.reshape(6, 19)[i, j] < 0:
            ax.text(j, i, "*", ha="center", va="center", fontsize=9)
    plt.colorbar(im, ax=ax, label="Cohen's d (epilepsy - non-epilepsy)"); ax.set_title("Out-of-fold control-referenced deviations (I2); * = 95% bootstrap CI excludes 0")
    plt.tight_layout(); plt.savefig(f"{R}/round27_A5_effect_map.png", dpi=150); plt.close()
    # A7 later sessions
    Z26 = np.load(f"{ek.D}/round26/features.npz", allow_pickle=True); ses = pd.DataFrame(dict(subject=Z26["ses_subject"], session=Z26["ses_session"], era=Z26["ses_era"], age=Z26["ses_age"], first=Z26["ses_first"]))
    ses["p"] = ses.subject.map({s: i for i, s in enumerate(SUBJ)}); SX = Z26["SX"]
    LP = np.full((10, N), np.nan)
    for (r, k, tr, te), dct in zip(J, Parallel(n_jobs=12)(delayed(a7_fold)(tr, te, ses, SX) for _, _, tr, te in J)):
        for i, v in dct.items():
            LP[r, i] = v
    elig = ~np.isnan(LP[0]); auc_l = np.mean([roc_auc_score(Y[elig], LP[r, elig]) for r in range(10)]); auc_f = np.mean([roc_auc_score(Y[elig], M["I2"][r, elig]) for r in range(10)])
    rho = stats.spearmanr(np.nanmean(LP, 0)[elig], M["I2"].mean(0)[elig]).correlation
    print(f"\n== A7a I2 applied to later sessions of test patients (eras <=2013) ==\npatients with eligible later sessions: {elig.sum()} (epilepsy {int(Y[elig].sum())}, non-epilepsy {int((1 - Y[elig]).sum())})"
          f"\nAUC later sessions {auc_l:.4f} vs first session {auc_f:.4f} on the same patients; Spearman(first, later) {rho:.3f}")
    rows = []
    for lab, nm_ in ((0, "non-epilepsy"), (1, "epilepsy")):
        diffs = []
        for s, g in ses[Y[ses.p.values] == lab].sort_values("session").groupby("subject"):
            f0 = g[g["first"]]
            if f0.empty:
                continue
            nxt = g[(~g["first"]) & (g.era == f0.era.iloc[0])]
            if len(nxt):
                diffs.append(SX[nxt.index[0]] - SX[f0.index[0]])
        diffs = np.array(diffs)
        if len(diffs) >= 3:
            t, p = stats.ttest_1samp(diffs, 0); q = stats.false_discovery_control(p)
            rows.append(dict(group=nm_, n_patients=len(diffs), features_shifted_q05=int((q < 0.05).sum()), median_abs_shift_sd=float(np.median(np.abs(diffs.mean(0)) / BP[Y == lab].std(0)))))
    S7 = pd.DataFrame(rows); print("== A7b within-person feature shift, first vs next session in the same era =="); print(S7.round(4).to_string(index=False))
    S7.to_csv(f"{R}/round27_A7_shift.csv", index=False); res["I2_later_sessions"] = LP
    # A8 hard patients
    H = pd.read_csv(f"{R}/round20_errors_hard_patients.csv"); hard = set(H[H.status == "hard"].subject); ish = np.array([s in hard for s in SUBJ])
    drowsy = np.array([(Z26["win_aidx"][Z26["win_patient"] == i] < 0).mean() for i in range(N)]); phot = np.array([Z26["win_photic"][Z26["win_patient"] == i].mean() for i in range(N)])
    nwin = np.array([len(fw[s]) for s in SUBJ]); beta_z = Z[:, [names.index(f"rel_beta_{c}") for c in fe.CH]].mean(1)
    vars_ = dict(age=AS[:, 0], male=AS[:, 1], era_le2009=(YG == "<=2009").astype(float), n_windows=nwin, drowsy_frac=drowsy, photic_frac=phot, transient_index=pk,
                 mean_abs_z=np.abs(Z).mean(1), mean_beta_z=beta_z)
    rows = []
    for lab, nm_ in ((1, "epilepsy"), (0, "non-epilepsy")):
        a_, b_ = ish & (Y == lab), (~ish) & (Y == lab)
        for v, x in vars_.items():
            if v in ("male", "era_le2009"):
                tab = [[int(x[a_].sum()), int(a_.sum() - x[a_].sum())], [int(x[b_].sum()), int(b_.sum() - x[b_].sum())]]; p = stats.fisher_exact(tab).pvalue
            else:
                p = stats.mannwhitneyu(x[a_], x[b_]).pvalue
            rows.append(dict(cls=nm_, variable=v, n_hard=int(a_.sum()), hard=float(np.median(x[a_]) if v not in ("male", "era_le2009") else x[a_].mean()), others=float(np.median(x[b_]) if v not in ("male", "era_le2009") else x[b_].mean()), p=p))
    S8 = pd.DataFrame(rows); S8["q_bh"] = stats.false_discovery_control(S8.p.values)
    print(f"\n== A8 hard patients ({ish.sum()}) vs others in the same class (exploratory; medians or proportions) =="); print(S8.round(4).to_string(index=False)); S8.to_csv(f"{R}/round27_A8_hard.csv", index=False)
    # A4 per-fold metric table
    A21 = np.load(f"{ek.D}/round21/age_preds.npz"); LB = np.full((10, N), np.nan); CB = np.full((10, N), np.nan)
    for r, k, tr, te in J:
        for P_, m_ in ((LB, "labram"), (CB, "cbramod")):
            d = json.load(open(f"{ek.D}/round25/{m_}/r{r}_k{k}.json"))["test"]; P_[r, te] = [d[s] for s in SUBJ[te]]
    TAB = {"C1": M["C1"], "I2": M["I2"], "P3": M["P3"], "NM": M["NM"], "F2": F2, "F2 nested (A2)": PN, "LaBraM frozen": np.load(f"{c17.OUT}/p5.npz")["LaBraM_raw"],
           "LaBraM-FT": LB, "CBraMod-FT": CB, "Metadata floor": A21["Metadata floor"], "Demographic floor": A21["Demo"]}
    rows = [dict(model=m, pooled_auc=c17.pooled_auc(P), **dict(zip(["fold_mean", "ci_lo", "ci_hi"], fold_ci(c17.fold_aucs(J, P))))) for m, P in TAB.items()]
    S4 = pd.DataFrame(rows); print("\n== A4 per-fold mean AUC with Nadeau-Bengio-corrected 95% CI (primary metric) and pooled AUC =="); print(S4.round(4).to_string(index=False))
    S4.to_csv(f"{R}/round27_A4_metrics.csv", index=False)
    np.savez(f"{OUT}/preds.npz", **res, K=K, Z_oof=Z, coefs=coefs)
    print(f"\ndone in {(time.time() - t0) / 60:.1f} min")
