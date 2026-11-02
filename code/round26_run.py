"""Round 26 (reports/round26_prediction.md, commit 69df653): I2 normative layer with within-person era calibration (E1, I2-WP) and
recording-state matching with photic exclusion (E2, I2-SM). Features: data/round26/features.npz (round26_features.py, label-free).
Everything that uses labels or control status is fitted on the training patients of the fold only; test patients are represented by their
first session only, and their later sessions are never read. CPU only.  Usage: python round26_run.py --dry-run | --run"""
import round17_common as c17
import sys, time, argparse, numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score
import evalkit as ek, round4 as r4

Y, N, AS, SUBJ, YG = c17.Y, c17.N, c17.AS, np.asarray(c17.SUBJ), r4.YEAR_G
with np.load(f"{ek.D}/round26/features.npz", allow_pickle=True) as _Z:          # plain arrays (an open NpzFile cannot go to workers)
    Z = {k: _Z[k] for k in _Z.files}
assert list(Z["SUBJ"]) == list(SUBJ)
BP = r4.BP; assert np.array_equal(Z["BP"], BP)
SES = pd.DataFrame(dict(subject=Z["ses_subject"], era=Z["ses_era"], age=Z["ses_age"], first=Z["ses_first"])); SX, SX_SM = Z["SX"], Z["SX_SM"]
PIDX = {s: i for i, s in enumerate(SUBJ)}; SES["p"] = SES.subject.map(PIDX)
ERAS = ["<=2009", "2010", "2011", "2012", "2013", "2014-15", ">=2016"]; D_I2 = r4.DESIGN["I2"]; D3 = D_I2[:, :3]
COL = {"2010": 3, "2011": 4, "2013": 5, "<=2009": 6}                     # r4.dummies: sorted levels without the reference 2012
WIN_P, WIN_X, WIN_PH = Z["win_patient"], Z["win_X"], Z["win_photic"]
OUT = f"{ek.D}/round26"; MIN_SUPPORT = 3


def nz(X, design, tr):
    """Identical to round4.normative_z, for any patient feature matrix."""
    c = tr[Y[tr] == 0]; B = np.linalg.lstsq(design[c], X[c], rcond=None)[0]
    return (X - design @ B) / (X[c] - design[c] @ B).std(0, ddof=np.linalg.matrix_rank(design[c]))


def lr(F, tr, te):
    return ek.nested_lr().fit(F[tr], Y[tr]).predict_proba(F[te])[:, 1]


def wp_delta(tr, b_age, sx, mode="ctrl", perm_seed=None):
    """Within-person era effects relative to 2012 from the training patients' sessions (subject fixed effects)."""
    pats = tr if mode == "all" else tr[Y[tr] == 0]
    S = SES[SES.p.isin(pats)].copy()
    if perm_seed is not None:                                              # negative control: break the within-person pairing
        later = np.where(~S["first"].values)[0]; S.iloc[later, S.columns.get_loc("subject")] = np.random.default_rng(perm_seed).permutation(S.subject.values[later])
    multi = S.groupby("subject").era.nunique(); S = S[S.subject.isin(multi[multi > 1].index)]
    support = {e: int(S[S.era == e].subject.nunique()) for e in ERAS}
    # eras connected to 2012 through shared subjects (bipartite graph)
    comp, frontier = {"2012"}, ["2012"]
    while frontier:
        e = frontier.pop(); subs = set(S[S.era == e].subject)
        for e2 in set(S[S.subject.isin(subs)].era) - comp:
            comp.add(e2); frontier.append(e2)
    used = [e for e in ERAS if e in comp and support[e] >= MIN_SUPPORT]
    if "2012" not in used:
        return {}, support, []
    y = sx[S.index.values] - np.outer(S.age.values, b_age)
    present = [e for e in ERAS if e != "2012" and (S.era == e).any()]
    E = np.column_stack([(S.era.values == e).astype(float) for e in present])
    g = S.subject.values; yd, Ed = y.copy(), E.copy()
    for s in np.unique(g):
        m = g == s; yd[m] -= yd[m].mean(0); Ed[m] -= Ed[m].mean(0)
    coef = np.linalg.lstsq(Ed, yd, rcond=None)[0]
    delta = {"2012": np.zeros(BP.shape[1])}
    for e in used:
        if e != "2012":
            delta[e] = coef[present.index(e)]
    return delta, support, used


def wp_z(X, tr, sx=SX, mode="ctrl", perm_seed=None, age_zero=False):
    c = tr[Y[tr] == 0]; B = np.linalg.lstsq(D_I2[c], X[c], rcond=None)[0]
    cross = {"2012": np.zeros(X.shape[1]), **{e: B[k] for e, k in COL.items()}}
    delta, support, used = wp_delta(tr, np.zeros(X.shape[1]) if age_zero else B[1], sx, mode, perm_seed)
    eff = np.vstack([delta.get(YG[i], cross[YG[i]]) for i in range(N)]); R = X - eff
    B2 = np.linalg.lstsq(D3[c], R[c], rcond=None)[0]; sd = (R[c] - D3[c] @ B2).std(0, ddof=np.linalg.matrix_rank(D3[c]))
    return (R - D3 @ B2) / sd, used, support


def pred(variant, tr, te, seed=None):
    if variant == "I2":
        F = np.hstack([nz(BP, D_I2, tr), AS])
    elif variant in ("WP", "WP_all", "WP_noage"):
        F = np.hstack([wp_z(BP, tr, mode="all" if variant == "WP_all" else "ctrl", age_zero=variant == "WP_noage")[0], AS])
    elif variant == "WP_perm":
        F = np.hstack([wp_z(BP, tr, perm_seed=seed)[0], AS])
    elif variant == "WP_SM":
        F = np.hstack([wp_z(Z["BP_SM"], tr, sx=SX_SM)[0], AS])
    elif variant in ("SM", "PHOTIC_ONLY", "STATE_ONLY"):
        F = np.hstack([nz(Z[f"BP_{variant}"], D_I2, tr), AS])
    elif variant == "STATE_CONCAT":
        F = np.hstack([nz(Z["BP_ALERT"], D_I2, tr), nz(Z["BP_DROWSY"], D_I2, tr), AS])
    elif variant == "SM_randhalf":
        F = np.hstack([nz(RANDHALF[seed], D_I2, tr), AS])
    return lr(F, tr, te)


def random_half(seed):
    """Negative control for E2: a random half of each patient's non-photic first-session windows (same size as the SM selection)."""
    rng = np.random.default_rng(seed); out = []
    for i in range(N):
        wi = np.where(WIN_P == i)[0]; wn = wi[~WIN_PH[wi]] if (~WIN_PH[wi]).any() else wi
        k = len(wn) if len(wn) < 20 else int(np.ceil(len(wn) / 2)); out.append(np.median(WIN_X[rng.choice(wn, k, replace=False)], 0))
    return np.vstack(out)


RANDHALF = {}


def checks():
    ok = True; tr, te = c17.JOBS[0][2], c17.JOBS[0][3]; c = tr[Y[tr] == 0]
    ok &= c17.pf(np.array_equal(nz(BP, D_I2, tr), r4.normative_z(D_I2, tr)), "nz() == round4.normative_z (I2)")
    z, used, sup = wp_z(BP, tr); print(f"   fold r0k0: eras using within-person delta {used}; support {sup}")
    ok &= c17.pf(np.abs(z[c].mean(0)).max() < 1e-8, f"I2-WP: training controls have mean z 0 ({np.abs(z[c].mean(0)).max():.1e})")
    tsub = set(SUBJ[te]); later_te = SES.index[SES.subject.isin(tsub) & ~SES["first"]].values; first_te = SES.index[SES.subject.isin(tsub) & SES["first"]].values
    sx2 = SX.copy(); sx2[later_te] = sx2[later_te] * 7 + 5
    ok &= c17.pf(np.array_equal(wp_z(BP, tr, sx=sx2)[0], z), f"leakage: changing the test patients' later sessions ({len(later_te)}) changes nothing")
    bp2 = BP.copy(); bp2[te] = bp2[te] * 7 + 5; sx3 = SX.copy(); sx3[first_te] = sx3[first_te] * 7 + 5
    ok &= c17.pf(np.allclose(wp_z(bp2, tr, sx=sx3)[0][tr], z[tr]), "leakage: changing the test patients' first sessions leaves training features unchanged")
    zz, _, _ = wp_z(BP, tr, mode="all"); ok &= c17.pf(np.isfinite(zz).all(), "I2-WP (all patients) finite")
    sm = Z["BP_SM"].copy(); a = nz(sm, D_I2, tr); sm[te] = sm[te] * 7 + 5
    ok &= c17.pf(np.allclose(nz(sm, D_I2, tr)[tr], a[tr]), "leakage: changing test patients' SM features leaves training features unchanged")
    zf, used_f, _ = wp_z(BP, tr, perm_seed=1); print(f"   negative-control fold r0k0: eras using delta after breaking the pairing {used_f}")
    return ok


if __name__ == "__main__":
    a = c17.args(); c17.header("round26_run"); t0 = time.time()
    if not checks():
        sys.exit("STOP")
    J = c17.JOBS; run = lambda v, J_=J, seed=None: c17.to_matrix(J_, Parallel(n_jobs=12)(delayed(pred)(v, tr, te, seed) for _, _, tr, te in J_))
    if a.dry_run:
        P = run("I2"); c17.pf(abs(c17.pooled_auc(P) - 0.71210) < 1e-5, f"I2 reproduced through the round-26 code path ({c17.pooled_auc(P):.5f})")
        r, k, tr, te = J[0]; print(f"dry-run fold r{r}k{k}: WP {roc_auc_score(Y[te], pred('WP', tr, te)):.4f} | SM {roc_auc_score(Y[te], pred('SM', tr, te)):.4f} | {time.time() - t0:.0f} s"); sys.exit()
    use_log = pd.DataFrame([dict(r=r, k=k, used=",".join(wp_z(BP, tr)[1]), used_all=",".join(wp_z(BP, tr, mode="all")[1])) for r, k, tr, _ in J])
    P = {"I2": run("I2")}
    if abs(c17.pooled_auc(P["I2"]) - 0.71210) > 1e-5:
        sys.exit("STOP: I2 not reproduced")
    for v in ["WP", "SM", "WP_all", "WP_noage", "PHOTIC_ONLY", "STATE_ONLY", "STATE_CONCAT", "WP_SM"]:
        P[v] = run(v); print(f"{v:13s} AUC {c17.pooled_auc(P[v]):.4f}", flush=True)
    perm = []
    for i in range(c17.N_SHUF):
        perm.append(run("WP_perm", seed=26100 + i)); print(f"   WP pairing broken {i:2d}: AUC {c17.pooled_auc(perm[-1]):.4f}", flush=True)
    rh = []
    for i in range(c17.N_SHUF):
        RANDHALF[26200 + i] = random_half(26200 + i); rh.append(run("SM_randhalf", seed=26200 + i)); print(f"   SM random half {i:2d}: AUC {c17.pooled_auc(rh[-1]):.4f}", flush=True)
    P169 = {v: run(v, c17.J169) for v in ("I2", "WP", "SM")}
    print(f"169 patients: " + " | ".join(f"{v} {c17.pooled_auc(p):.4f}" for v, p in P169.items()))
    np.savez(f"{OUT}/preds.npz", **P, WP_perm=np.stack(perm), SM_randhalf=np.stack(rh), **{f"{v}_169": p for v, p in P169.items()})
    use_log.to_csv(f"{c17.R}/round26_wp_eras_used.csv", index=False)
    print("eras using within-person delta (controls), per fold:", use_log.used.value_counts().to_dict())
    print(f"done in {(time.time() - t0) / 60:.1f} min")
