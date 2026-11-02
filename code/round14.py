"""Round 14 (reports/round14_prediction.md, commit bdda0c6): final table with patient-bootstrap CIs and clinical metrics (point 5),
interpretation of I2 (point 6). Descriptive. CPU only."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ["CUDA_VISIBLE_DEVICES"] = ""
import warnings, numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import evalkit as ek, round4 as r4, round7 as r7, round10 as r10, features as fe
warnings.filterwarnings("ignore"); R = "/home/miftah_f/RTX-5090/tuep_fresh/reports"
Y, BP, AS, N, DES = r4.Y, r4.BP, r4.AGESEX, len(r4.Y), r4.DESIGN["I2"]
BANDS = {"rel delta": range(0, 19), "rel theta": range(19, 38), "rel alpha": range(38, 57), "rel beta": range(57, 76), "rel gamma": range(76, 95), "log total": range(95, 114)}
assert fe.NAMES[38].startswith("rel_alpha_") and fe.NAMES[57].startswith("rel_beta_") and fe.NAMES[95].startswith("logtot_")


def to_matrix(J, preds):
    P = np.full((10, N), np.nan)
    for (r, _, _, te), p in zip(J, preds):
        P[r, te] = p
    return P


def rep_metrics(P, idx):
    rows = []
    for p in P:
        q, y = p[idx], Y[idx]; ok = ~np.isnan(q); q, y = q[ok], y[ok]; ci, cs = r4.calib(y, q)
        rows.append(dict(auc=roc_auc_score(y, q), **r4.binm(y, q > 0.5), brier=np.mean((q - y) ** 2), calib_int=ci, calib_slope=cs))
    return pd.DataFrame(rows).mean()


def boot(P, seed, n=2000):
    pats = np.where(~np.isnan(P[0]))[0]; rng = np.random.default_rng(seed); out = []
    for _ in range(n):
        b = rng.choice(pats, len(pats))
        if len(np.unique(Y[b])) == 2:
            out.append(rep_metrics(P, b))
    return pd.DataFrame(out).quantile([0.025, 0.975])


def i2_fold(tr, te, cols=None):
    """I2 exactly as round4, optionally with a subset of band-power columns; returns test probabilities and standardised coefficients."""
    Z = r4.normative_z(DES, tr); Z = Z if cols is None else Z[:, cols]
    m = ek.nested_lr().fit(np.hstack([Z, AS])[tr], Y[tr]); return m.predict_proba(np.hstack([Z, AS])[te])[:, 1], m[-1].coef_[0]


def spec90_job(method, r, k, tr, te):
    oof = np.zeros(len(tr))
    for a, b in StratifiedKFold(5, shuffle=True, random_state=100 * r + k).split(tr, Y[tr]):
        oof[b] = r4.fit_predict(method, tr[a], tr[b], 0)
    thr = np.percentile(oof[Y[tr] == 0], 90); return thr, r4.fit_predict(method, tr, te, 0)


def ppv_npv(se, sp, pi):
    return se * pi / (se * pi + (1 - sp) * (1 - pi)), sp * (1 - pi) / (sp * (1 - pi) + (1 - se) * pi)


if __name__ == "__main__":
    # ---------------- predictions (committed code)
    PR = r7.predictions(); J200, J199 = r7.J200, r7.J199
    nest = Parallel(n_jobs=12)(delayed(r10.nested_job)(r, k, tr, te) for r, k, tr, te in r10.J200)
    M = {"I2": to_matrix(J200, PR["I2"]), "Nested pipeline": to_matrix(J200, [o["p"] for o in nest]), "C1": to_matrix(J200, PR["C1"]),
         "DNM-F": to_matrix(J200, PR["DNM-F"]), "Tang-F (199)": to_matrix(J199, PR["Tang-F@199"]), "Rohira-F": to_matrix(J200, PR["Rohira-F"]),
         "BENDR (pretrained)": to_matrix(J200, PR["BENDR"]), "LaBraM (pretrained)": to_matrix(J200, PR["LaBraM"]), "Metadata floor": to_matrix(J200, PR["Floor"])}
    ref = {"I2": 0.71210, "Nested pipeline": 0.6946, "C1": 0.66627, "DNM-F": 0.6109, "Tang-F (199)": 0.5485, "Rohira-F": 0.5501, "BENDR (pretrained)": 0.5534, "LaBraM (pretrained)": 0.6400, "Metadata floor": 0.6479}
    pts = {m: rep_metrics(P, np.where(~np.isnan(P[0]))[0]) for m, P in M.items()}
    for m in M:
        print(("PASS" if abs(pts[m].auc - ref[m]) < 5e-4 else "FAIL") + f" {m}: AUC {pts[m].auc:.4f} reproduces {ref[m]}")
    # check: our I2 fold function == round4
    p0, _ = i2_fold(J200[0][2], J200[0][3]); print(("PASS" if np.allclose(p0, r4.fit_predict("I2", J200[0][2], J200[0][3], 0)) else "FAIL") + " i2_fold == round4.fit_predict('I2')")
    CI = dict(zip(M, Parallel(n_jobs=9)(delayed(boot)(P, 14000 + i) for i, P in enumerate(M.values()))))
    rows = []
    for m in M:
        row = {"method": m}
        for k_ in ["auc", "acc", "sens", "spec", "prec", "f1", "bac", "brier", "calib_int", "calib_slope"]:
            row[k_], row[k_ + "_lo"], row[k_ + "_hi"] = pts[m][k_], CI[m].loc[0.025, k_], CI[m].loc[0.975, k_]
        rows.append(row)
    T = pd.DataFrame(rows).set_index("method"); T.to_csv(f"{R}/round14_final_table.csv")
    fmt = lambda m, k_, s=100: f"{s * T.loc[m, k_]:.1f} [{s * T.loc[m, k_ + '_lo']:.1f}–{s * T.loc[m, k_ + '_hi']:.1f}]"
    print("\n== final table: strict P3, threshold 0.5, mean over 10 repeats [patient-bootstrap 95% CI] ==")
    for m in M:
        print(f"{m:20s} AUC {fmt(m, 'auc')} | acc {fmt(m, 'acc')} | sens {fmt(m, 'sens')} | spec {fmt(m, 'spec')} | prec {fmt(m, 'prec')} | F1 {fmt(m, 'f1')} | "
              f"BAC {fmt(m, 'bac')} | Brier {fmt(m, 'brier', 1)} | slope {fmt(m, 'calib_slope', 1)}")
    # ---------------- clinical metrics for I2 and C1
    print("\n== clinical operating points (I2, C1) ==")
    clin = []
    for m in ["I2", "C1"]:
        res = Parallel(n_jobs=12)(delayed(spec90_job)(m, r, k, tr, te) for r, k, tr, te in J200)
        se, sp, roc_se = [], [], []
        for r in range(10):
            idx = [(j, jb) for j, jb in enumerate(J200) if jb[0] == r]
            q = np.zeros(N, bool)
            for j, (_, _, _, te) in idx:
                q[te] = res[j][1] > res[j][0]
            se.append(q[Y == 1].mean()); sp.append(1 - q[Y == 0].mean())
            p = M[m][r]; thr_roc = np.percentile(p[Y == 0], 90); roc_se.append((p[Y == 1] > thr_roc).mean())
        se, sp = np.mean(se), np.mean(sp); s5, p5 = T.loc[m, "sens"], T.loc[m, "spec"]
        row = dict(method=m, sens_at_target_spec90=se, spec_achieved=sp, roc_point_sens_at_spec90=np.mean(roc_se))
        for pi in (0.1, 0.3, 0.5):
            row[f"PPV@{int(pi * 100)}%_spec90thr"], row[f"NPV@{int(pi * 100)}%_spec90thr"] = ppv_npv(se, sp, pi)
            row[f"PPV@{int(pi * 100)}%_thr0.5"], row[f"NPV@{int(pi * 100)}%_thr0.5"] = ppv_npv(s5, p5, pi)
        clin.append(row)
        print(f"   {m}: threshold from training-fold OOF (target spec 90%): sens {se:.3f}, spec achieved {sp:.3f}; "
              f"ROC point (not deployable) sens@spec90 {np.mean(roc_se):.3f}")
        print("      " + "; ".join(f"prev {int(pi * 100)}%: PPV {row[f'PPV@{int(pi * 100)}%_spec90thr']:.2f} NPV {row[f'NPV@{int(pi * 100)}%_spec90thr']:.2f} (thr90) | "
                                   f"PPV {row[f'PPV@{int(pi * 100)}%_thr0.5']:.2f} NPV {row[f'NPV@{int(pi * 100)}%_thr0.5']:.2f} (thr 0.5)" for pi in (0.1, 0.3, 0.5)))
    pd.DataFrame(clin).to_csv(f"{R}/round14_clinical.csv", index=False)
    # ---------------- interpretation of I2
    out = Parallel(n_jobs=12)(delayed(i2_fold)(tr, te) for _, _, tr, te in J200)
    C = np.array([o[1] for o in out]); names = list(fe.NAMES) + ["age", "male"]; mean = C.mean(0)
    cons = (np.sign(C) == np.sign(mean)).mean(0)
    Zall = r4.normative_z(DES, np.arange(N)); d = (Zall[Y == 1].mean(0) - Zall[Y == 0].mean(0)) / np.sqrt((Zall[Y == 1].var(0, ddof=1) + Zall[Y == 0].var(0, ddof=1)) / 2)
    Cf = pd.DataFrame(dict(feature=names, coef_mean=mean, coef_sd=C.std(0), sign_consistency=cons, cohens_d=list(d) + [np.nan, np.nan]))
    Cf.to_csv(f"{R}/round14_I2_coefficients.csv", index=False)
    top = Cf.iloc[:114].reindex(Cf.iloc[:114].coef_mean.abs().sort_values(ascending=False).index).head(20)
    print("\n== I2 coefficients (standardised, 50 folds) ==")
    print(f"   age {mean[114]:+.3f} (consistency {cons[114]:.2f}); male {mean[115]:+.3f} ({cons[115]:.2f})")
    print("   by band: " + "; ".join(f"{b}: sum|coef| {np.abs(mean[list(ix)]).sum():.2f}, signed mean {mean[list(ix)].mean():+.3f}" for b, ix in BANDS.items()))
    print(f"   top-20 |coef| features with sign consistency >= 0.9: {(top.sign_consistency >= 0.9).sum()}/20")
    print(top.round(3).to_string(index=False))
    print("   largest univariate effects (Cohen's d of I2 z, epilepsy - non): " +
          ", ".join(f"{r.feature} {r.cohens_d:+.2f}" for r in Cf.iloc[:114].reindex(Cf.iloc[:114].cohens_d.abs().sort_values(ascending=False).index).head(8).itertuples()))
    # drop-one-band
    fa_full = np.array([roc_auc_score(Y[te], o[0]) for (_, _, _, te), o in zip(J200, out)]); print("\n== drop-one-band (descriptive) ==")
    db = []
    for b, ix in BANDS.items():
        cols = [i for i in range(114) if i not in ix]; o2 = Parallel(n_jobs=12)(delayed(i2_fold)(tr, te, cols) for _, _, tr, te in J200)
        Pb = to_matrix(J200, [x[0] for x in o2]); a = rep_metrics(Pb, np.arange(N)).auc; fb = np.array([roc_auc_score(Y[te], x[0]) for (_, _, _, te), x in zip(J200, o2)])
        dd, pp = ek.corrected_ttest(fb, fa_full); db.append(dict(dropped=b, auc=a, dAUC=dd, p=pp)); print(f"   without {b:10s}: AUC {a:.4f} (d {dd:+.4f}, p {pp:.3f})")
    pd.DataFrame(db).to_csv(f"{R}/round14_drop_band.csv", index=False)
    # figure
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    H = mean[:114].reshape(6, 19); S = cons[:114].reshape(6, 19); v = np.abs(H).max()
    fig, ax = plt.subplots(figsize=(11, 3.6)); im = ax.imshow(H, cmap="RdBu_r", vmin=-v, vmax=v, aspect="auto")
    ax.set_xticks(range(19)); ax.set_xticklabels(fe.CH, rotation=90, fontsize=8); ax.set_yticks(range(6)); ax.set_yticklabels(list(BANDS), fontsize=9)
    for i in range(6):
        for j in range(19):
            if S[i, j] >= 0.9:
                ax.text(j, i, "•", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, label="mean standardised coefficient (+ = epilepsy)"); ax.set_title("I2 logistic-regression coefficients, 50 training folds (• sign consistent in ≥90% of folds)", fontsize=9)
    fig.tight_layout(); fig.savefig(f"{R}/round14_I2_coefficients.png", dpi=150); print(f"\nfigure saved: {R}/round14_I2_coefficients.png")
