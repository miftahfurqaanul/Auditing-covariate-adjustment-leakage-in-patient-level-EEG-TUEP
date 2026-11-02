"""Round 12b (reports/round12b_kural_I2_prediction.md, commit 6c1fcdd): generalisation of I2 to Kural et al. 2021. CPU only."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ["CUDA_VISIBLE_DEVICES"] = ""
import warnings, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import evalkit as ek, round4 as r4, round12_kural as k12
warnings.filterwarnings("ignore"); R = k12.R
BP, AS, Y, DES = r4.BP, r4.AGESEX, r4.Y, r4.DESIGN["I2"]


def i2_fit(tr):
    """I2 exactly as round4: OLS on training controls with the I2 design, z for all, + age/sex -> nested_lr."""
    c = tr[Y[tr] == 0]; B = np.linalg.lstsq(DES[c], BP[c], rcond=None)[0]; sd = (BP[c] - DES[c] @ B).std(0, ddof=np.linalg.matrix_rank(DES[c]))
    Z = (BP - DES @ B) / sd; m = ek.nested_lr().fit(np.hstack([Z, AS])[tr], Y[tr]); return m, B, sd, Z


if __name__ == "__main__":
    # ---- check: our I2 == round4.fit_predict("I2") on a canonical TUEP fold
    F = ek.folds(); pos = {s: i for i, s in enumerate(r4.SUBJ)}; te0 = np.array([pos[s] for s in F[(F.repeat == 0) & (F.fold == 0)].subject])
    tr0 = np.setdiff1d(np.arange(len(Y)), te0); m0, B0, sd0, Z0 = i2_fit(tr0)
    print(("PASS" if np.allclose(m0.predict_proba(np.hstack([Z0, AS])[te0])[:, 1], r4.fit_predict("I2", tr0, te0, 0)) else "FAIL") + " i2_fit == round4.fit_predict('I2') on fold r0k0")
    # ---- I2 classifier on all TUEP
    allp = np.arange(len(Y)); m, B, sd, Z = i2_fit(allp); c = np.where(Y == 0)[0]
    print(("PASS" if np.abs(Z[c].mean(0)).max() < 1e-8 else "FAIL") + f" mean z of TUEP controls = 0 (max {np.abs(Z[c].mean(0)).max():.1e}); chosen C {m[-1].C_[0]:.4g}")
    # ---- Kural (same subjects as round 12)
    L = k12.kural_labels(); Fk = k12.kural_features(); D = L.join(Fk); D = D[D.ok]; yk = D.y.values; Xk = np.vstack(D.feat.values); ASk = D[["age", "male"]].values
    print(f"Kural subjects used {len(yk)} ({yk.sum()} / {(1 - yk).sum()}); excluded {sorted(set(L.index) - set(D.index))}")
    DK = np.column_stack([np.ones(len(yk)), ASk[:, 0], ASk[:, 1], np.zeros((len(yk), DES.shape[1] - 3))])     # new batch -> year dummies 0 (= reference group 2012, the most frequent; comment corrected 2026-10-12, was '2013')
    p_i2 = m.predict_proba(np.hstack([(Xk - DK @ B) / sd, ASk]))[:, 1]
    m_c1 = ek.nested_lr().fit(np.hstack([BP, AS]), Y); p_c1 = m_c1.predict_proba(np.hstack([Xk, ASk]))[:, 1]
    a_c1 = roc_auc_score(yk, p_c1); print(("PASS" if abs(a_c1 - 0.6222) < 5e-5 else "FAIL") + f" C1 transfer AUC {a_c1:.4f} == round 12 (0.6222)")
    # ---- test 1: I2 transfer vs 0.5
    a_i2, ci_i2, p_perm = k12.auc_ci_perm(yk, p_i2, 12101)
    # ---- test 2: paired bootstrap I2 - C1 on the same subjects
    rng = np.random.default_rng(12102); d = []
    for _ in range(2000):
        i = rng.integers(0, len(yk), len(yk))
        if len(np.unique(yk[i])) == 2:
            d.append(roc_auc_score(yk[i], p_i2[i]) - roc_auc_score(yk[i], p_c1[i]))
    d = np.array(d); p_pair = min(1, 2 * min((d <= 0).mean(), (d >= 0).mean())); p_pair = max(p_pair, 1 / 2000)
    h = k12.holm([p_perm, p_pair])
    print("\n== I2 trained on TUEP, applied to Kural (Kural as a new batch) ==")
    for name, p in [("I2 transfer", p_i2), ("C1 transfer", p_c1)]:
        bm = r4.binm(yk, p > 0.5); print(f"   {name}: AUC {roc_auc_score(yk, p):.4f}; acc {bm['acc']:.3f} sens {bm['sens']:.3f} spec {bm['spec']:.3f} prec {bm['prec']:.3f} "
                                         f"f1 {bm['f1']:.3f} bac {bm['bac']:.3f}; brier {np.mean((p - yk) ** 2):.3f}; mean p {p.mean():.2f}")
    print(f"   test 1 I2 transfer vs 0.5: AUC {a_i2:.4f} [{ci_i2[0]:.3f}-{ci_i2[1]:.3f}], permutation p {p_perm:.4f}, Holm {h[0]:.4f}")
    print(f"   test 2 I2 - C1 transfer: dAUC {a_i2 - a_c1:+.4f}, paired bootstrap 95% CI [{np.percentile(d, 2.5):+.4f}, {np.percentile(d, 97.5):+.4f}], p {p_pair:.4f}, Holm {h[1]:.4f}")
    # ---- I2-local (offset from Kural training controls), same folds as round 12
    P = {k: np.zeros((10, len(yk))) for k in ["I2 transfer", "I2-local", "C1 transfer"]}; fa = {k: [] for k in P}
    for r in range(10):
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=2000 + r).split(Xk, yk):
            ck = tr[yk[tr] == 0]; off = (Xk[ck] - DK[ck] @ B).mean(0)
            pr = {"I2 transfer": p_i2[te], "I2-local": m.predict_proba(np.hstack([(Xk[te] - DK[te] @ B - off) / sd, ASk[te]]))[:, 1], "C1 transfer": p_c1[te]}
            for k_, v in pr.items():
                P[k_][r, te] = v; fa[k_].append(roc_auc_score(yk[te], v))
            assert abs(fa["I2-local"][-1] - fa["I2 transfer"][-1]) < 1e-12
    print("\n== repeated CV within Kural (10 x 5, same folds as round 12), mean over repeats ==")
    rows = []
    for k_ in P:
        per = pd.DataFrame([dict(auc=roc_auc_score(yk, p), **r4.binm(yk, p > 0.5), brier=np.mean((p - yk) ** 2)) for p in P[k_]]).mean()
        rows.append(dict(method=k_, **per.to_dict())); print(f"   {k_:12s} AUC {per.auc:.4f} acc {per.acc:.3f} sens {per.sens:.3f} spec {per.spec:.3f} prec {per.prec:.3f} f1 {per.f1:.3f} bac {per.bac:.3f} brier {per.brier:.3f}")
    nk = len(yk); dd, pp = ek.corrected_ttest(fa["I2-local"], fa["C1 transfer"], int(0.8 * nk), nk - int(0.8 * nk))
    print(f"   descriptive: fold-level I2-local - C1 transfer dAUC {dd:+.4f}, corrected p ({int(0.8 * nk)}/{nk - int(0.8 * nk)}) {pp:.4f}")
    T = pd.DataFrame(rows).set_index("method")
    T.loc["I2 transfer", ["auc_all", "lo", "hi", "p_perm", "p_holm"]] = [a_i2, ci_i2[0], ci_i2[1], p_perm, h[0]]
    T.loc["I2 transfer", ["dAUC_vs_C1", "d_lo", "d_hi", "p_vs_C1", "p_holm_vs_C1"]] = [a_i2 - a_c1, np.percentile(d, 2.5), np.percentile(d, 97.5), p_pair, h[1]]
    T.to_csv(f"{R}/round12b_kural_I2_results.csv"); print("\n" + T.round(4).to_string())
