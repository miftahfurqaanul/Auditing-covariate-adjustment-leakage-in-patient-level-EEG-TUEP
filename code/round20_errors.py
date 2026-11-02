"""Round 20 part B (reports/round20_prediction.md, commit a03c37f): EXPLORATORY error analysis of F2 (I2 + P3 + NormMamba-F, probability mean).
Frozen definitions: hard = misclassified by F2 (threshold 0.5) in >= 8/10 repeats; easy = correct in >= 8/10; universal-hard = hard and the
repeat-mean scores of I2, P3 and NM are all on the wrong side of 0.5. Comparisons WITHIN each class (hard vs easy). Results are for hypothesis
generation only; nothing here is used to change, select or tune any model on these 200 patients. CPU only."""
import round17_common as c17
import numpy as np, pandas as pd
from scipy import stats
import round4 as r4, round18_fusion as f18, evalkit as ek

J, Y, N, SUBJ = c17.JOBS, c17.Y, c17.N, c17.SUBJ


def metadata():
    S = r4.S.copy(); S["era"] = r4.YEAR_G; S["sex"] = np.where(S.male == 1, "M", "F")
    rec = pd.read_csv(f"{ek.D}/features_bandpower_recordings.csv"); first = rec.session_no == rec.groupby("subject").session_no.transform("min")
    g = rec[first].groupby("subject")[["n_windows", "n_clean"]].sum()
    S["n_clean_first"] = g.n_clean.reindex(S.index).fillna(0).values
    S["reject_frac_first"] = (1 - g.n_clean / g.n_windows.clip(lower=1)).reindex(S.index).values
    idx = pd.read_csv(f"{ek.D}/index.csv"); S["n_sessions_total"] = idx.groupby("subject").session.nunique().reindex(S.index).values
    fi = idx[idx.session_no == idx.groupby("subject").session_no.transform("min")]
    S["ltm_truncated_first"] = fi.groupby("subject").truncated_ltm.any().reindex(S.index).fillna(False).astype(str).values
    mz = np.zeros(N)
    for r, k, tr, te in J:
        mz[te] += np.abs(r4.normative_z(r4.DESIGN["I2"], tr)[te]).mean(1) / 10
    S["mean_abs_z_I2"] = mz; S["y"] = Y
    return S


def cliffs(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float); d = a[:, None] - b[None, :]; return ((d > 0).sum() - (d < 0).sum()) / d.size


def mc_chi2(x, g, n=10000, seed=20300):
    """Chi-square test of independence with a Monte Carlo (permutation) p-value."""
    T = pd.crosstab(x, g).values; stat = stats.chi2_contingency(T, correction=False)[0]; rng = np.random.default_rng(seed); cnt = 0
    for _ in range(n):
        cnt += stats.chi2_contingency(pd.crosstab(x, rng.permutation(g)).values, correction=False)[0] >= stat - 1e-12
    return stat, (cnt + 1) / (n + 1)


CONT = ["age", "dur_min", "n_files", "n_clean_first", "reject_frac_first", "n_sessions_total", "mean_abs_z_I2"]
CAT = ["sex", "era", "device", "montage", "fs", "ltm_truncated_first"]

if __name__ == "__main__":
    c17.header("round20_errors (EXPLORATORY)")
    M = f18.load_members(); F2 = f18.fuse(M, ["I2", "P3", "NM"])
    assert abs(c17.pooled_auc(F2) - 0.72709) < 1e-5; print("PASS F2 reproduces 0.72709")
    wrong = ((F2 > 0.5).astype(int) != Y[None, :]).sum(0)                     # number of repeats misclassified (0..10)
    status = np.where(wrong >= 8, "hard", np.where(wrong <= 2, "easy", "ambiguous"))
    mean = {m: np.nanmean(M[m], 0) for m in ("I2", "P3", "NM")}; mean["F2"] = F2.mean(0)
    wrong_mean = {m: (mean[m] > 0.5).astype(int) != Y for m in mean}
    universal = (status == "hard") & wrong_mean["I2"] & wrong_mean["P3"] & wrong_mean["NM"]
    S = metadata(); S["status"] = status; S["wrong_repeats"] = wrong; S["universal_hard"] = universal
    for m in mean:
        S[f"score_{m}"] = mean[m]
    print("\n1. counts (class x status):"); print(pd.crosstab(np.where(Y == 1, "epilepsy", "non-epilepsy"), status).to_string())
    print(f"   universal-hard: {int(universal.sum())} (epilepsy {int((universal & (Y == 1)).sum())}, non-epilepsy {int((universal & (Y == 0)).sum())})")
    rows = []
    for cls, name in [(1, "epilepsy"), (0, "non-epilepsy")]:
        h, e = (Y == cls) & (status == "hard"), (Y == cls) & (status == "easy")
        for v in CONT:
            a, b = S[v].values[h].astype(float), S[v].values[e].astype(float)
            p = stats.mannwhitneyu(a, b, alternative="two-sided").pvalue if len(a) and len(b) else np.nan
            rows.append(dict(cls=name, variable=v, n_hard=int(h.sum()), n_easy=int(e.sum()), hard=f"{np.median(a):.3g} [{np.percentile(a, 25):.3g}-{np.percentile(a, 75):.3g}]",
                             easy=f"{np.median(b):.3g} [{np.percentile(b, 25):.3g}-{np.percentile(b, 75):.3g}]", effect=f"Cliff's delta {cliffs(a, b):+.2f}", test="Mann-Whitney", p=p))
        sel = h | e
        for v in CAT:
            x, g = S[v].astype(str).values[sel], np.where(h[sel], "hard", "easy"); T = pd.crosstab(x, g)
            if T.shape == (2, 2):
                p = stats.fisher_exact(T.values)[1]; test = "Fisher"
            elif T.shape[0] < 2:
                p, test = np.nan, "constant"
            else:
                _, p = mc_chi2(x, g); test = "chi2 Monte Carlo 10000"
            prop = lambda m_: ", ".join(f"{k}:{c}" for k, c in pd.Series(x[g == m_]).value_counts().sort_index().items())
            rows.append(dict(cls=name, variable=v, n_hard=int(h.sum()), n_easy=int(e.sum()), hard=prop("hard"), easy=prop("easy"), effect="", test=test, p=p))
    Tt = pd.DataFrame(rows); ok = Tt.p.notna(); h_ = np.full(len(Tt), np.nan); h_[ok.values] = f18.holm(Tt.p[ok].values); Tt["p_holm_partB"] = h_
    Tt.to_csv(f"{c17.R}/round20_errors_tests.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 60)
    print("\n2. hard vs easy WITHIN each class (exploratory; p uncorrected, plus Holm over all part-B tests):"); print(Tt.round(4).to_string(index=False))
    print("\n3. error overlap between views (repeat-mean scores, threshold 0.5):")
    for a_ in ("I2", "P3", "NM", "F2"):
        print(f"   {a_}: wrong {int(wrong_mean[a_].sum())}/200 (epilepsy {int((wrong_mean[a_] & (Y == 1)).sum())}, non-epilepsy {int((wrong_mean[a_] & (Y == 0)).sum())})")
    for a_, b_ in [("I2", "P3"), ("I2", "NM"), ("P3", "NM")]:
        both = wrong_mean[a_] & wrong_mean[b_]; either = wrong_mean[a_] | wrong_mean[b_]
        print(f"   {a_} & {b_}: both wrong {int(both.sum())}, either wrong {int(either.sum())}, overlap (both/either) {both.sum() / either.sum():.2f}")
    print(f"   all three wrong: {int((wrong_mean['I2'] & wrong_mean['P3'] & wrong_mean['NM']).sum())}")
    res = wrong_mean["I2"] & ~wrong_mean["F2"]; dam = ~wrong_mean["I2"] & wrong_mean["F2"]
    print(f"   F2 vs I2: rescued (I2 wrong, F2 right) {int(res.sum())} [epi {int((res & (Y == 1)).sum())}, non {int((res & (Y == 0)).sum())}]; "
          f"damaged (I2 right, F2 wrong) {int(dam.sum())} [epi {int((dam & (Y == 1)).sum())}, non {int((dam & (Y == 0)).sum())}]")
    cols = ["y", "status", "wrong_repeats", "score_F2", "score_I2", "score_P3", "score_NM", "age", "sex", "era", "device", "montage", "fs", "dur_min",
            "n_clean_first", "reject_frac_first", "n_sessions_total", "mean_abs_z_I2"]
    H = S[S.universal_hard][cols].sort_values(["y", "score_F2"]); H.to_csv(f"{c17.R}/round20_errors_hard_patients.csv")
    S[cols + ["universal_hard"]].to_csv(f"{c17.R}/round20_errors_patients.csv")
    print("\n4. universal-hard patients:"); print(H.round(3).to_string())
