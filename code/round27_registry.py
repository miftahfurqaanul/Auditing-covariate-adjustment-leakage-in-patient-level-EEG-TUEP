"""Round 27 A1 (reports/round27_prediction.md, commit 23f825b): registry of every pre-registered PRIMARY test of rounds 4-26 plus the
reproduction p-value audit; per-family Holm (as reported) and project-wide Holm, Bonferroni and Benjamini-Hochberg FDR. CPU only.
Rounds 4-6 are taken from the result sentences of their prediction files (no CSV); everything else from the result CSVs."""
import numpy as np, pandas as pd

R = "/home/miftah_f/RTX-5090/tuep_fresh/reports"; rows = []


def add(rnd, fam, test, p, ph, src):
    rows.append(dict(round=rnd, family=fam, test=test, p=float(p), p_holm_family=float(ph), source=src))


# rounds 4-6: values quoted in the result sections (round4_prediction.md l.62, round5_prediction.md l.58, round6_atm_prediction.md l.73-74)
for t, p, h in [("I1 - C1", 0.758, 1.000), ("I2 - C1", 0.093, 0.279), ("I2b - C1", 0.560, 1.000)]:
    add(4, "r4 (Holm 3)", t, p, h, "round4_prediction.md")
for t, p, h in [("H5a - C1", 0.766, 1.000), ("H5a - I2", 0.162, 0.486), ("H5b - I2", 1.000, 1.000)]:
    add(5, "r5 (Holm 3)", t, p, h, "round5_prediction.md")
for t, p, h in [("DNM-M - DNM-F", 0.279, 0.559), ("DNM-M - C1", 0.093, 0.280), ("DNM-M - I2", 0.011, 0.042), ("DNM-F - C1", 0.330, 0.559)]:
    add(6, "r6 (Holm 4)", t, p, h, "round6_atm_prediction.md")
d = pd.read_csv(f"{R}/pvalue_audit.csv")
for _, r in d.iterrows():
    add(3, "audit reproductions vs C1 (Holm 4)", r.test, r.recomputed_p, r.holm_p, "pvalue_audit.csv")
d = pd.read_csv(f"{R}/round7_tests.csv")
for _, r in d.iterrows():
    add(7, "r7 (Holm 6)", r.test, r.p, r.p_holm, "round7_tests.csv")
d = pd.read_csv(f"{R}/round10_within_200.csv"); d = d[d.p_holm.notna()]
for _, r in d.iterrows():
    add(10, "r10 within-stratum (Holm 6; supersedes r9)", f"{r.test} [{r.strata}]", r.p, r.p_holm, "round10_within_200.csv")
d = pd.read_csv(f"{R}/round11_results.csv"); d = d[d.p_holm.notna()]
for _, r in d.iterrows():
    add(11, "r11 (Holm 3)", f"{r.method} - I2", r.p_vs_I2, r.p_holm, "round11_results.csv")
d = pd.read_csv(f"{R}/round12_kural_results.csv"); d = d[d.p_holm.notna()]
for _, r in d.iterrows():
    add(12, "r12 Kural (Holm 3)", f"{r.method} vs 0.5", r.p if pd.notna(r.p) else 1.0, r.p_holm, "round12_kural_results.csv")
d = pd.read_csv(f"{R}/round12b_kural_I2_results.csv").iloc[0]
add(12.5, "r12b Kural I2 (Holm 2)", "I2 transfer vs 0.5", d.p_perm, d.p_holm, "round12b_kural_I2_results.csv")
add(12.5, "r12b Kural I2 (Holm 2)", "I2 transfer - C1 transfer", d.p_vs_C1, d.p_holm_vs_C1, "round12b_kural_I2_results.csv")
d = pd.read_csv(f"{R}/round13_within_montage.csv"); d = d[d.p_holm.notna()]
for _, r in d.iterrows():
    add(13, "r13 within montage (Holm 4)", f"{r.test} [{r.strata}]", r.p, r.p_holm, "round13_within_montage.csv")
d = pd.read_csv(f"{R}/round15_baselines.csv"); d = d[d.p_holm.notna()]
for _, r in d.iterrows():
    add(15, "r15 baselines (Holm 3)", f"I2 - {r.method}", r.p, r.p_holm, "round15_baselines.csv")
for f, rnd, fam in [("round16_tests.csv", 16, "r16 (Holm 3)"), ("round21_age_tests.csv", 21, "r21 age inputs (Holm 3)"), ("round24_primary.csv", 24, "r24 (Holm 3)"),
                    ("round25_primary.csv", 25, "r25 (Holm 2)"), ("round26_primary.csv", 26, "r26 (Holm 2)"), ("round22_combined_tests.csv", 22, "r22 C (Holm 2)")]:
    d = pd.read_csv(f"{R}/{f}"); d = d[d.p_holm.notna()]
    for _, r in d.iterrows():
        add(rnd, fam, r.test, r.p, r.p_holm, f)
for f, rnd, n in [("round17_results.csv", 17, 5), ("round18_results.csv", 18, 3), ("round19_results.csv", 19, 2)]:
    d = pd.read_csv(f"{R}/{f}"); d = d[d.p_holm.notna()]
    for _, r in d.iterrows():
        add(rnd, f"r{rnd} (Holm {n})", r.method, r.p, r.p_holm, f)
d = pd.read_csv(f"{R}/round20_harmonisation_tests.csv"); d = d[d.family.str.startswith("primary")]
for _, r in d.iterrows():
    add(20, "r20 (Holm 4)", r.test, r.p, r.p_holm, "round20_harmonisation_tests.csv")
d = pd.read_csv(f"{R}/round21_age_within.csv"); d = d[d.strata == "age3"]
for _, r in d.iterrows():
    add(21, "r21 within age (Holm 12 incl. secondary age x era rows)", f"{r.test} [age3]", r.p, r.p_holm, "round21_age_within.csv")
d = pd.read_csv(f"{R}/round22_AB_tests.csv")
for _, r in d.iterrows():
    add(22, f"r22 {r.part} (Holm {2 if r.part == 'A' else 4})", r.test, r.p, r.p_holm, "round22_AB_tests.csv")
d = pd.read_csv(f"{R}/round23_calibration.csv")
for _, r in d.iterrows():
    add(23, "r23 calibration (Holm 4)", f"{r.measure} [{r['sample']}]", r.p, r.p_holm, "round23_calibration.csv")
d = pd.read_csv(f"{R}/round23_within_combined.csv"); d = d[d.p_holm_combined.notna()]          # primary = combined effect only (per-view rows are descriptive)
for _, r in d.iterrows():
    add(23, "r23 within combined (Holm 3)", f"{r.effect} [{r.strata}]", r.p, r.p_holm_combined, "round23_within_combined.csv")

T = pd.DataFrame(rows); m = len(T); o = np.argsort(T.p.values, kind="stable"); p = T.p.values
hg = np.empty(m); run = 0
for j, i in enumerate(o):
    run = max(run, min(1.0, p[i] * (m - j))); hg[i] = run
bh = np.empty(m); prev = 1.0
for j in range(m - 1, -1, -1):
    i = o[j]; prev = min(prev, p[i] * m / (j + 1)); bh[i] = min(prev, 1.0)
T["p_holm_global"], T["p_bonf_global"], T["q_bh_global"] = hg, np.minimum(1, p * m), bh
T = T.sort_values("p").reset_index(drop=True)
T.to_csv(f"{R}/round27_registry.csv", index=False)
print(f"registry: {m} primary tests in {T.family.nunique()} pre-registered families (rounds 3-26)")
print(f"significant within their own family (Holm < 0.05): {(T.p_holm_family < 0.05).sum()}; project-wide Holm < 0.05: {(T.p_holm_global < 0.05).sum()}; "
      f"BH q < 0.05: {(T.q_bh_global < 0.05).sum()}; BH q < 0.10: {(T.q_bh_global < 0.10).sum()}")
print(T.head(16)[["round", "test", "p", "p_holm_family", "p_holm_global", "q_bh_global"]].round(4).to_string(index=False))
