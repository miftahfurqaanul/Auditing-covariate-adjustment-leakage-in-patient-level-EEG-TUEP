# Analysis plan for the additional analyses (round 28), written BEFORE code and results

Date: 2026-10-21. Status: exploratory and complementary; NOT part of the registry of 89 primary tests, does not change the Holm or BH correction of the registry, and is not used to choose models. All results are reported, including those that weaken claims. The source data are read read-only from `/home/miftah_f/RTX-5090/tuep_fresh` (canonical out-of-fold predictions, 50 folds of 10 x 5). No GPU. No git commit (only the SHA-256 of this document and of the code, recorded before running in `PRAREGISTRASI_hash.txt`).
Reason: the questions a reviewer is most likely to ask (correlation between views, decision curves, strata sizes, partial and full confounder bias, Hamdan-style confound leakage).

## 0. Shared definitions

- Model score \(\hat y_m\): the mean over 10 repetitions of the out-of-fold probability per patient (200 values). Models: I2, P3, NM, F2 (= mean of I2, P3, NM), C1 (without the normative layer), the acquisition-metadata baseline ("Floor"; recomputed with `round9.lr_acq` on the canonical folds and must reproduce the pooled AUC 0.6479), I2-noAS and F2-noAS-partial (without age and sex; `round21/age_preds.npz`), LaBraM-FT and CBraMod-FT (`round25`). The reproduction must pass: I2 0.71210, F2 0.72709, C1 0.66627 (pooled AUC), otherwise stop.
- Label Y: 1 epilepsy (100), 0 non-epilepsy (100). Era: `round4.YEAR_G` (<=2009, 2010, 2011, 2012, 2013). Device: `round4.DEV_G` (X, Intution/Ultrasom, other; THREE groups, different from the four groups in Table 2). Montage: AR (including AR(A)) or LE (`round17_common.MONT`). Age: `round4.AGE` (continuous).
- All permutations and bootstraps use the fixed seeds written below. The permutation p = (1 + number of random statistics >= the original statistic) / (B + 1), B = 10,000.

## 1. Correlation between views (descriptive)

Spearman between the scores of I2, P3, NM, and Floor: (a) all 200 patients (score averaged over repetitions; 95% percentile bootstrap CI from 2,000 patient resamples, seed 28900); (b) within the epilepsy class and within the non-epilepsy class; (c) the mean of 10 correlations per repetition. Additional descriptive: Spearman of each view with era (ordinal), age, and the Floor score. No decision rule.
Prediction: the correlation between the three views in all patients 0.4--0.8; the within-class correlation lower than in all patients; the correlation of each view with Floor below the correlation between views.

## 2. Decision curves (net benefit; descriptive)

Models: F2, I2, Floor; comparators: treat-all and treat-none. Thresholds pt = 0.10, 0.15, ..., 0.90; treat if score >= pt. NB = (TP - FP * pt/(1-pt)) / 200, computed per repetition and then averaged; treat-all: 0.5 - 0.5 * pt/(1-pt). 95% CI: bootstrap of 2,000 patient resamples (seed 28901) on the NB averaged over repetitions, also for the differences F2 - Floor, F2 - I2, and F2 - treat-all. Prevalence sensitivity (descriptive; assumes the scores are calibrated at prevalence 0.5, so the score threshold = odds pt/(1-pt) * (1-pi)/pi): pi = 0.10, 0.20, 0.30, with NB = sens * pi - (1 - spec) * (1 - pi) * pt/(1-pt).
Reporting rule: state the threshold range where the NB of F2 > max(treat-all, 0) and where the CI of the difference against Floor and I2 does not include 0. No claim of clinical benefit.
Prediction: at prevalence 0.5, F2 exceeds treat-all for pt >= 0.50 and exceeds treat-none for pt up to about 0.6; the CI of F2 - Floor and F2 - I2 includes 0 at most thresholds.

## 3. Number of patients per stratum (descriptive)

For the strata used in Results (year, device [3 groups], year x device, year x montage) and three age groups (<=40, 41--60, >60): per stratum the number of epilepsy, non-epilepsy, epilepsy-non-epilepsy pairs within the stratum (n_e x n_n), and their percentage of the 10,000 pairs; the number of strata that contain both classes; the number of patients in strata that contain both classes; the smallest cell.

## 4. Spisak-style confounder test (on the out-of-fold scores)

Own implementation (conditional permutation test with a binary label, so permutation within class is exact) and the comparison package `mlconfound` 0.21.3 (pygam replaced by a stub; only the categorical/linear branch used) on a subset of tests, B = 2,000.
- Partial test (H0: the score is independent of the confounder c given the label): statistic R^2(score, c) (eta-squared if c is categorical; r^2 for age); c is permuted within each label class. Seed 28000 + test index.
- Full test (H0: the score is independent of the label given c): statistic R^2(score, label); the label is permuted within each cell of c. c = era, device (3 groups), year x device. Seed 28100 + test index.
- Primary family (Holm correction within the family): partial = {I2, F2} x {era, device, montage, age} (8 tests); full = {I2, F2} x {era, device, year x device} (6 tests). Other models and the per-repetition sensitivity (median p over 10 repetitions for I2 and F2 against era) are descriptive, raw p.
Interpretation rule (fixed before running): partial p_Holm < 0.05 means the score still carries confounder variance beyond what the label explains (partial confounder bias detected); partial p_Holm >= 0.05 does not prove the absence of bias (limited power). Full p_Holm < 0.05 means the score carries label information beyond the confounder.
Prediction: Floor partial for era p < 0.001 (positive control); C1 partial for era p < 0.05; I2 and F2 partial for era uncertain (my probability 50% p_Holm < 0.05); full test for I2 and F2 on era and year x device p < 0.05.

## 5. Hamdan-style confound leakage test (normative layer)

Following the two lines of evidence of Hamdan et al. (features permuted before confound removal; a variant that keeps the joint distribution of confound and features). Features = the 114 first-session spectral features (`round4.BP`). Pipeline and folds identical to I2 (`round27_validity.sim_fold`: C1 = raw features, I2 = normative z-score from training controls; each without and with age/sex); the real label is kept; 50 canonical folds; metric pooled AUC; 100 replicates per scenario.
- H1: every feature is permuted independently among the 200 patients (seed 28100 + r). The features no longer carry a relation with the label or the confound.
- H2a: every feature is permuted independently within era (seed 28300 + r): the era-feature and era-label relations are kept, the individual-feature relation is destroyed.
- H2b: the same, within the year x device cell (seed 28500 + r).
- Real-data reference: the same pipeline on `BP` without permutation (must reproduce the pooled AUC of I2 0.7121 for the variant with age/sex, tolerance 5e-3).
Decision rule (same as A3 of round 27): mean pooled AUC of I2 without age/sex <= 0.52: no evidence of leakage; 0.52--0.55: small use; > 0.55: evidence of leakage. Also reported: the 5th--95th percentile, the difference from C1 per replicate, and whether the 5th percentile > 0.5.
Prediction: H1 I2 about 0.50 (0.49--0.51); H2a and H2b C1 above I2; I2 without age/sex 0.50--0.55 (partly from A3: the estimation of the era coefficient from few controls).

## 6. Promised double checks

(i) Spearman, net benefit, and the R^2 statistics recomputed by a second code written separately (ROC/statsmodels formulas); (ii) the partial and full tests compared with `mlconfound` (p aligned within Monte Carlo error); (iii) one H1 replicate and one H2a replicate recomputed with a second implementation of the pipeline (direct scikit-learn) and must agree to 1e-6; (iv) reproduction of the existing numbers (I2, F2, C1, Floor).

## Addendum 1 (written after a one-replicate dry run, BEFORE the full run; 2026-10-21)

The dry run ran 1 replicate of each scenario (to check the code and the reference reproduction; the real-data reference passed: I2 with age/sex 0.7121, I2 without 0.7104, C1 0.6663). One H2a replicate gave a pooled AUC of C1 without age/sex 0.603 and I2 0.640, outside the prediction 0.50--0.55 for I2. One replicate is not enough to conclude (the SD of the AUC between replicates in A3 is about 0.04). Therefore ONE diagnostic is added, fixed before the full results were seen:
- In every H1, H2a, H2b replicate and in the real data, also compute the AUC within era strata and within year x device strata (estimator `round9.strat_auc`; scores averaged over 10 repetitions) for C1 and I2 without age/sex. Reason: if the features are permuted within era, the individual information is destroyed, so the AUC within era must be about 0.5; if it is far above 0.5, the permutation did not destroy the individual information and that scenario is not valid as a null test. Rule: a scenario is declared valid as a null test if the mean AUC within its strata is within 0.48--0.52.
- The Hamdan decision rule above (0.52 / 0.55) is not changed; if a scenario is not valid as a null test, this is reported and the interpretation is limited.

## Addendum 2 (POST HOC; written after the main round 28 results were seen, before this analysis was run; 2026-10-21)

The main H1 result (pure noise features) shows a pooled AUC of I2 without age/sex of 0.564 (above the 0.55 rule) while C1 is 0.508. Two POST HOC analyses (exploratory, not in the plan; reported as post hoc) are added to understand the mechanism and compare with the real data:
- (P1) Real data against the null distribution: for the difference I2 - C1 (without age/sex) in the pooled AUC, within era strata, and within year x device strata, compare the real-data value with the difference in each H1, H2a, H2b replicate (the proportion of null replicates >= the real data). Only uses the round 28 result files that already exist.
- (P2) Mechanism: repeat H1 (features permuted independently among the 200 patients; 100 replicates; seed 28700 + r) with different normative-layer design matrices: intercept only; age and sex; era only; full (intercept, age, sex, era). Working hypothesis: the leakage comes from the term -D*B_hat (the coefficient estimation error from the same training controls for all patients in one fold); if true, the intercept-only design gives an AUC of about 0.5 and the leakage increases with the covariates that are related to the label.
