# Analysis plan, round 29: sensitivity tests of the normative layer without the leak (written BEFORE code and results)

Date: 2026-10-21. Status: sensitivity tests requested by the owner after round 28 found covariate leakage in the normative layer; exploratory, not part of the registry of 89 tests; NOT a new model and does NOT replace the main results (I2 and F2 remain as the planned results). Patients and the 50 canonical folds are the same as in all previous rounds (27 rounds + round 28 were run on the same data without a hidden test set), so every improvement of a number from this round must be read with selection optimism in mind. All results are reported, including if worse. CPU only; no git commit; SHA-256 hashes of the document and code are recorded in `PRAREGISTRASI_hash_putaran29.txt` before the run.

## 0. Background and mechanism conjecture

In the current normative layer (V0), the coefficients B (intercept, age, sex, era) are fitted by OLS on the non-epilepsy patients of the training data of a fold, then z = (x - D B)/s for all patients. The residuals of the training controls are exactly orthogonal to D (in-sample), whereas the residuals of the training epilepsy patients and the test patients contain the term -D(B_hat - B) (estimation error). This asymmetry allows logistic regression to learn the component that is linear in the covariates as an epilepsy marker, and that component is related to the label through age and era. Round 28: pure-noise features produce an I2 AUC of 56.4 (C1 50.8), rising with the covariates related to the label (intercept design 49.7; age+sex 51.8; era 53.6; full 55.4).
Conjecture: if all patients used to train the model (controls and epilepsy) are treated symmetrically with respect to the estimation of B (none is inside the estimation sample), the leakage disappears.

## 1. Normative layer variants (I2 view only: 114 spectral features, `round4.BP`; all with the same L2 logistic regression, `ek.nested_lr()`, 50 canonical folds)

- C1: raw features (without the normative layer; reference).
- V0: the current layer (intercept, age, sex, era design; training controls). Must reproduce the pooled AUC of I2 without age/sex 0.71042 and with 0.71210.
- V1 (separate reference): the training controls are split in two at random stratified by era (seed 29000 + fold index): the first half R (size ceil(n_era/2) per era) only to fit B and s; the second half and all training epilepsy patients train the logistic regression (R is not used for training). z is computed for all patients with B and s from R.
- V2 (cross-fitted): training patients are split into 5 random class-stratified groups (seed 29000 + fold index); the z of patients in group k uses the B and s fitted on the training controls outside group k; the z of test patients uses B and s from all training controls.
- V3 (without era): as V0 with the design intercept, age, sex.
Each variant is run without and with age/sex as input of the logistic regression (without = leakage test; with = counterpart of I2).
Not run: P3 and NM (they need a GPU, about 4 hours for NM per variant); therefore F2 is not recomputed and only the I2 view is tested. This is a limitation that will be stated.

## 2. Tests

A. Leakage (noise features): scenario H1 (every feature permuted independently among the 200 patients; seed 29100 + r) and H2a (permuted within era; seed 29300 + r), 100 replicates per scenario; the SAME permuted features for all variants in one replicate (paired). Metric: pooled AUC without age/sex for C1, V0, V1, V2, V3; AUC within era strata as a validity diagnostic.
 Rule R1 ("leakage solved") per variant: mean pooled AUC <= 0.52 on H1 AND on H2a; 0.52-0.55 = small use; > 0.55 = leakage.
B. Real data: per-fold AUC (mean of 50 folds; Nadeau-Bengio corrected CI) and pooled AUC for C1, V0-V3 (without and with age/sex); per-fold differences against C1 and against V0 (Nadeau-Bengio corrected t-test; descriptive, without correction); AUC within era and year x device strata and a bootstrap test within strata (`round10.within`, 2,000 replicates) against the metadata baseline and C1 for the variants with age/sex.
 Rule R2 ("benefit of the normative layer without leakage is supported") for variants that meet R1: the per-fold difference against C1 is positive with p < 0.05 (Nadeau-Bengio) in the variants without AND with age/sex, AND the proportion of H1 replicates with a difference (variant - C1) >= the real-data difference is <= 0.05 (on the pooled AUC).
C. Confounder test (Spisak; round28 implementation): partial test (era, device, montage, age) and full test (era, device, year x device) on the scores of V0-V3 without and with age/sex, 10,000 permutations (seed 29200 + index). Primary family (Holm, 2 tests): partial era for V1 and V2 with age/sex. Others descriptive.
 Rule R3: the partial era bias "disappears" if p_Holm >= 0.05 (this does not prove the absence of bias).

## 3. Predictions (my subjective probabilities, written before the results)

- V1 and V2: AUC H1 and H2a 0.48-0.52 (R1 met; chance V2 70%, V1 70%). V3: stays 0.52-0.55 (R1 not met; 70%).
- Real data: pooled AUC of V1 lower than V0 (0.66-0.70; fewer training controls), V2 0.68-0.71, V3 close to V0.
- Per-fold difference against C1: V1 and V2 between 0.00 and +0.04, CI includes 0 (chance 65%); R2 not met (65%).
- Partial era for V1 and V2: p_Holm >= 0.05 (55%).
- Results within strata (I2 vs the metadata baseline in year x device): remain far above the baseline for all variants (difference > +0.25; 85%).

## 4. Promised double checks

(i) A second implementation of V1, V2, V3 (explicit per patient, loop) for 3 folds and 1 H1 replicate must agree to 1e-9; (ii) V0 must reproduce 0.71042 and 0.71210 and the H1/H2a reference of round 28 (same replicate, different seed: not compared identically; compared on the mean within Monte Carlo error of 3 SE); (iii) the confounder test is compared with `mlconfound` for V1 and V2 (partial era and full); (iv) split properties: R and the cross-fit groups do not overlap, do not contain test patients, and R is not used to train V1.

## Addendum 1 (written after the dry run, BEFORE the full run; 2026-10-21)

The dry run (one H1 replicate and one real-data run for each variant) has been seen. What was seen: V0 reproduced exactly (0.71042 without and 0.71210 with age/sex; C1 0.66627). Real data (pooled AUC without / with age-sex): C1 0.648 / 0.666; V1 0.665 / 0.665; V2 0.710 / 0.709; V3 0.656 / 0.670. One H1 replicate (pooled AUC, without age-sex): C1 0.488; V0 0.579; V1 0.571; V2 0.582; V3 0.522. One replicate is not enough to conclude (SD between replicates about 0.04), but it shifts the mechanism conjecture: the leakage does not seem to be due to the asymmetry of the estimation sample (V1 and V2 remove it but do not appear to reduce the leakage), but to covariates related to the label that re-enter through the error term -D(B_hat - B) in all patients. According to this new conjecture, V3 (without era) reduces the leakage because fewer covariates are related to the label, and exact removal of the covariates in all training patients (classical covariate regression) eliminates it.
Therefore ONE last variant is ADDED (no other variant will be added):
- V4 (covariate regression on all training patients): B and s are fitted by OLS on ALL training patients (controls and epilepsy; intercept, age, sex, era design), z = (x - D B)/s for all patients. This is no longer a "control-based" layer but a standard covariate adjustment; the training residuals are orthogonal to D.
All rules R1-R3 apply to V4. Prediction of V4 (written before the full results): H1 and H2a 0.48-0.52 (chance 70%); real-data AUC lower than V0 (0.60-0.68; 65%) because information correlated with era and age is also removed; within strata it stays above the metadata baseline (80%).
The predictions about V1 and V2 in section 3 are recorded as they are; they will most likely miss for R1.
