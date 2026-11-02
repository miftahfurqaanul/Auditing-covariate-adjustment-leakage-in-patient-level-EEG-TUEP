# Analysis plan, round 30: P3, NM, and F2 without the leak, and a comprehensive leakage audit (written BEFORE code and results)

Date: 2026-10-21. Request of the owner: all leakage listed in the manuscript must be clean. Status: fixes and audit of the P3 and NM view pipelines and the F2 fusion, following the findings of rounds 28-29 (the control-based normative layer leaks covariates related to the label; covariate regression on all training patients, V4, removes it in I2). Patients and the 50 canonical folds are the same as in all previous rounds (27 rounds + rounds 28 and 29 on the same data, without a hidden test set); all results are reported, including worse ones. No git commit; SHA-256 of the document and code are recorded in `PRAREGISTRASI_hash_putaran30.txt` before the run. One GPU process at a time; CPU scripts use CUDA_VISIBLE_DEVICES="" and 12 workers (8 while the GPU is running).
Information already known before this document: the I2-V4 result (round 29). The results of P3-V4, NM-V4, and F2-V4 were not known.

## 1. Definition of V4 for P3, NM, and F2

- I2-V4: V4 predictions with age/sex from round 29 (`hasil29/B_prediksi_data_asli.npz`, key `V4_AS`).
- P3-V4: all steps as P3 (`round17_p3.p3_predict`: covariances per window in 5 bands, per-era realignment, tangent space, mean per patient, PCA 50 on training patients, + age/sex, `ek.nested_lr`), except that the per-era alignment reference = the Euclidean mean of the window covariances of ALL training patients in that era (controls and epilepsy; no label; global reference fallback if < 10 windows). Test patients are never used for the reference.
- NM-V4: all steps as NormMamba-F (`normmamba.run_fold`, variant F; a grid of four configurations on a 20% validation of training patients, one seed; the chosen configuration is trained with five seeds; seeds and validation split identical to round 16), except that the coefficients of the segment normative layer (intercept, age, sex, era design) are fitted by OLS on the segments of ALL training patients (not only controls), and s = the residual standard deviation on the same segments. The patient summary (median token + standard age/sex) is unchanged.
- F2-V4 = mean probability of I2-V4, P3-V4, NM-V4 (no parameters). The selection of the nested composition is NOT recomputed (it would need about 20 GPU hours); the composition stays I2+P3+NM as in the plan of round 18.
- Comparators without covariate adjustment (unchanged, free of this type of leakage): C1 (raw I2), Riemann without alignment (`round17/p3.npz` Riemann_raw), raw Mamba (`round16` raw variant), metadata and demographic baselines.

## 2. Leakage test (features without information; the real label, age, sex, era, and device are kept)

- P3 (CPU): the set of covariance windows of each patient is exchanged between patients: H1 = random permutation over the 200 patients; H2a = permutation within era; 100 replicates per scenario (seed 30100 + r and 30300 + r); the windows follow the new owner (age, sex, era, label of the new owner). P3 pipeline (current V0 and V4) without age/sex; 50 canonical folds; pooled AUC; AUC within era strata as a validity diagnostic. The pair V0 and V4 uses the same permuted data.
- NM (GPU, reduced version because the full cost is about 4 hours per replicate): only repetition 0 (5 folds), fixed configuration d = 32 and L = 32, one seed, maximum 20 epochs (early stopping as usual), the age/sex columns of the patient summary filled with zero; the segment set of each patient is exchanged at random among the 200 patients (H1); 20 replicates (seed 30500 + r); NM variants V0 and V4 on the same permuted data. Metric: AUC of repetition 0 (200 patients). Limited power (SD of the AUC per replicate about 0.04; mean of 20 replicates about 0.01).
- Rule R1 for each variant: a mean AUC on features without information <= 0.52 means no evidence of leakage; 0.52-0.55 small use; > 0.55 leakage. For NM with limited power: the conclusion "no evidence of leakage" is stated only if the mean <= 0.52 AND the 95th percentile of the mean (SE 1.645) <= 0.54.

## 3. Real-data evaluation (50 canonical folds)

Reported: per-fold AUC with the Nadeau-Bengio corrected CI, pooled AUC, threshold-0.5 metrics, Brier score and calibration slope for P3-V4, NM-V4, F2-V4. Per-fold differences (Nadeau-Bengio corrected t-test): F2-V4 - I2-V4; F2-V4 - C1; F2-V4 - old F2; P3-V4 - raw Riemann; NM-V4 - raw Mamba; combined effect = mean of the three view differences (V4 - raw) per fold. New primary family (Holm, 2 tests, following round 25): F2-V4 - LaBraM-FT and F2-V4 - CBraMod-FT. Others descriptive without correction. AUC within strata (year, device, year x device, year x montage) against the metadata baseline and C1 (`round10.within`, 2,000 replicates). Confounder test (Spisak; round 28 implementation; 10,000 permutations): partial (era, device, montage, age) and full (era, device, year x device) for P3-V4, NM-V4, F2-V4; primary family (Holm, 2 tests): partial era for F2-V4 and NM-V4. Sensitivity without patients of era <= 2009 (same predictions, pooled AUC).

## 4. Predictions (subjective probabilities)

- P3-V4: AUC H1/H2a 0.48-0.52 (75%); old P3 (V0) above 0.52 (55%).
- NM-V4: mean AUC H1 <= 0.52 (65%); old NM (V0) above 0.52 (50%; it is not yet clear whether a nonlinear network leaks as much as I2).
- Real data: P3-V4 per-fold AUC 62-70 (lower than the old P3 70.8; 60%); NM-V4 62-70 (50%); F2-V4 per-fold AUC 66-71 (lower than the old F2 73.5; 80%).
- F2-V4 - I2-V4: |difference| < 0.03 and not significant (70%).
- F2-V4 - LaBraM-FT and CBraMod-FT: remain positive (65%) but p_Holm < 0.05 for both (35%).
- Partial era of F2-V4: p >= 0.05 (65%).
- Within year x device strata, F2-V4 remains > the metadata baseline with a difference > +0.25 (85%).

## 5. Comprehensive leakage audit (v2)

Run on all pipelines listed in the manuscript; the pass criteria are set here.
A. Perturbation test (dynamic; CPU): for I2-V4, P3-V4, NM-V4 (token and summary construction), and the fitted reference: changing the data or labels of test patients must not change the features, reference, or coefficients of training patients, and must not change the predictions of test patients (for labels). Also for the CPU comparators: metadata baseline, Rohira-F, DNM-F, Tang-F (the part that can run on CPU), frozen embedding + logistic regression (BENDR, LaBraM).
B. Random-feature test (section 2) for every pipeline with covariate or domain adjustment; global label permutation (100 permutations) for I2-V4 and P3-V4 as a check of leakage through selection (note: this test CANNOT detect covariate leakage because it destroys the covariate-label relationship; this is why the random-feature test is needed).
C. (Static) code review for every fitted step: normalization, PCA, choice of C, NM grid, early stopping, FM fine-tuning validation (LaBraM-FT, CBraMod-FT), published comparators, fold splitting, and the strata computation: is any statistic computed from test patients or from all patients before the split. Each finding is recorded as file:line.
D. Identity and integrity checks (the existing audit_kebocoran.py: 16 checks, rerun) plus: no identical recording across patients, first session only, consistent labels.
E. Check that all predictions used in the manuscript come from pipelines that pass A-C; for components that cannot be fully audited (foundation-model pretraining includes the parent corpus of TUEP without labels), this is stated as a limitation.

## 6. Double checks

A second implementation of P3-V4 (eigh and explicit alignment per band) and of the NM-V4 tokens (OLS per segment via sklearn) on 3 folds; reproduction of one NM-V0 fold (r0k0) against the old JSON (test AUC of the fold within 0.02; the GPU is not fully deterministic); reproduction of P3-V0 (pooled AUC 0.68318) and NM-V0 (0.70509); `mlconfound` for F2-V4 era (partial) and year x device (full); all numbers in the manuscript are verified with asserts.

## 7. Treatment of the registry and the manuscript

The registry of 89 primary tests is not changed. Tests involving models with the old normative layer are marked "with leakage" and kept only as the original planned output. All main numbers of the manuscript will be replaced by leak-free results when this round finishes. The new test family (section 3) is Holm-corrected within the family and BH-corrected over all new tests; it is not merged with the old registry.

## Addendum 1 (written BEFORE the NM-V4 results and before the full P3 run finished; 2026-10-21)

While reading the code it was found that other numbers in the manuscript also depend on the old normative layer (V0). Recomputed with I2-V4 (or F2-V4 where needed), all descriptive with no new decision rule, in the script `round30_eval2.py` (CPU):
1. Published and pretrained comparators (per-fold difference, Nadeau-Bengio corrected t-test): I2-V4@199 - Tang-F@199 (I2-V4 retrained on 199 patients as in `round7`), I2-V4 - Rohira-F, I2-V4 - DNM-F, I2-V4 - BENDR, I2-V4 - frozen LaBraM, and F2-V4 - DNM-F, Rohira-F, BENDR, frozen LaBraM (F2-V4@199 is not computed because P3 and NM are not retrained on 199 patients). Correction: BH over all new tests together with section 3.
2. Transfer to Kural: I2-V4 (fitted on all 200 TUEP patients; era dummy 0 for Kural as the reference group, as in `round12b`) vs C1 transfer; AUC with CI and a permutation test against 0.5; the difference I2-V4 - C1 with a paired bootstrap (2,000). Emotiv uses the band-power model without a normative layer and does not change.
3. Deviation map (A5) with the out-of-fold V4 z; transient sensitivity (A6) with I2-V4 retrained on features from which the sharp windows were dropped; between-session stability (A7a) with I2-V4 (later sessions of test patients; model from the first session; z uses the V4 coefficients of the fold); hard patients (A8) with the same definition (F2-V4 wrong in >= 8 of 10 repetitions and the mean score of I2-V4, P3-V4, NM-V4 on the wrong side; variables: age, sex, era <= 2009, number of windows, drowsiness fraction, photic fraction).
4. Nested F2 is NOT recomputed (reason in section 1); the old nested number (72.4) and the nested comparison with foundation models can only be reported as the old pipeline.
5. DNM-M (control-based adjustment) uses the old normative layer and is not recomputed; its row is removed from the main test table.

## Addendum 2 (written AFTER seeing the first 10 replicates of the P3 null and the diagnostic of 6 replicates; BEFORE the complete P3 null, the NM null, and the full NM-V4 results; 2026-10-21)

Interim observation (not a final result): in the first 10 H1 replicates, the mean pooled AUC of P3 V0 and P3 V4 are both about 0.565 (sample percentiles 0.50-0.64), above the 0.52 rule. A diagnostic of 6 replicates (same permuted data) shows that the AUC of P3 without realignment ('none' mode) averages 0.49 whereas V4 is 0.554, so era-specific realignment with a reference from all training patients does NOT remove the leakage. Conjecture: the era-specific reference is fitted from few patients (about 32 per era), its estimation error is shared by all patients of one era and brings the era identity (related to the label) into the features, similar to the coefficient error in I2-V0; the tangent residuals are not orthogonal to era as in covariate regression.
Action (fixed before further results):
1. The P3 V0/V4 null run (100 replicates) is allowed to finish and is reported in full; V4 for P3 is declared to FAIL R1 if the final result confirms this observation.
2. A new variant P3-V5 (the only addition for P3): realignment with ONE global reference (Euclidean mean of the window covariances of all training patients, without era separation), followed by covariate regression on the PCA features: OLS of the principal components (50) on [1, age, sex, era indicators] fitted on ALL training patients, residual + age/sex (as I2-V4) -> `ek.nested_lr`. H1 and H2a null with 100 replicates with seeds 30700 + r and 30900 + r (permutation of the window sets as before), the same permuted data for V5; the same R1 rule; the same real data. The leak-free F2 uses P3-V5 if R1 is met; otherwise a leak-free F2 cannot be stated and only the F2 with the other two views is reported.
3. The GPU order is changed: the (reduced) NM null is run BEFORE the full NM-V4 so as not to spend about 4 GPU hours if NM-V4 leaks; the first two folds of NM-V4 (r0k0 and r0k1) are already finished and valid (resumable). If the NM null shows leakage in V4, NM will be changed in a way to be written in addendum 3 before the full run.
4. Methodological note: the random-feature test based on exchanging window sets uses the owner's era; for era-specific transforms this means the "wrong" era is applied to the windows, which exposes the estimation error of the era reference. This test is valid as a detection of estimation leakage, but the result without realignment ('none') serves as a control.

## Result note (written after the NM null finished; not a design addendum)

Reduced NM null (20 replicates, H1, without age/sex): NM V0 mean AUC 0.561 (sd 0.047), NM V4 0.515 (sd 0.043; SE of the mean 0.0095; 95th percentile of the mean 0.531). Rule R1 for NM (mean <= 0.52 and 95th percentile of the mean <= 0.54): V4 MET, V0 not. The full NM-V4 (50 folds) was run afterwards (the first two folds had been finished before).
