# Analysis plan, round 31: frozen BENDR and LaBraM embeddings with V4 covariate adjustment (2026-10-21)

Aim: to test whether the comparison of foundation models with I2-adj is fairer when the FROZEN BENDR and LaBraM embeddings receive the same treatment as I2-V4 (OLS covariate adjustment on all training patients) and the same `nested_lr` probe. CPU only; no fine-tuning (the fine-tuned LaBraM-FT and CBraMod-FT models are NOT changed and remain without adjustment).
Code: `round31_fm_beku.py`. Data: per-recording embeddings of the first session (`eegbench_rep_D.npz`), patient = mean over recordings (as in round7). 50 canonical folds.
Variants per encoder: raw (without age/sex), raw+AS, V4, V4+AS. V4: OLS regression on [intercept, age, sex, era indicator] (I2 design matrix, 7 columns) fitted on all training patients; residuals divided by the training residual SD. BENDR (8,704-d): StandardScaler + PCA 128 fitted on the training patients before the probe; LaBraM (200-d) used directly. Age and sex are appended before `nested_lr` in the +AS variants.
Primary tests (Holm, 2 tests): V4+AS - raw+AS for BENDR and for LaBraM (per-fold AUC difference, Nadeau-Bengio corrected t-test). Descriptive (BH): V4 - raw; V4+AS - round7 comparator (without age/sex); V4+AS - C1, I2-adj, LaBraM-FT, metadata baseline. AUC within year x device strata.
Leakage test: patient embeddings permuted between patients without age/sex (H1 global; H2a within era), 20 replicates per cell, raw and V4 variants; rule as in round 30 (mean pooled AUC <= 0.52 no evidence of leakage; > 0.55 leakage). With 20 replicates the power is limited, as for the NM null.
Mandatory checks: reproduction of the round7 comparators (BENDR 0.553, LaBraM 0.640; tolerance 6e-4) and the V4 perturbation test (test X, test labels, orthogonality of training residuals).

## Predictions (subjective probabilities; written before the full run)
- Round7 reproduction passes (95%).
- V4 null <= 0.52 for both encoders and both scenarios (75%); raw null <= 0.52 (80%).
- V4+AS - raw+AS: |difference| < 0.03 and not significant for both (70%).
- Per-fold AUC of V4+AS: LaBraM 60-68, BENDR 50-60 (65%).
- V4+AS does not exceed I2-adj (68.0) on the per-fold AUC for both encoders (85%).
- Within year x device strata V4+AS > metadata baseline (43.2) with a difference > +0.20 for LaBraM (55%).

## Note
A dry run of the first 3 folds (test AUC per fold, not a summary) was seen before this document was finalized, only to make sure the code runs; it did not change the design. Patients and folds are the same as in the previous 30 rounds.
