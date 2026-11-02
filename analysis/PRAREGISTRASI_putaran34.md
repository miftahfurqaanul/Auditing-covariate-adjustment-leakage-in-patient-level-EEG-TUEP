# Analysis plan, round 34: Tang-F with V4 covariate adjustment (GPU + CPU) (2026-10-21)

Complements `PRAREGISTRASI_putaran32_33.md` (tier 4 without Tang-F). Tang-F = Tang et al. 2024: order-2 PS features per second (506-d), 60-second sequences, Bi-LSTM 2x128 + attention; round7: 199 patients (one patient excluded), 50 canonical folds, 100 epochs, best epoch by patient-level validation AUC (20% of training patients). Small data (4,702 sequences, 0.57 GB; about 12 seconds per fold on the GPU), so the two variants are retrained in one script.
Code: `round34_tang_v4.py` (GPU, one GPU process at a time; `--check`, `--run`, `--null n`) and `round34_eval_tang.py` (CPU). Patients and folds are the same as in round7.

## Variants
- F: Tang-F of round7 retrained exactly (seed 2000 + 10r + k, split_val 100r + k), only saving the validation scores. cuDNN nondeterminism can shift the scores; the per-fold AUCs are compared with round7 (reported; the run stops at evaluation if the mean |difference| >= 0.06).
- V4: D_patient @ B is subtracted from every second of features; B = OLS of the 506 features on the I2 design [intercept, age, sex, era indicator] fitted on ALL training patients (training + validation; no labels), computed exactly from per-patient sums. The built-in standardization is kept. No age/sex as input (equivalent to Rohira-V4).
- +AS (stacking, CPU evaluation only): logistic regression on the 32 validation patients of each fold with features [score logit, age, sex] applied to the test patients, for F and V4. Note: the validation scores are slightly optimistic (the epoch is chosen from the same validation); n = 32; this is a descriptive variant.
- Comparators recomputed on 199 patients: I2-adj@199 (V4 + age/sex + nested_lr) and C1@199.

## Tests
Primary (Holm, 3 tests; per-fold AUC, Nadeau-Bengio t-test): Tang-V4 - Tang-F; I2-adj@199 - Tang-V4; Tang-V4+AS - Tang-F+AS. Descriptive (BH): I2-adj@199 - Tang-F; I2-adj@199 - Tang-V4+AS; Tang-V4 and Tang-V4+AS - C1@199; Tang-F+AS - Tang-F; repeated Tang-F - Tang-F round7. Pooled AUC and AUC within year x device strata.
Leakage test (`--null`, reduced version as for NM): only repetition 0 (5 folds), the sequences of each patient are replaced by those of another patient (H1 global; H2a within era; the new owner carries its own label/era), variants F and V4, 20 replicates per cell (seeds 34000 + rep and 34300 + rep), pooled AUC over 199 patients. Rule: "no evidence of leakage" only if the mean <= 0.52 AND the mean + 1.645 SE <= 0.54; > 0.55 leakage.
Mandatory checks before the run (`--check`/`--run`, CPU 2 epochs): exact B = direct OLS over all seconds; training residuals orthogonal to the design; changing the test data does not change B; the `train_fold` copy is identical to `tang_fixed.train_fold` on CPU.

## Predictions (subjective probabilities)
- Mandatory checks pass (90%); the mean AUC of the repeated Tang-F within 0.03 of round7 (80%).
- Tang-V4 - Tang-F: |difference| < 0.04 and not significant (65%); per-fold AUC of Tang-V4 between 50 and 62 (70%).
- I2-adj@199 - Tang-V4 positive and Holm < 0.05 (55%; I2-adj@199 - Tang-F is already +0.130, p 0.009).
- V4 null <= 0.52 (60%; a nonlinear LSTM can exploit the era estimation error as P3-V4 did; for the H1 scenario F <= 0.52 70%).
- Tang-V4+AS does not differ from I2-adj@199 (75%).

## Note
The dry run/`--check` on CPU and the smoke evaluation were seen before this document; no GPU training result was seen. Decision rule: no claim of superiority comes from these results; results are reported as they are.
