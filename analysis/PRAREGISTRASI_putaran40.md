# ANALYSIS PLAN, ROUND 40 (DRAFT 2, approved by the owner 2026-10-23 including the GPU cells; hashed before the run): correcting the R1/R2 metric to the mean per-fold AUC and diagnostic test of the mechanism

Draft date: 2026-10-23. Status: the CPU+GPU scope was approved by the owner on 2026-10-23. The SHA-256 hash is recorded AFTER approval and BEFORE the first run (`PRAREGISTRASI_hash_putaran40.txt`). After that this file is not changed; changes only through dated addenda.
Design source: Priority Experiment 3 of the internal critical evaluation of the project (rechecked; see section 1 for corrections to its assumptions).

## 1. Findings of the check before the design (facts verified from code and CSV)

1. All replicate CSVs of rounds 36-39 (`hasil36/*null_replikasi.csv`, `hasil37/tang_*_null_replikasi.csv`, `hasil39/B_cpu.csv`, `B_gpu.csv`) store **one pooled AUC per replicate**: `roc_auc_score(y, P)` over all patients from one 5-fold CV (`round39_simulasi_kebocoran.py` lines 171, 175; `round37_tang_log.py` line 61; `round38_nm_paired.py` line 67). **The per-fold AUC was not stored.** So "recompute from the CSV" (the instruction in the execution instructions) is **not possible**; the per-fold value must be computed by rerunning the cell concerned with the same seed.
2. Consequence: the correction of the R1/R2 metric is done on (A) the round 39 simulation CPU cells (cheap), (B) the real-EEG null cells that are CPU (Rohira RF/LR, round 36), and (C, a separate stage, after A-B) the GPU null of NM/Tang. Stage C is not part of the approval of this draft.
3. The claim "the network baseline < 0.5 is due to the pooled AUC" is a **hypothesis** (H40-1), not a fact; it has not been tested.
4. The theoretical claim "asymmetric support boundary / symlog kink" is an **exploratory hypothesis** (H40-3).

## 2. Hypotheses and decision rules (fixed before the run)

**H40-0 (reproduction, condition to proceed).** Rerunning a cell with the same seed and code reproduces the stored pooled AUC with a maximum difference < 1e-9 on CPU cells, and (because of GPU nondeterminism) the maximum difference is reported without a hard threshold for GPU/NM cells (reference: the NM replicates of round 38 store the pooled AUC; soft criterion <= 0.02 per replicate, mean <= 0.005). If a CPU cell fails, stop and record it; no round 40 result is reported as confirmation.

**H40-1 (pooled bias).** In the H1 null cells without adjustment (LR, RF, ET; raw transform), the mean per-fold AUC (mean of 5 folds, then mean over replicates) is within 0.500 +- 0.010 while the pooled AUC is lower than the per-fold AUC; bias size = per-fold - pooled, bootstrap CI over replicates.
- Supported if: the lower CI bound of the bias > 0.003 and |mean per-fold - 0.5| <= 0.010.
- Not supported if: the bias CI includes 0 or the mean per-fold deviates > 0.010 from 0.5.
- The same criterion applies to NM-N0 H1 (stage D, 50 replicates; pooled AUC of round 38: 0.4896 on 20 replicates); prediction supported 0.45.
- Prediction: chance supported for RF/ET 0.35; LR 0.30 (I do not know whether the CPU baseline is really < 0.5; a baseline < 0.5 was recorded for the GPU networks, which are not part of this stage).

**H40-2 (R1/R2 again on the mean per-fold).** The rules and thresholds are NOT changed (R1: mean <= 0.52 and mean + 1.645 SE <= 0.54 = no evidence; > 0.55 = leakage; paired R2 only for H1: difference <= 0.01 and upper bound <= 0.03; for H2a use the absolute AUC). SE = SD between replicates of the per-fold mean of each replicate divided by the square root of the number of replicates. Output: a decision table per cell for the pooled vs the per-fold version and the number of cells that change category.
- Prediction: >= 90% of cells do not change category (chance 0.75); cells that change are concentrated in the category "uncertain" (chance 0.70).
- Addition (not a replacement of R1): TOST equivalence test at margin +-0.02 for the mean per-fold null, alpha 0.05, only for cells with >= 30 replicates.

**H40-3 (mechanism diagnostics, exploratory).** The diagnostics are computed per fold only from the training windows and patients (raw features, covariates D, and training labels for the covariate score; no test data), then averaged over 5 folds:
- `skew` = the median across features of |skewness| of the raw training-window features;
- `Dq05` = window-level residual = raw features minus the window OLS on D (fitted on training windows); covariate score eta = linear predictor of the logistic regression label ~ D[:,1:] on training patients; upper and lower tertiles of eta among training patients; Dq05 = the median across features of |q05(residual | upper tertile) - q05(residual | lower tertile)| / IQR of the residual;
- `cusp` = the mean across features of the proportion of training windows with |z| < 0.25, z = (x - median)/(IQR/1.349) (training median and IQR), divided by 0.1974 (the value for a normal), so cusp = 1 for a normal;
- comparator: `lk` = log10 of the median kurtosis (stored as `kurtosis`, from all windows, without labels).
Model: delta (RF, XGB is not used in this stage; RF and ET only) ~ lk + n/m/p/s/imbalance (baseline) versus baseline + skew + Dq05 + cusp, trained with **leave-one-distribution-out** (lognormal, t3, chi-square, contamination, normal in turn).
- A diagnostic is considered to "explain better than kurtosis" only if the LODO out-of-sample R2 rises >= 0.10 and the cell-bootstrap CI (2,000) of that rise does not include 0.
- Otherwise it is reported as "not supported"; there is no tuning of the diagnostics after seeing results (one definition per diagnostic, above).
- Prediction: skew+Dq05 beats kurtosis by this criterion with chance 0.35; cusp explains the symlog leakage on normal features with chance 0.25 (only a few normal cells; most likely inconclusive).
- A statement of theory may enter the manuscript only as "an empirical diagnostic on simulation" and not as a theorem.

## 3. Computational design

- **A. Round 39 CPU simulation** (LRpat, RF, ET; transforms raw, rank, symlog, symlogc, log; adjust none/pat/win per `configs_b`): the code `round39_simulasi_kebocoran.py` is used as a module (data generator, transforms, adjustment, classifiers, identical seeds); `round40_simulasi_perlipatan.py` adds an AUC output for each of the 5 folds (`fold_auc_0..4`), the pooled AUC (identical to round 39), and the H40-3 diagnostics. Stage B: **50 replicates** (heavy cell n=500, m=200, p=200: 30), not 100 as in round 39, to reduce time; the first 50 replicates use the same seeds as replicates 0-49 of round 39, so H40-0 can be checked. Stage C CPU (normal, t3, chi-square; s = 1; H1): 100 replicates.
- **B. Real-EEG null on CPU:** the LR and RF cells of round 36 (`round36_diagnosis_rohira.py`, `round36c_paired.py`) on 30 replicates H1 and H2a, with per-fold output.
- **C. GPU simulation:** XGBoost, MLP, Bi-LSTM + attention on the cells: lognormal H1 base, contamination H1 base, both with s = 0 (control), and normal/t3/chi-square H1 s = 1 (stage C); configuration `configs_b(..., "gpu")`; **20 replicates** per cell (round 39 used 30; the first 20 use the same seeds); per-fold output.
- **D. GPU NM (NormMamba) paired null:** N0 vs V4, H1 and H2a, **50 replicates** (seeds 30500+rep and 30900+rep as in round 38, so the first 20 replicates can be checked against `hasil30/nm_paired_replikasi.csv`), per-fold and pooled AUC. H40-1 check for the network: the N0 H1 baseline.
- **E. (Conditional, scope not approved now)** Tang symlog null (L, Lp) with per-fold output: only run after A-D finish and with an addendum that records the measured cost.
- One GPU process at a time; the GPU chain (C then D) runs concurrently with the CPU chain (A then B). CPU: `CUDA_VISIBLE_DEVICES=""`, 12 workers (8 if the GPU is busy). A pilot of 20 replicates on the base cells measures cost; the scope of the full run is not changed after the pilot except by addendum.

## 4. No-leakage guarantee and integrity

- All fitting (rank-normal quantiles, OLS coefficients, standardization) only on the training data of each fold; the per-fold AUC uses only the test patients of that fold. No test label enters the H40-3 diagnostics (computed from training features, covariates, and residuals).
- Independent check of round 40 (separate script `round40_cek_independen.py`): (i) reproduction of the stored pooled AUC (H40-0); (ii) the mean of the 5 per-fold AUCs recomputed with `sklearn` from the saved predictions (CPU cells: per-patient predictions saved to `hasil40/A_prediksi_*.npz` for a subset of 5 replicates); (iii) per-fold mean vs pooled on 3 cells computed by hand; (iv) the summary recomputed from the raw rows (1e-9); (v) test-label perturbation: shuffling test labels must not change the predictions of the fold.
- Negative control: cells s = 0, normal features, raw/rank: the decision "no evidence" on >= 95% of rows (reference round 39: 97.7%). Failures are reported, the criterion is not loosened.

## 5. What is reported whatever the result

A table of predictions vs results; cells that change decision; the pooled bias with CI; diagnostic results including those not supported; all check failures. No claim of "significant" without a test and p; no forbidden words (project rule).

## 6. Limits

Simulation is not clinical data. R1/R2 are operational criteria at the simulated n; they do not guarantee "no leakage at all". Round 40 was run on the same 200 patients as the previous rounds (researcher degrees of freedom remain unmeasured; Experiment 5 estimates the selection optimism).

## 7. Hash plan

`sha256sum PRAREGISTRASI_putaran40.md > PRAREGISTRASI_hash_putaran40.txt` before `run_queue_40.sh`; this draft file is renamed to `PRAREGISTRASI_putaran40.md` at that time without change of content (or with a change requested by the owner, recorded).
