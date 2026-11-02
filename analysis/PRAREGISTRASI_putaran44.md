# ANALYSIS PLAN, ROUND 44 (Priority experiment 2): F2-Clean (without age/sex in the final classifier), rank-normal before V4 with an age^2 term, IPW comparator, 30-window sensitivity, Spisak test

Date: 2026-10-23. Written and hashed BEFORE any variant of round 44 was run. Already known: F2-adj 69.7 (per fold) = mean of I2-adj, P3-adj, NM-adj; the three views feed age and sex (AS) to the final classifier after V4; Spisak test (partial, given Y) on the F2-adj score: era eta^2 0.01 (Holm 1.0), age r^2 0.13 (p <= 1e-4); round 40: NM-V4 H1 difference +0.019 (does not pass fully); round 41: AUC of F2-adj within year x device strata with a 10-year age caliper 0.730 [0.615, 0.829]; round 42 (CPU): rank-normal and log remove the RF/XGB leakage, raw does not. Hash: `PRAREGISTRASI_hash_putaran44.txt`. Changes only through dated addenda.

## 1. Question
The dependence of the F2-adj score on age (r^2 0.13) arises because age and sex are fed back as classifier inputs after V4 removes their linear effect. Does F2-Clean (without AS) (i) remove the dependence on age according to the Spisak test, (ii) at a small cost in AUC, (iii) still retain EEG signal within year x device x age-caliper strata, and (iv) is rank-normal + V4 (with an age^2 term) better than or equivalent to IPW weighting (features unchanged)?

## 2. Design
**Protocol:** 50 canonical folds (`A.J`); all fitting only on the 160 training patients; per-fold AUC primary (Nadeau-Bengio CI), pooled AUC secondary. Patient scores for the Spisak test and the strata analysis = mean over the 10 repetitions of the out-of-fold score.
**Covariate design:** D1 = `A.D_I2` (intercept, age, sex, 4 era indicators); D2 = D1 + an age^2 column = ((age - mean over all patients)/SD over all patients)^2 (a function of the covariate only; no label).
**Views and variants (without AS unless written +AS):**
- I2: patient median BP (114 features, `A.BP`) -> [rank-normal] -> V4 (OLS on D1 or D2, residual / training SD) -> [+AS] -> `nested_lr` (9 values of C, I2-adj recipe). Variants: `asis` (no rank, D1, +AS; reproduction control of I2-adj), `noAS` (D1), `noAS-D2`, `rank-noAS-D1`, `rank-noAS-D2`.
- P3: P3-V5 (one global Riemannian reference from the windows of training patients, 855-d tangent vectors averaged per patient, PCA 50 on training patients) -> [rank-normal on the components] -> V4 (D1/D2) -> [+AS] -> `nested_lr`. Same variants as I2 (`asis` = P3-adj, reproduction control).
- NM (GPU): the NM-V4 recipe of round 30 ((d, L) grid by inner validation, 5 seeds) with per-segment adjustment on design D2 and the summary branch WITHOUT age/sex (`tokens(..., no_as=True)`), 50 folds. Rank-normal is not applied to NM (the tokens are already close to normal; kurtosis 0.8). NM-D1 without AS is not run.
- **F2-Clean (main)** = mean probability of I2 `rank-noAS-D2`, P3 `rank-noAS-D2`, NM without AS (D2). **F2-Clean-noNM** = mean of I2 and P3 (same variants), which does not contain the NM component that has not passed R1/R2 fully. The D1 and `noAS` variants are reported descriptively (F2-Clean-noNM-D1, F2-Clean-noNM-noRank).
**IPW (comparator):** propensity model P(y=1 | D1) logistic regression on training patients; weight w_i = 1/P(y_i | D1_i), clipped at the 5th and 95th percentile of the training weights, normalized to mean 1; features are NOT transformed. IPW-I2: raw BP (no rank, V4, AS) -> `nested_lr` with sample weights (+AS variant descriptive). IPW-RF: Rohira RF (set A, 48 raw window features, 100 trees, round 33 recipe, trained on 80% of training patients) with patient weights passed to the windows, 50 folds; plus a plasmode leakage test (round 42 recipe: H1/H2a, gamma 0, 0.5, 1, 30 replicates, same seeds, raw-IPW only) with R1 on fold_mean. IPW for NM and Tang is NOT run in this round (GPU cost; postponed, not cancelled).
**30-window sensitivity:** for each patient take min(30, n) first-session windows at random without replacement (seed 44000 + patient index); I2 and P3 `rank-noAS-D2` are recomputed on that window set (the Riemannian reference and the per-patient mean only from the selected windows). Only 2 patients have < 30 windows (minimum 12 windows).
**Spisak test:** `round28_analisis.partial_test` (partial: permute c within each class of Y, 2,000 permutations, seed 44100): age (continuous), era, device, montage (categorical), on the score averaged per patient.
**Analysis within strata:** AUC_k within year x device strata with age caliper k = 5, 10, 15, infinity (`round41` code), patient bootstrap 2,000 (seed 44200); F2-Clean and comparators.

## 3. Hypotheses, test family, decision rules
- **H44-0 (reproduction):** I2 `asis` reproduces `hasil29` V4_AS and P3 `asis` reproduces `hasil30` P3_V5_AS (maximum per-fold AUC difference <= 1e-9). If it fails: stop.
- **H44-1 (age dependence):** F2-Clean "passes" if the partial age r^2 < 0.02 AND p > 0.10 (decision rule; not part of the Holm family). Also reported for F2-Clean-noNM, each view, and IPW. Reference: F2-adj r^2 0.13.
- **Primary family (Holm over 4 tests):** T1: per-fold AUC F2-Clean - F2-adj (two-sided, Nadeau-Bengio); T2: F2-Clean - C1; T3: F2-Clean - metadata (Floor); T4: AUC_10 (year x device, age caliper 10 years) of F2-Clean > 0.5 (bootstrap, two-sided). The claim "F2-Clean retains the signal after removing AS" is made if T4 passes Holm < 0.05 and H44-1 is met; no claim of "equivalence" with F2-adj (T1 is not an equivalence test).
- **Descriptive (no claim, no correction):** I2/P3 variants (effect of rank, effect of age^2, effect of removing AS); F2-Clean-noNM; IPW vs `rank-noAS-D2` (per-fold AUC and age r^2); 30 windows vs all windows; within strata for all comparators; R1 of IPW-RF plasmode.
- **A priori controls:** (K1) H44-0; (K2) label permutation within year x device cells on F2-Clean (200x): mean AUC_k 0.5 +- 0.02; (K3) on IPW-RF plasmode gamma = 0: R1 "no evidence" on H1 and H2a (reference round 42: RF pipeline without adjustment 0.49-0.50).

## 4. Predictions (probabilities, before the run)
- H44-1: F2-Clean passes 0.55 (I2 0.75, P3 0.70, NM without AS 0.50; F2-Clean-noNM 0.65).
- Per-fold AUC of F2-Clean 66-69; F2-Clean - F2-adj between -0.03 and 0: 0.65; T1 passes Holm 0.10; T2, T3 pass Holm 0.05.
- T4 (AUC_10 > 0.5, Holm < 0.05): 0.60 (wide CI; in round 41 F2-adj passed at k = 10).
- The age^2 term lowers the I2 age r^2 further: 0.55; raises the AUC: 0.20.
- IPW-I2 AUC lower than `rank-noAS-D2` and larger age r^2: 0.60. IPW-RF plasmode gamma = 1 H1: R1 "no evidence": 0.90; real-data AUC of IPW-RF 50-56.
- 30 windows: |F2-Clean-noNM(30) - F2-Clean-noNM| <= 0.02: 0.70.

## 5. No-leakage guarantee and integrity
- AS is not used in the final classifier; V4 and rank are fitted on training patients; IPW weights and clipping bounds from training patients; the age^2 column is only a function of age.
- Independent check `round44_cek_independen.py` (written before interpretation): (i) I2/P3 `asis` variants = stored results (1e-9); (ii) second implementation of rank/V4/D2/IPW on 2 folds (1e-8); (iii) test data x5+3 do not change the training features; flipped test labels do not change predictions; (iv) F2-Clean reassembled; per-fold AUC and Nadeau-Bengio p recomputed (1e-9); (v) Spisak test: r^2 statistic recomputed with regression; (vi) AUC_k within strata with a brute-force loop (1e-12); (vii) summary from the raw rows; (viii) permutation control.
- Raw results to npz/CSV before summarizing; resumable; CPU `CUDA_VISIBLE_DEVICES=""` (8 workers); NM on the GPU after the plasmode finishes (one GPU process).

## 6. Limits
n = 200; F2-Clean (components, design D2, rank) was chosen before the run but after 43 rounds on the same patients (researcher degrees of freedom are not zero); the Spisak test examines linear/continuous dependence on age according to the r^2 statistic used, not all forms of dependence; "passes" is not "no leakage at all"; "not significant" is not "equivalent"; NM-D1 without AS and IPW for NM/Tang were not done.
