# ANALYSIS PLAN, ROUND 46 (R1 on the per-fold AUC for C1 and V0--V4; written BEFORE running; 2026-10-27)

Source: item T5 of the revision instructions; the gap noted in Methods 2.6: the leakage table (rounds 28 and 29) applies R1 only to the pooled AUC, whereas the primary metric of the study is the per-fold AUC. CPU only (`CUDA_VISIBLE_DEVICES=""`, 12 workers). No new model: pipeline, seeds, and folds are identical to round 29 (`round29_sensitivitas.py::run_matrix`; H1 features from `round28_analisis.shuffle_cols`, seed 29100 + r).

## Question
Does the R1 decision (no evidence / uncertain / leakage) for C1, V0, V1, V2, V3, V4 on features without information H1 change if the per-fold AUC (mean of 50 folds) is used instead of the pooled AUC?

## Design
- Cells: H1 only (features exchanged between all patients), 100 replicates (r = 0..99), 6 variants without age and sex in the final classifier (C1, V0, V1, V2, V3, V4), 50 canonical folds. There is no H2a or H2b in this round.
- Raw output: AUC on each set of 40 test patients per fold, `hasil46/H1_auc_per_lipatan.npz` (100 x 6 x 50) and `hasil46/H1_per_replikasi.csv` (mean of 50 folds and the pooled AUC of round 29 per replicate). Saved before summarizing.
- Statistics: per variant, the mean over replicates of the mean of 50 folds (m), the SD between replicates, SE = SD / sqrt(100).
- Rule R1 unchanged (Methods 2.6): no evidence if m <= 0.520 and m + 1.645 SE <= 0.540; leakage if m > 0.55; otherwise uncertain. The pooled-AUC decision uses the same rule on the `V*` columns of `hasil29/A_kebocoran_replikasi.csv` (the values of the leakage table are unchanged).
- Integrity check inside the script: the pooled AUC of every replicate and variant must equal `hasil29/A_kebocoran_replikasi.csv` (|difference| < 1e-9); otherwise the run stops.
- Independent check (`round46_cek_independen.py`): the summary is recomputed from the raw output with a second code, and three replicates (r = 0, 17, 53) are retrained sequentially with the second code and compared exactly with the raw output.

## Predictions (subjective probabilities; written before running; only one replicate, r = 0, was run for timing and reproduction of the pooled AUC; per-fold values were neither printed nor saved)
- (P1) C1: no evidence (90%).
- (P2) V0: leakage (85%). (P3) V1 and V2: leakage (70%, both).
- (P4) V3: uncertain (60%). (P5) V4: no evidence (85%).
- (P6) The per-fold decision equals the pooled-AUC decision in at least 5 of 6 variants (85%); in all six (65%).
- (P7) The mean difference (per-fold minus pooled) has an absolute value <= 0.010 in all variants (90%).

## Cost estimate
One replicate 5.3 s (6 variants x 50 folds; wall time on 12 workers); 100 replicates about 9 minutes; no GPU. Below the 2-hour threshold.

## What will be reported even if negative
All m, SE, per-fold and pooled decisions, and predictions that missed. If a decision differs from the pooled AUC, the difference is reported in Results and in the caption of the leakage table. The numbers of the old table are not changed. Deviations are recorded as dated addenda in this file.
