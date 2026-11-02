# ANALYSIS PLAN, ROUND 47 (AUC of F2 without NM; written BEFORE computing; 2026-10-27)

Source: the simulated-review report of round 2 (2026-10-27), concern P2-R2.M1: F2 includes NM (a nonlinear classifier) that has not fully passed the leakage test (NM-V4: per-fold R1 "uncertain"; NM without age/sex does not pass the age rule), so the F2 AUC of 69.7 cannot yet be called leak-free; only the linear views (I2 with V4, P3 with V5) were shown to be clean. Question: how much do the per-fold AUC and the differences from the baselines change when NM is removed from the fusion?

## Design
- Data: stored out-of-fold predictions `hasil30/eval_prediksi_v4.npz` (10 repetitions x 200 patients; keys `I2_adj`, `P3_adj`, `NM_adj`, `F2_adj`, `C1`, `Floor`). No new training; CPU; no GPU.
- The "F2 without NM" score = arithmetic mean of the `I2_adj` and `P3_adj` probabilities (the F2 fusion rule with no learned parameters, without NM). Shape check (run before the hash and displaying no AUC): `F2_adj` equals the mean of the three views to 1e-9.
- Statistics: per-fold AUC on the 50 canonical folds (`round28_analisis.J`, `round17_common.fold_aucs`), mean; Nadeau--Bengio CI (n1 = 160, n2 = 40, J = 50); pooled AUC (mean of 10 repetitions). Per-fold differences against `F2_adj` (full F2), `C1`, `Floor` (acquisition-metadata baseline), and `I2_adj`, with the Nadeau--Bengio corrected t-test (function `round29_sensitivitas.nb_diff`).
- Primary family (Holm over three tests): F2 without NM against F2, C1, and the metadata baseline. The difference from I2 is descriptive.
- Raw output: `hasil47/auc_per_lipatan.npz` (50 per-fold AUCs for F2 without NM, F2, C1, metadata, I2, P3, and the pooled AUC), saved before summarizing (`round47_ringkas.py`).
- Independent check `round47_cek_independen.py`: per-fold AUCs recomputed with a self-written Mann--Whitney rank from the stored predictions; the summary and NB statistics recomputed with a second code.

## Predictions (subjective probabilities; written before computing; known beforehand only the per-fold AUC of I2 68.0, P3 66.6, F2 69.7, F2-Clean without NM 66.8 from other analyses)
- (P1) Per-fold AUC of F2 without NM >= 67.0 (70%).
- (P2) |difference from F2| <= 0.030 (85%).
- (P3) Not significantly different from the metadata baseline and C1 (p > 0.05 before Holm) in all three primary tests (90%).
- (P4) Per-fold AUC of F2 without NM higher than F2-Clean without NM (66.8) (60%).

## What will be reported even if negative
All AUCs, CIs, differences, p values, and Holm; predictions that missed; if F2 without NM differs significantly from the baseline, the Conclusion of the manuscript is changed accordingly and reported to the owner. Deviations are recorded as dated addenda in this file.
