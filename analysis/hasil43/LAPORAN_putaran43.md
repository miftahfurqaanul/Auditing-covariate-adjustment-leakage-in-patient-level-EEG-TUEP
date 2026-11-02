# Report of round 43 (Priority Experiment 5, 2026-10-23): equalizing foundation-model pipelines, anatomical compression, selection-optimism correction

Plan `PRAREGISTRASI_putaran43.md` (hash in `PRAREGISTRASI_hash_putaran43.txt`). Scripts: `round43_fm_kompresi.py` (A-B-C), `round43_optimisme.py` (D), `round43_ringkas.py` (`hasil43/ringkas.log`, `ringkas_*.csv`), `round43_cek_independen.py` (9 groups passed; `cek_independen.log`). CPU only. Equal pipeline: [standardization + PCA 50] -> rank-normal -> V4 -> [+ age, sex] -> L2 LR with C from 6 values by 5-fold inner CV, all steps refitted in every inner fold; 50 canonical folds.

## Parts A-B (per-fold AUC, +AS)
| Model | Per-fold AUC [95% CI] | Difference vs I2-adj (68.0) | p | Holm |
|---|---|---|---|---|
| LaBraM frozen, equalized | 63.6 [54.2, 73.0] | -0.044 [-0.138, +0.049] | 0.35 | 0.69 |
| CBraMod frozen (layer 12), equalized | 66.2 [58.7, 73.6] | -0.019 [-0.103, +0.066] | 0.66 | 0.69 |
| BENDR frozen, equalized | 53.2 [44.9, 61.4] | **-0.149 [-0.255, -0.043]** | 0.007 | **0.028** |
| Comp50 (anatomical compression, 50 features) | 58.8 [50.8, 66.9] | -0.092 [-0.196, +0.012] | 0.082 | 0.25 |
Descriptive (no correction): BP-rank (114 features, rank-normal + V4 + C-6, without PCA) 68.6 (+0.005 vs I2-adj; p 0.85): equalizing with rank-normal + 6 C values does not change I2; BP-eq (PCA 50 on 114 features) 65.9 (-0.022; p 0.44): PCA50 alone lowers by about 0.03; Comp25 (median only) 61.8; CBraMod layer 4 / 8 / 12: 61.2 / 66.0 / 66.2; without age-sex: LaBraM 61.5, CBraMod 63.6, BENDR 48.8, Comp50 54.2, BP-rank 66.2. Against BP-eq (the same PCA50 pipeline, descriptive): CBraMod +0.003, LaBraM -0.023, BENDR -0.127 (without p and correction).
**Interpretation:** (1) under the same pipeline, frozen CBraMod is equivalent to band power (BP-eq) and LaBraM slightly below it, but both do not differ significantly from I2-adj at n = 200 (wide CI); (2) frozen BENDR is significantly lower (Holm 0.028) and worse than in round 31 (60.3 -> 53.2), so the BENDR gap is not caused by unequal pipelines; (3) the 50-feature anatomical compression does not raise the AUC and in fact lowers it (58.8; a larger EPV does not help): channel-specific information in the 114 features is lost when averaged per region; (4) there is no claim of "equivalence": the results are not significant.

## Part C (R1, pipeline without age/sex; permuted features; real label; 20 replicates)
All 10 cells (BP-eq, LaBraM-eq, CBraMod-eq, BENDR-eq, Comp50 x H1/H2a) have the R1 decision "no evidence of leakage" (mean per-fold AUC 0.492-0.512; SE 0.007-0.015).

## Part D (bootstrap 2,000; AUC = mean over 10 repetitions of the pooled AUC)
| Set | K | Winner | Full AUC | Mean optimism (SD) | Corrected AUC | Same bootstrap winner |
|---|---|---|---|---|---|---|
| main | 15 | F2-adj | 0.6939 | 0.0196 (0.069) | **0.6742** | 75% |
| all | 50 | F2-adj | 0.6939 | 0.0381 (0.070) | **0.6557** | 51% |
The fixed-model bias (in-bag AUC - OOB for one candidate without selection) averages -0.0008 (main) and -0.0003 (all): the optimism above comes from selection, not from the procedure.
Studentized max-T test (Reality Check): metadata reference: best d +0.046 (F2-adj), p 0.44 (main), 0.58 (all), Holm 0.72 / 1.0; C1 reference: d +0.028, p 0.36 (main), 0.56 (all), Holm 0.72 / 1.0. **After accounting for selection, there is no evidence that the best candidate exceeds the metadata baseline or C1.**
Calibration of the procedure (labels permuted, 40x): winner AUC 0.547, optimism 0.058, corrected AUC 0.489, max-T p < 0.05 in 5.0% (as nominal).

## Predictions vs results
| Prediction | Chance | Result |
|---|---|---|
| LaBraM-eq 0.62-0.66 | - | 63.6 (within the range) |
| CBraMod-eq 0.60-0.64 | - | 66.2 (above the range) |
| BENDR-eq 0.55-0.60 | - | 53.2 (below the range) |
| Comp50 0.66-0.70 | - | 58.8 (**missed by a wide margin**) |
| BENDR-eq negative and passes Holm | 0.35 | occurred (Holm 0.028) |
| Comp50 positive and passes Holm | 0.05 | did not (correct) |
| Comp50 within +-0.03 of I2-adj | 0.60 | **no** (-0.092) |
| Equalization narrows the BENDR and LaBraM gap vs round 31 | 0.55 | **partly**: LaBraM -0.055 -> -0.044; BENDR -0.077 -> -0.149 (widened) |
| All R1 of part C "no evidence" | 0.70 | occurred (10/10) |
| Optimism of the S_utama winner 0.015-0.035 | 0.65 | 0.0196 (occurred) |
| F2-adj is the winner of S_utama | 0.40 | occurred (also S_semua) |
| Reality Check vs metadata p > 0.05 in S_utama | 0.65 | occurred (0.44) |
| Reality Check vs C1 p > 0.05 | 0.90 | occurred (0.36) |
| p of S_semua larger than S_utama | 0.90 | occurred (0.58 vs 0.44; 0.56 vs 0.36) |

## Limits
Frozen embeddings come from the window pipeline of the project (mean of the first-session windows, not the per-recording EEG-Bench recipe); PCA50 alone lowers BP by about 0.03, so the comparison of foundation models is read against BP-eq (the same pipeline), not only against I2-adj; CBraMod layer 12 was chosen a priori, other layers are descriptive. The optimism correction applies only to selection among candidates whose predictions are stored (15 or 50); it does not cover the design degrees of freedom (features, architecture, 42 rounds) and is not a substitute for a hidden test set. "Not significant" is not "equivalent"; n = 200.
