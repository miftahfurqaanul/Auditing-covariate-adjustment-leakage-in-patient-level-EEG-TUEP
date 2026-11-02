# Report of round 41 (Priority Experiment 4, 2026-10-23): AUC within year x device strata with an age caliper

Plan `PRAREGISTRASI_putaran41.md` (hash in `PRAREGISTRASI_hash_putaran41.txt`). Scripts: `round41_strata_kaliper.py` (raw results `hasil41/bruto_*.npz`), `round41_ringkas.py` (`hasil41/ringkas.log`, `ringkas_*.csv`), `round41_cek_independen.py` (8 groups passed; `cek_independen.log`). No model was trained; round 30 out-of-fold predictions. Patient-level bootstrap of 2,000 within cells.

## Effective pairs (year x device)
k = 5: 197 pairs (2.0% of 10,000; 141 patients); k = 10: 374 (3.7%; 161); k = 15: 532 (5.3%; 173); infinity: 1,267 (12.7%; 196).

## Primary family (Holm over 6 tests; all passed)
| Test | Value [95% CI] | p | p Holm |
|---|---|---|---|
| A_5: AUC of F2-adj | 0.726 [0.595, 0.843] | 0.0010 | 0.004 |
| A_10 | 0.730 [0.615, 0.829] | 0.0005 | 0.003 |
| A_15 | 0.720 [0.612, 0.817] | 0.0010 | 0.004 |
| B_5: F2-adj - metadata | +0.284 [+0.067, +0.491] | 0.018 | 0.018 |
| B_10 | +0.283 [+0.101, +0.451] | 0.0020 | 0.004 |
| B_15 | +0.267 [+0.098, +0.420] | 0.0005 | 0.003 |
Sanity checks: (1) k = infinity reproduces round 30 (F2 0.769534, metadata 0.431728); (2) Demo (age + sex) at k = 5: 0.472 [0.324, 0.617] (within [0.40, 0.60]), k = 10: 0.511, k = 15: 0.509 (from 0.631 without caliper): the caliper removes the age information; (3) label permutation within cells: mean AUC_k of F2 0.498/0.498/0.497/0.495.

## AUC within year x device strata (point estimates)
| Model | k=5 | k=10 | k=15 | infinity |
|---|---|---|---|---|
| F2-adj | 0.726 | 0.730 | 0.720 | 0.770 |
| I2-adj | 0.706 | 0.727 | 0.718 | 0.770 |
| P3-adj | 0.685 | 0.687 | 0.669 | 0.738 |
| NM-adj | 0.675 | 0.693 | 0.690 | 0.714 |
| C1 | 0.655 | 0.682 | 0.662 | 0.729 |
| LaBraM-FT | 0.553 | 0.591 | 0.571 | 0.640 |
| CBraMod-FT | 0.599 | 0.615 | 0.607 | 0.667 |
| Demo | 0.472 | 0.511 | 0.509 | 0.631 |
| metadata | 0.442 | 0.447 | 0.453 | 0.432 |
Descriptive difference F2-adj - C1: +0.071 / +0.048 / +0.058 / +0.040 (all CIs include 0); F2-adj - I2-adj: +0.020 / +0.003 / +0.002 / 0.000 (CI includes 0); F2-adj - LaBraM-FT: +0.173 / +0.139 / +0.149 / +0.129 (CI does not include 0; not in the primary family); F2-adj - CBraMod-FT: +0.127 [-0.005, 0.255] / +0.115 [-0.003, 0.228] / +0.113 [0.002, 0.222] / +0.103.
"Year" strata (secondary, descriptive): F2-adj 0.683 / 0.666 / 0.666 / 0.735; metadata 0.559 / 0.559 / 0.551 / 0.516; Demo 0.444 / 0.497 / 0.505 / 0.610.

## Predictions vs results
| Prediction | Chance | Result |
|---|---|---|
| A_15 passes Holm | 0.70 | passed (correct) |
| A_10 passes | 0.60 | passed (correct) |
| A_5 passes | 0.35 | **passed (the prediction was too pessimistic)** |
| B_10 passes | 0.65 | passed (correct) |
| AUC_k of F2 falls monotonically from infinity to k = 5 | 0.55 | **not monotone** (0.770 -> 0.720 -> 0.730 -> 0.726; k=5, 10, 15 do not differ meaningfully) |
| Demo(k=5) within [0.40, 0.60] | 0.85 | correct (0.472) |
| Claim survives (A_10 and B_10 pass) | 0.55 | correct |

## Honest interpretation and its limits
1. The advantage of the EEG over the metadata baseline within year x device strata **survives simultaneous age control** (caliper 5/10/15 years): AUC of F2-adj 0.72-0.73, Holm 0.003-0.004 (A_k). The AUC falls by about 0.04 from k = infinity, within bootstrap noise.
2. **B_k exaggerates**: the metadata within strata is below 0.5 (0.44-0.45; the CI includes 0.5), so the difference of +0.28 is mostly the AUC of F2-adj - 0.5 plus a small difference below 0.5. The more honest statement is A_k (AUC above 0.5), not "ahead by 0.28 of metadata".
3. **The signal is not specific to F2**: I2-adj 0.727 (k=10), C1 (without adjustment) 0.682 [0.559, 0.801], both above 0.5; F2-adj - C1 and F2-adj - I2-adj do not differ (CI includes 0). This result supports "there is an EEG signal beyond age, year, device", not "F2 is better".
4. The CI is wide (k=5: 0.595-0.843; 197 pairs, 141 patients); there is no claim of "equivalence" between models; "not significant" is not "equivalent".
5. The age caliper controls age only; unobserved confounders (medication, referral indication, the epilepsy label derived from medication/clinical history) are uncontrolled. The strata, k, and the test family were chosen after round 30 (researcher degrees of freedom); the predictions come from a pipeline chosen after 39 rounds on the same 200 patients.
6. F2-adj contains NM-adj with a possible small leak (round 40: +0.019); I2-adj (clean according to R1/R2) gives an equivalent result within strata, so the conclusion of point 1 does not depend on NM.
