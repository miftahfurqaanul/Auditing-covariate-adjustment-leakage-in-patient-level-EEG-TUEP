# ANALYSIS PLAN, ROUND 48 (dynamic leakage audit on 50 folds; written BEFORE the full run; 2026-10-27)

Source: the simulated-review report of round 3 (2026-10-27), concern P3-R2.M4: the manuscript states that the automated leakage checks passed "on three of the 50 folds (0, 17, 33)"; the reviewer asked about the results on the other folds. CPU only (`CUDA_VISIBLE_DEVICES=""`), no GPU, no new model training beyond what the audit code does.

## Design
- Code: part A of `round30_audit_v2.py` (perturbation test) copied without change of the checks to `round48_audit_lipatan.py`; only the fold list is extended to all 50 canonical folds (`round28_analisis.J`). Six checks per fold: (1) I2-V4: changing the data of test patients changes neither the features of training patients nor the training indices; (2) I2-V4: flipping the labels of test patients does not change the test predictions; (3) P3-V5: changing the covariances of test patients does not change the features of training patients; (4) P3-V5: flipping the labels of test patients does not change the test predictions; (5) NM-V4: changing the segments of test patients does not change the tokens and summary of training patients; (6) NM-V4: flipping the labels of test patients does not change anyone's tokens. In total 300 checks. The other audit parts (global label permutation, residual orthogonality, the old `audit_kebocoran.py`) are not repeated; the round 30 results stand.
- Pass criterion: all 300 checks pass. Every failure is reported as it is and the Methods sentence is not strengthened beyond the result.
- Raw output: `hasil48/audit_lipatan_0_49.csv` (one row per check: fold, ok), saved before summarizing.
- Independent check `round48_cek_independen.py`: (A) the output file is complete (300 rows, 6 per fold, folds 0--49); (B) invariance of the V4 features of training patients to changes in test data is reimplemented with plain numpy (own least squares) on folds 3, 22, and 41; (C) the checks on folds 0, 17, 33 match `hasil30/audit_v2_hasil.csv`.

## Cost estimate
One fold takes 11 s (sequential, CPU); 50 folds take about 9 minutes; below the 2-hour threshold. Fold 0 was already run once for timing (`--timing`): 6 of 6 passed (and folds 0, 17, 33 already passed in round 30).

## Predictions (subjective probabilities; written before the full run)
- (P1) All 300 checks pass (90%).
- (P2) If there are failures, they are limited to NM-V4 (checks 5--6) (70% conditional).

## What will be reported even if negative
The number of passes/failures per path and fold; if there are failures, the Methods sentence is changed to follow the result and the failures are reported to the owner. Deviations are recorded as dated addenda.
