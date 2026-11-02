# Code audit, round 30: train-test leakage (2026-10-21)

Scope: all paths that produce numbers in the manuscript. Two parts: (A) dynamic tests (run; results in `hasil30/audit_v2.log`, `audit_v2_hasil.csv`) and (B) static review of the comparator code (read; evidence as file:line). The static review does not replace the dynamic tests; it only shows that no path was found that feeds test data or labels into fitting.

## A. Dynamic tests (`round30_audit_v2.py`): 24 of 24 passed

| Group | Check | Result |
|---|---|---|
| I2-V4 (folds 0, 17, 33) | changing test data does not change training features; flipping test labels does not change test predictions | passed |
| P3-V5 (0, 17, 33) | changing test covariances does not change training features; flipping test labels does not change predictions | passed |
| NM-V4 (0, 17, 33) | changing test segments does not change training tokens/summary; flipping test labels does not change anyone's tokens; the coefficients come only from training-patient segments | passed |
| Orthogonality | I2-V4: training residuals orthogonal to the design (max coefficient 5.4e-15). I2-V0: training controls orthogonal (3.9e-15) but training epilepsy not (1.39) | passed (explains the V0 asymmetry) |
| Global label permutation | I2-V4 (100): mean AUC 0.4927 (SE 0.0042), 95th percentile 0.552. P3-V5 (100): 0.5021 (SE 0.0041), 95th percentile 0.569 | passed; the 95th percentile matches the null expectation (SD between permutations about 0.041) |
| existing `audit_kebocoran.py` | rerun | 16 PASS, 0 FAIL |

Independent double check: `round30_cek_independen.py` (8 groups passed; second implementation of NM-V4 tokens and P3-V5 features, integrity of 50 NM JSON files, reconstruction of F2 and the tests, brute-force AUC within strata, `mlconfound`, null summary).

## B. Static review of the comparator code

| Path | Evidence (file:line) | Finding |
|---|---|---|
| Canonical fold splitter | `audit_kebocoran.py:29-30` (each patient exactly once in test; 160/40; stratified) | no patient on both sides |
| Logistic models (C1, Floor, Demo, final I2/P3-V4/V5) | `evalkit.py:19-22` (`StandardScaler` inside the pipeline; C chosen by inner CV on training data) | scaling and C do not see the test data |
| Metadata baseline | `baseline_clean.py:39-43` (`lr_fp`: `fit(Xtr, ytr)` then predict `Xte`) | clean |
| Rohira-F (RF) | `rohira_fixed.py:30-36` (`split_val` from `tr`, model fitted on `tr_p`, threshold from `va_p`), `:39` (baseline fitted on `tr`) | validation taken from training data; the threshold does not use test data |
| Validation splitter | `tang_fixed.py:90-94` (stratified permutation inside `tr_p`) | clean |
| LaBraM-FT / CBraMod-FT | `round25_fm.py:58` (`GroupShuffleSplit` per patient on training recordings), `:196` (number of training/val/test windows), `:206` (`val_auc`, `best_epoch`, `val_threshold` recorded from validation), `:227` (check: training/val/test separated per patient) | epoch and threshold selection use validation inside training |
| FM evaluation | `cbramod_ft_eval.py:43` (test predictions per patient from the fold JSON; threshold from `val_threshold`) | the threshold is not from test |
| Early P3, I2 V0 | see part A and `PRAREGISTRASI_putaran30.md` | procedurally clean (does not use test data) but LEAKS covariates through the control vs epilepsy asymmetry; replaced by V4/V5 |

## C. Cannot be removed (must be stated in the manuscript as limitations)

1. LaBraM/CBraMod/BENDR pretraining may include the TUH corpus; overlap with TUEP patients cannot be determined. This bias favors the foundation models (conservative for the F2 vs FM comparison).
2. The 27 earlier rounds of analysis used the same 200 patients and 50 folds; the choice of variants V1-V5 was guided by null tests (not AUC), but the real-data AUC of I2 has been seen since round 29, so there is an unmeasured researcher degree of freedom.
3. Nested F2 (72.4) and DNM-M were not recomputed (Addendum 1); their numbers are only from the old pipeline.
4. The reduced NM null (20 replicates, H1, without age/sex; mean 0.515, mean 95th percentile 0.531): the conclusion is "no evidence of leakage", not "shown to be clean".
5. The NM-V4 configuration (d, L) is chosen per fold by validation inside training (as in `normmamba.run_fold`): [32,32] 19x, [64,32] 14x, [64,96] 11x, [32,96] 6x.

Limitation of the audit: the static review does not cover all files of the 100+ scripts in `tuep_fresh/code/`; only the paths that produce manuscript numbers (above). Scripts of earlier rounds (round7-27) not reviewed again here rely on `audit_kebocoran.py` and the earlier `verify_all.py` verification.
