# ANALYSIS PLAN, ROUND 50 (response to the second output of the "Q1 Journal Reviewer Assistant", STRICT AUDIT mode; written BEFORE computing; 2026-11-02)

Source: the output of an automated reviewer tool (file of 2026-11-02 23:22; on the 25-page `main_en.pdf`). Its three "fatal issues": (1) the decision rules are inconsistent for high-variance pipelines, without empirical validation; (2) missing numbers (windows per patient in the simulation; AUC differences of some configurations); (3) no validation on another dataset. Analysis status: **follow-up** (after all results of rounds 3--49 were known). No part is confirmatory with respect to the earlier exploration.

What HAD been seen before this document was written: (i) all round-42 plasmode results (30 or 20 replicates) and the 49B decision frequencies, including R2 "leakage" in 31.5--48.0% for the raw-feature tree cells at gamma = 0; (ii) the round-39 summary for the unequal-window cells (`hasil39/ringkasan_B.csv`, imb = True), read on 2026-11-02 while checking the manuscript; (iii) the structure of the Emotiv data (Zenodo 1252141): 308 subjects with clean "whole" 10 s windows (1--27 per subject; 299 subjects with >= 5 windows), the covariate columns (country, eyes condition first, start time). The label--covariate relation in Emotiv, the covariate AUC, and all Emotiv plasmode AUCs had NOT been computed. TUEP plasmode replicates 30+ (RF) and 20+ (XGB) had NOT been computed.

## 50A. Extension of the TUEP plasmode replicates (fatal issue 1)

Question: are the R2 false alarms for trees on raw features at gamma = 0 (where leakage is impossible) a consequence of the small number of replicates, as the normal model of 49B predicts (SD 7.8 pp: false alarms 4.2% with 20 replicates, < 1% with 50)?

Design: the round-42 plasmode code is NOT changed (`round42_cpu.unit_window`, same seeds `kseed(rep, ...)`); only the replicates are extended: RF replicates 30--119 and XGB replicates 20--119, condition H1, gamma in {0, 1}; 120 replicates per cell in total. Output to `hasil50/A_RF.csv`, `hasil50/A_XGB.csv` (hasil42 is not touched). Reproduction check: replicate 0 (H1, gamma 0) is recomputed and must be identical to `hasil42` (maximum AUC difference < 1e-9 for RF; XGB on GPU tolerance 1e-6, any difference is recorded).

Analysis (per fold, as in 49B): for the four null cells (RF/XGB x raw-win/raw-pat, gamma 0) and the two positive cells (RF/XGB raw-pat, gamma 1): mean delta (with minus without adjustment) over 120 replicates with a 95% t CI; frequencies of the R2 decision (and R1 for the adjusted arm) with 20, 50, and 100 replicates from 10,000 bootstrap draws from the pool of 120 (function `round49_kalibrasi.freq`, original thresholds), Monte Carlo CI from 500 resamples of the pool.

Predictions (subjective probability):
- P50A-1: the mean delta of the four null cells over 120 replicates has |mean| <= 1.0 pp and a 95% CI that includes 0 (70%). If not, it is reported as a small unexplained systematic bias, not as noise.
- P50A-2: the R2 "leakage" frequency with 50 replicates is <= 5% in all four null cells (65%); with 20 replicates it is below 31.5% (the lowest 49B value) in at least three of the four cells (60%).
- P50A-3: positive cells (raw-pat gamma 1) R2 "leakage" >= 95% with 20 replicates for RF and XGB (85%).

## 50B. Plasmode on the independent Emotiv cohort (fatal issue 3)

Question: does covariate adjustment also create spurious discrimination on real EEG features from another archive (consumer device with 14 channels, two countries, different acquisition covariates), and does rank-normal reduce it? This tests the size of the effect and the mitigation on other data, not diagnostic performance or clinical value.

Data: Zenodo 1252141 (Guinea-Bissau and Nigeria), existing preprocessing (`tuep_fresh/code/external_prep.py`: 10 shared channels, 10-channel CAR, FIR 0.5--45 Hz, 200 Hz, "whole" 10 s windows, rejection by PTP and head motion). Inclusion (set now): subjects with >= 5 clean windows (299). Covariates (design D_E, 5 columns): [1, country (Nigeria = 1), eyes condition first closed (1/0), recording day within the country (days since that country's first recording, z), start hour (decimal hour, z)]; z computed over all included subjects (no labels; covariates known for all). The real label (epilepsy/control) is used only for eta.

Synthetic labels: as in round 42: eta = linear predictor of an unpenalized logistic regression of the real label on D_E (centred); y_sim = 1 for the floor(N/2) subjects with the highest gamma*eta + e (e ~ standard logistic, the same for every gamma within a replicate), gamma in {0, 0.5, 1}. Features are swapped between subjects as window blocks (H1). The covariate AUC (eta -> real label) is reported.

Features and pipelines (parallel to round 42):
- Subject level (LR): log relative power in 5 bands + log total power per channel (60 features; `external_prep.bandpower`, stored in `ext_zenodo_whole_bp.npz`), median over windows per subject; pipelines raw/rank x none/pat; `evalkit.nested_lr`; 30 replicates.
- Window level (RF, XGB): raw band power (sum of the Welch PSD, Hann, nperseg 400, noverlap 200 at 200 Hz) in the six Rohira bands (1--4, 4--7.5, 7.5--13, 13--16, 16--30, 30--40 Hz) x 10 channels = 60 features, computed from `ext_zenodo_whole_windows.npz`; pipelines raw-none, raw-win, raw-pat, symlog-pat, rank-pat, log-pat; RF with 100 trees (as in round 42; 80% of the training subjects used for fitting, `split_val`), XGB (the same, GPU); subject score = mean window probability; RF 30 replicates, XGB 20.
- Folds: one StratifiedKFold(5) per replicate and gamma; main metric the per-fold AUC (mean of 5 folds).
- The rank transform and the adjustment are fitted on the training subjects only (round-42 functions generalized to D_E and N_E).

Decision rules: R1 and R2 with the original thresholds (unchanged). Control: at gamma = 0, at least 95% of the adjusted cells do not have the R1 decision "leakage".

Primary family (Holm, two tests; two-sided one-sample t over the per-replicate differences at gamma = 1): (i) RF raw-pat minus raw-none; (ii) XGB raw-pat minus raw-none. Secondary (Holm, two tests): raw-pat minus rank-pat for RF and XGB at gamma = 1. Descriptive: LR, symlog, log, win, gamma 0.5.

Predictions:
- P50B-1: RF and XGB raw-pat at gamma = 1 have the R2 decision "leakage" (delta > 3 pp) and the primary family Holm < 0.05 (65%; lower than TUEP because the Emotiv features have 10 channels and fewer windows per subject, median 21).
- P50B-2: rank-normal lowers the RF and XGB delta by at least 3 pp compared with raw-pat (60%).
- P50B-3: subject-level LR |delta| <= 1.0 pp or R2 "no evidence"/"uncertain" in all pipelines (80%).
- P50B-4: the gamma = 0 control passes (85%).

## 50C. Reporting completeness (descriptive, from stored results; no new model computation)

(i) The round-39 simulation grid is written out explicitly in the Methods (windows per patient 50 in the base cell, 10 or 200 in the one-factor variations; one unequal variation m_p = round(50 exp(0.6 z)) truncated to 12--200; patients 100/200/500; features 10/50/200; replicates). (ii) The results of the round-39 unequal cells (window-weighted vs per-patient adjustment) are reported; the manuscript sentence "the effect of unequal window counts was not tested" (true only for 49D) is corrected. (iii) The Monte Carlo error of the plasmode differences (SE of delta) is reported as a range in the plasmode table caption.

## Notes

Negative results are reported. If 50B does not replicate the leakage, the manuscript states that the size of the effect depends on the archive and the features. Round naming: this analysis becomes round 50 (the sixth follow-up); the round count in the manuscript is updated. Limitations that remain: no independent clinical cohort with comparable labels and complete acquisition metadata; no medication data; no prospective evaluation.

## Addendum 1 (written AFTER the 50A and 50B outputs were seen; 2026-11-03)
(i) Print correction: two print lines of `round50_ringkas.py` (P50B-1, P50A-3) used `DataFrame.pipe` (a pandas method) as a column and therefore printed empty lists; replaced by `['pipe']`. The computations and the CSV files did not change; the corrected script was hashed again (see the hash log). (ii) Descriptive analysis outside the plan: all RF/XGB pipelines over 120 replicates (`hasil50/deskriptif_120_semua_pipeline.py`), reported as descriptive. (iii) Prediction outcomes: P50A-1 missed (the CIs of all four null cells include 0, but the RF means are +1.05 and +1.09 pp > 1.0); P50A-2, P50A-3, P50B-1 to P50B-4 met. (iv) Independent checks (`round50_cek_independen.py`, written after the 50B output and before the 50A output): seven groups passed.
