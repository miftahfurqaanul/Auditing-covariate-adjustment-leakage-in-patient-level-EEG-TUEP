# ANALYSIS PLAN, ROUND 41 (Priority experiment 4): AUC within year x device strata with an age caliper

Date: 2026-10-23. Written and hashed BEFORE any within-strata AUC with an age caliper was computed. Seen before writing: (i) the AUC within year x device strata WITHOUT caliper (round 30; F2-adj 0.7695, I2-adj 0.7695, P3-adj 0.7380, NM-adj 0.7143, C1 0.7293, metadata 0.4317, LaBraM-FT 0.6401, CBraMod-FT 0.6669, Demo 0.6314; reproduced exactly from `hasil30/eval_prediksi_v4.npz`); (ii) the number of effective pairs per caliper (design structure, without predictions), in section 1. Hash: `PRAREGISTRASI_hash_putaran41.txt`. Changes only through dated addenda.

## 1. Aim and design
Question: does the advantage of the EEG model over the metadata baseline within year x device strata (F2-adj 77.0 vs 43.2) survive when age is controlled simultaneously? Note: the F2-adj model includes age and sex as inputs, and epilepsy patients are younger (median 52 vs 59); the "Demo" comparator (age + sex) gives 63.1 within strata without caliper.

- **Data:** the existing out-of-fold predictions (`hasil30/eval_prediksi_v4.npz`, 10 repetitions x 200 patients, 50 canonical folds; no model is retrained); age from `round4.AGE`; strata from `round9.STRATA`.
- **Valid pairs:** (epilepsy patient i, non-epilepsy patient j) with the same year x device cell AND |age_i - age_j| <= k, k in {5, 10, 15, infinity} years. AUC_k = (number of pairs with score_i > score_j + 0.5 x ties) / number of valid pairs. Score = mean over 10 repetitions (same as `round10.within`); secondary reporting: mean AUC per repetition.
- **Effective pairs (design structure; from the data, without predictions):** year x device: k=5: 197 pairs (2.0% of 10,000), 141 patients covered; k=10: 374 (3.7%), 161; k=15: 532 (5.3%), 173; k=infinity: 1,267 (12.7%), 196. Secondary, "year" strata: k=5: 331 (3.3%), 191; k=10: 608 (6.1%), 195; k=15: 855 (8.6%), 197; infinity: 1,902 (19.0%), 200.
- **Minimum-pairs rule (a priori):** if valid pairs < 150 for a (strata, k), the result is reported as "cannot be estimated" and does not enter the test family. (All values above are >= 197, so the rule is not active; it is still recorded.)
- **Bootstrap:** 2,000 replicates, resampling patients WITHIN the year x device cell (as in round 30); the same bootstrap sample is used for all models and all k (paired differences). Pairs are rebuilt from the bootstrap sample (including the age caliper). 95% percentile CI; two-sided p = 2 x min(proportion <= null value, proportion >= null value), minimum 1/2000. Seed 41000.

## 2. Hypotheses, test family, decision rules
**Primary family (Holm, 6 tests):**
- A_k (k = 5, 10, 15): AUC_k of F2-adj > 0.5 (null value 0.5).
- B_k (k = 5, 10, 15): AUC_k(F2-adj) - AUC_k(metadata) > 0.
**Claim decision:** "the EEG advantage over metadata survives age control" if A_10 and B_10 both pass Holm (p_Holm < 0.05) AND the sanity checks below are met. If only A_15/B_15 pass: "survives at a loose caliper". If A_10 does not pass: the claim is limited to year x device strata without age control; dependence on age cannot be excluded. k = 10 is fixed a priori as the main k; all k are reported.
**Descriptive (no claim, no correction):** all models (I2-adj, P3-adj, NM-adj, C1, metadata, LaBraM-FT, CBraMod-FT, Demo) at all k and both strata; differences F2-adj - {C1, I2-adj, Demo, LaBraM-FT, CBraMod-FT}; k = infinity; "year" strata; mean AUC per repetition; number of pairs and patients covered.
**Sanity checks (a priori):** (1) k = infinity reproduces round 30 (F2-adj 0.769534; metadata 0.431728; max difference 1e-12). (2) Demo at k = 5 lies in [0.40, 0.60] (if not, the caliper does not remove age information sufficiently and the claim above is not made). (3) Label permutation within year x device cells (200 times, seed 41001), mean AUC_k of F2-adj within 0.5 +- 0.02 for all k.

## 3. Predictions (probabilities, before the run)
- A_15 passes Holm 0.70; A_10 0.60; A_5 0.35 (only 197 pairs).
- B_10 passes Holm 0.65 (metadata within strata < 0.5, so B_k is statistically almost identical to A_k).
- AUC_k of F2-adj decreases monotonically from k = infinity to k = 5: 0.55 (large bootstrap error, SE roughly 0.04-0.08).
- Demo(k=5) in [0.40, 0.60]: 0.85.
- Claim "survives age control" (A_10 and B_10 pass): 0.55.
- The failure scenario I consider most likely: the point estimate falls to 0.60-0.68 with a wide CI that includes 0.5 at k = 5, 10.

## 4. No-leakage guarantee and integrity
- There is no training; age only defines the evaluation pairs, not a new model input. The predictions are already out-of-fold (160 training / 40 test; all fitting only on training patients).
- Independent check `round41_cek_independen.py`: AUC_k and number of pairs with a brute-force loop (1e-12) for F2-adj and metadata at all k and both strata; k = infinity = round 30; the first bootstrap recomputed with a second implementation (loop); the summary recomputed from the raw bootstrap (1e-9); permutation control.
- The raw results (bootstrap matrix) are saved to `hasil41/` BEFORE summarizing.

## 5. Limits
Pairs share patients (dependence) -- handled by the patient-level bootstrap but the CI at k = 5 is very wide; the primary family is small (6 tests) but the strata, k, and hypotheses were chosen after round 30 (researcher degrees of freedom); the predictions come from a pipeline already chosen after 39 rounds; only 200 patients; the age caliper controls age, not all unobserved confounders (for example medication). "Not significant" is not "equivalent". No claim of clinical usefulness.
