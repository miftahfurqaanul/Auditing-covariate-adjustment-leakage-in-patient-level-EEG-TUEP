# Report of round 40 (2026-10-23): correcting the R1/R2 metric to the mean per-fold AUC and diagnostic test of the mechanism

Plan: `PRAREGISTRASI_putaran40.md` + addenda 1-3 (hash in `PRAREGISTRASI_hash_putaran40.txt`). Scripts: `round40_simulasi_perlipatan.py`, `round40_nm_perlipatan.py`, `round40_rohira_perlipatan.py`, `round40_ringkas.py`, `round40_cek_independen.py` (7 check groups passed; `hasil40/cek_independen.log`). Raw data: `hasil40/{A_cpu,C_cpu,GB_gpu,GC_gpu,nm_perlipatan_replikasi,rohira_perlipatan_replikasi}.csv`. Summary: `hasil40/ringkas.log`, `ringkas_*.csv`.
Counts: 38,620 simulation rows (A 30,640; C 4,800; GB 1,920; GC 1,260), NM 200 rows (50 replicates x H1/H2a x N0/V4), Rohira 240 rows (30 replicates x H1/H2a x 4 cells).

## Results by hypothesis (prediction vs result)
| Hypothesis | Prediction (pre-run) | Result | Status |
|---|---|---|---|
| H40-0 reproduction | required | CPU: maximum pooled-AUC difference 0.0 on 30,640 + 4,800 rows; GPU (soft): 0.0000 on 3,180 rows; NM: 0.0000 on 80 pairs; Rohira: 0.0 on 160 pairs | passed |
| H40-1 pooled-AUC bias | RF 0.35, LR 0.30, NM 0.45 supported | **Not supported for any classifier.** Bias = per-fold - pooled: LRpat -0.0011 [-0.0019, -0.0002]; RF +0.0010 [0.0007, 0.0013]; XGB +0.0010; MLP +0.0019; LSTM -0.0005; NM-N0 H1 +0.0024 [-0.0024, 0.0072]. Mean per fold 0.4906-0.5038, all within 0.5 +- 0.010 (XGB lowest, 0.4906). Bias was detected for RF/XGB/MLP but <= 0.002, below the 0.003 threshold. | prediction missed |
| H40-2 per-fold R1/R2 | >= 90% of cells do not change (0.75); changes in the category "uncertain" (0.70) | R1: 7 of 430 cells changed (1.6%); R2 (H1): 6 of 366 (1.6%). All seven and all six changes involve "uncertain". | prediction met |
| H40-3 LODO diagnostics | skew+Dq05 beats kurtosis 0.35 | **Not supported**: out-of-sample R2 of M0 -0.270, M1 -21.688, dR2 -21.4 (CI [-295, -1.8]). Kurtosis (M0) alone does not generalize across distribution families (R2 < 0). | prediction missed |
| H40-3 cusp | 0.25 | Descriptive only: cusp = 1.0008 (normal), 1.0058 (t3), 1.0090 (chi2): does not discriminate. RF symlog delta (fold_mean): normal +0.0125, chi2 +0.0159, t3 +0.0027; skew 0.048/0.580/0.267 (not monotone). The symlog effect on normal features is real (stage C replicates on the per-fold metric) but is not explained by the proposed diagnostics. | not supported |

## Additional findings (not registered as hypotheses; reported as post hoc/descriptive)
1. **The NM baseline < 0.5 of round 38 is sampling variation**: N0 H1 pooled 0.4896 on the first 20 replicates vs 0.5093 on the next 30 replicates (50 replicates: 0.5014, SE 0.006). This does not explain the Tang baseline (0.477-0.490), which was not repeated in round 40 (stage E was not run).
2. **NM-V4 (50 replicates)**: H1 difference (V4 - N0) per fold +0.0191 (SE 0.0061, t-test p 0.003; first 20 replicates +0.0261, next 30 +0.0144); pooled +0.0162 (SE 0.0059); R2 "uncertain" on both metrics. R1 V4 H1: pooled 0.5176 "no evidence", per fold 0.5229 "uncertain". H2a: V4 0.5049 (per fold), no evidence. **NM-V4 still does not pass fully**: there is a small covariate use (about +0.02 AUC) that cannot be ruled out; the manuscript decision to report F2 without the NM adjustment as a limit remains appropriate.
3. **Rohira RF-logpat vs RF-none (30 replicates)**: H1 difference +0.0058 (SE 0.0049), R2 "no evidence"; per-fold R1 H1 0.5078 and H2a 0.5193 "no evidence". RF-none H2a 0.5219 "uncertain" (legitimate era signal, not leakage).
4. **Pooled bias on real EEG features (post hoc, t-test over replicates)**: LR-none H1 per fold - pooled = +0.0068 (SE 0.0017, p 0.0005); RF-none +0.0010 (p 0.024); LR-pat +0.0003 (p 0.85); RF-logpat +0.0013 (p 0.006). The pooled AUC is indeed lower than the per-fold AUC for LR without adjustment in the real data, but its size is < 0.01.
5. **The leakage map holds on the per-fold metric** (H1, s > 0, per-patient adjustment, RF/XGB/MLP/LSTM; per-fold delta): rank-normal mean -0.0006, max +0.0188 (lognormal MLP, 20 replicates, "uncertain"), 0% "leakage", 95.6% "no evidence"; raw mean +0.0654 (60% "leakage"); symlog +0.0394 (37%); symlogc +0.0311 (29%); log +0.0016 (0%).
6. **Post hoc leave-one-cell-out (not the registered LODO; 143 correlated units)**: out-of-sample R2 of M0 0.325; M0+skew 0.371; M0+Dq05 0.406; M1 (all) 0.207; M0+cusp -0.267. The small gain from skew/Dq05 does not meet the registered criterion and must not be reported as a demonstrated diagnostic.

## Implications for the manuscript
- R1/R2 are reported on the mean per-fold AUC as the primary metric and the pooled AUC as secondary; the decisions are identical in 98.4% of cells.
- The claim "the network baseline < 0.5 because of pooling" must be dropped (H40-1 not supported). The claim "kurtosis/skewness/cusp explains the mechanism" must not be made; what remains is an empirical description of the leakage map (depends on distribution, classifier, label-covariate strength, transform).
- Pipeline status: NM-V4 does not pass fully (R2 H1 +0.019; per-fold R1 0.523 "uncertain"); Rohira log + per-patient adjustment passes; stage E (Tang) was not done.
- Limit: simulation + one EEG set; 39+1 rounds on the same 200 patients.
